import tensorflow as tf
import numpy as np
import math

from tf_dqn.common import network_utils
from . import pysc2_common_net_funcs


class SC2Network:
    def __init__(
            self,
            config
    ):
        self._config = config
        self._learning_rate = config['learning_rate']

        if config['reg_type'] == 'l1':
            self._regularizer = tf.contrib.layers.l1_regularizer(scale=config['reg_scale'])
        elif config['reg_type'] == 'l2':
            self._regularizer = tf.contrib.layers.l2_regularizer(scale=config['reg_scale'])

        # these are computed at runtime (in pysc2_runner.py), and not manually set in the config
        self._action_components = config['env']['computed_action_components']
        self._action_list = config['env']['computed_action_list']
        self._num_control_groups = config['env']['num_control_groups']

        self._use_histograms = False
        if 'use_histograms' in config['network_structure']:
            self._use_histograms = config['network_structure']['use_histograms']

        # define the placeholders
        self._global_step = None
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._terminal = None
        self._per_weights = None
        self._training = None

        # the output operations
        self._q = None
        self._actions_selected_by_q = None
        self._td_abs = None
        self._optimizer = None
        self.var_init = None
        self.saver = None

        # now setup the model
        self._define_model()

    def _get_network(self, inputs, use_histograms=False):
        screen = pysc2_common_net_funcs.preprocess_state_input(inputs, self._config)

        with tf.variable_scope('shared_spatial_network'):
            shared_spatial_net = network_utils.get_layers(
                screen,
                self._config['network_structure']['shared_spatial_network'],
                self._config['network_structure']['default_activation'],
                self._training,
                use_histograms=use_histograms
            )

        if self._config['network_structure']['scale_gradients_at_shared_spatial_split']:
            with tf.variable_scope('spatial_gradient_scale'):
                # scale because multiple action component streams are meeting here
                # (always one more branch than number of spatial components)
                spatial_count = 1
                for name, using in self._action_components.items():
                    if using and name in pysc2_common_net_funcs.spatial_components:
                        spatial_count += 1
                scale = 1 / spatial_count
                shared_spatial_net = (1 - scale) * tf.stop_gradient(shared_spatial_net) + scale * shared_spatial_net

        if self._config['dueling_network']:
            with tf.variable_scope('dueling_gradient_scale'):
                # scale the gradients entering last shared layer, as in original Dueling DQN paper
                scale = 1 / math.sqrt(2)
                shared_spatial_net = (1 - scale) * tf.stop_gradient(shared_spatial_net) + scale * shared_spatial_net

        # for dueling net, split here
        if self._config['dueling_network']:
            with tf.variable_scope('value_network'):
                fc_value = network_utils.get_layers(
                    shared_spatial_net,
                    self._config['network_structure']['value_network'],
                    self._config['network_structure']['default_activation'],
                    self._training,
                    use_histograms=use_histograms
                )
                value = tf.layers.dense(
                    fc_value,
                    1,
                    activation=None,
                    kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
                    name='value'
                )
        else:
            # returning this from the function for debugging purposes, so need it to exist if not using dueling net
            value = None

        with tf.variable_scope('shared_non_spatial_network'):
            shared_non_spatial = network_utils.get_layers(
                shared_spatial_net,
                self._config['network_structure']['shared_non_spatial_network'],
                self._config['network_structure']['default_activation'],
                self._training,
                use_histograms=use_histograms
            )

        if self._config['network_structure']['scale_gradients_at_shared_non_spatial_split']:
            with tf.variable_scope('non_spatial_gradient_scale'):
                # scale because multiple action component streams are meeting here
                non_spatial_count = 0
                for name, using in self._action_components.items():
                    if using and name not in pysc2_common_net_funcs.spatial_components:
                        non_spatial_count += 1
                scale = 1 / non_spatial_count
                shared_non_spatial = (1 - scale) * tf.stop_gradient(shared_non_spatial) + scale * shared_non_spatial

        num_options = pysc2_common_net_funcs.get_num_options_per_function(self._config)

        # create each component stream
        component_streams = {}
        # final q vals with value added
        action_q_vals = {}
        # if another stream requires the output of another stream
        component_one_hots_or_embeddings = {}
        for c in pysc2_common_net_funcs.component_order:
            # are we using this component?
            if self._action_components[c]:
                with tf.variable_scope(c + '_stream'):
                    stream_input = shared_non_spatial
                    if c in pysc2_common_net_funcs.spatial_components:
                        stream_input = shared_spatial_net

                    # optionally one stream of fully connected layers per component
                    spec = self._config['network_structure']['component_stream_default']
                    if c in self._config['network_structure']['component_stream_specs']:
                        spec = self._config['network_structure']['component_stream_specs'][c]

                    # optionally feed one hot OR embedded versions of earlier stream outputs to this stream
                    dependencies = None
                    if self._config['network_structure']['use_stream_outputs_as_inputs_to_other_streams']:
                        if c in self._config['network_structure']['stream_dependencies']:
                            dependencies = []
                            for d in self._config['network_structure']['stream_dependencies'][c]:
                                dependencies.append(component_one_hots_or_embeddings[d])

                    component_stream = network_utils.get_layers(
                        stream_input,
                        spec,
                        self._config['network_structure']['default_activation'],
                        self._training,
                        dependencies,
                        use_histograms=use_histograms
                    )

                    if c not in pysc2_common_net_funcs.spatial_components or self._config['network_structure']['end_spatial_streams_with_dense_instead_of_flatten']:
                        # make a dense layer with width equal to number of possible actions
                        dense = tf.layers.Dense(
                            num_options[c],
                            name=c
                        )
                        component_streams[c] = dense(component_stream)
                        if self._use_histograms:
                            weights = dense.kernel
                            bias = dense.bias
                            name = 'final_dense_' + c + '_'
                            tf.summary.histogram(name + 'weights', weights)
                            tf.summary.histogram(name + 'bias', bias)
                    else:
                        # flatten a conv output
                        component_streams[c] = tf.reshape(component_stream, [-1, num_options[c]], name=c)
                if self._use_histograms:
                    tf.summary.histogram('advantage_' + c, component_streams[c])
                if self._config['dueling_network']:
                    # action_q_vals is A(s,a), value is V(s)
                    # Q(s,a) = V(s) + A(s,a) - 1/|A| * SUM_a(A(s,a))
                    with tf.variable_scope('q_vals'):
                        advantage = component_streams[c]
                        action_q_vals[c] = tf.add(value, (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)), name=name)

                else:
                    action_q_vals[c] = component_streams[c]

                # filter out actions ('function') that are illegal for this state
                if c == 'function':
                    with tf.variable_scope('available_actions_mask'):
                        # available actions mask; avoids using negative infinity, and is the right size
                        action_neg_inf_q_vals = action_q_vals['function'] * 0 - 1000000
                        action_q_vals['function'] = tf.where(inputs['available_actions'], action_q_vals['function'], action_neg_inf_q_vals)

                if self._config['network_structure']['use_stream_outputs_as_inputs_to_other_streams']:
                    with tf.variable_scope('stream_action_one_hot'):
                        found_dependency = False
                        for stream, dependencies in self._config['network_structure']['stream_dependencies'].items():
                            if self._action_components[stream] and c in dependencies:
                                found_dependency = True
                                break
                        if found_dependency:

                            action_index = tf.math.argmax(action_q_vals[c], axis=-1)
                            if c == 'screen':
                                # special handling for screen->screen2 only
                                action_one_hot = tf.one_hot(action_index, num_options[c])
                                action_one_hot = tf.reshape(action_one_hot, [-1, self._config['env']['screen_size'], self._config['env']['screen_size'], 1])
                                component_one_hots_or_embeddings[c] = tf.stop_gradient(action_one_hot)
                            if num_options[c] <= 10:
                                action_one_hot = tf.one_hot(action_index, num_options[c])
                                # argmax should be non-differentiable but just to remind myself use stop_gradient
                                component_one_hots_or_embeddings[c] = tf.stop_gradient(action_one_hot)
                            else:
                                component_one_hots_or_embeddings[c] = tf.keras.layers.Embedding(
                                    input_dim=num_options[c],
                                    output_dim=math.ceil(num_options[c] ** (1 / 4.0))
                                )(action_index)

        # return action_q_vals
        return action_q_vals, value, component_streams

    def _define_model(self):
        # placeholders for (s, a, s', r, terminal)
        with tf.variable_scope('states_placeholders'):
            self._states = pysc2_common_net_funcs.get_state_placeholder(self._config)
        with tf.variable_scope('action_placeholders'):
            self._actions = {}
            for name, using in self._action_components.items():
                if using:
                    self._actions[name] = tf.placeholder(shape=[None, ], dtype=tf.int32, name=name)
        with tf.variable_scope('next_states_placeholders'):
            self._next_states = pysc2_common_net_funcs.get_state_placeholder(self._config)
        self._rewards = tf.placeholder(shape=[None, ], dtype=tf.float32, name='reward_placeholder')
        self._terminal = tf.placeholder(shape=[None, ], dtype=tf.float32, name='terminal_placeholder')
        self._per_weights = tf.placeholder(shape=[None, ], dtype=tf.float32, name='per_weights_placeholder')
        self._training = tf.placeholder(shape=[], dtype=tf.bool, name='training_placeholder')

        # primary and target Q nets
        with tf.variable_scope('Q_primary', regularizer=self._regularizer):
            # self._q, self._q_weights = self._get_network(self._states)
            self._q, self._value, self._advantage = self._get_network(self._states, self._use_histograms)
        with tf.variable_scope('Q_target'):
            # self._q_target, self._q_target_weights = self._get_network(self._next_states)
            self._q_target, _, _ = self._get_network(self._next_states)
        # used for copying parameters from primary to target net
        self._q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_primary")
        self._q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_target")

        if self._config['double_DQN']:
            # action selected by q for Double DQN
            with tf.variable_scope('actions_selected_by_q'):
                self._actions_selected_by_q = {}
                for name, q_vals in self._q.items():
                    self._actions_selected_by_q[name] = tf.argmax(q_vals, axis=-1, name='name')

            # next action placeholders for Double DQN
            with tf.variable_scope('action_next_placeholders'):
                self._actions_next = {}
                for name, val in self._q.items():
                    if val is not None:
                        self._actions_next[name] = tf.placeholder(shape=[None, ], dtype=tf.int32, name=name)

        # one hot the actions from experiences
        with tf.variable_scope('action_one_hot'):
            action_one_hot = pysc2_common_net_funcs.get_action_one_hot(self._actions, self._config)

        # these mask out the arguments that aren't used for the selected function from the loss calculation
        with tf.variable_scope('argument_masks'):
            argument_masks = pysc2_common_net_funcs.get_argument_masks(self._config)
            # y_masks = {}
            # for name in self._actions_next:
            #     y_masks[name] = tf.reduce_max(next_states_action_one_hot['function'] * argument_masks[name], axis=-1)

        # The q value by the primary Q network for the actual actions taken in an experience
        with tf.variable_scope('prediction'):
            training_action_q = {}
            for name, q_vals in self._q.items():
                training_action_q[name] = tf.reduce_sum(q_vals * action_one_hot[name], reduction_indices=-1, name=name)

        # one hot the actions from next states
        with tf.variable_scope('next_states_action_one_hot'):
            if self._config['double_DQN']:
                # in DDQN, actions have been chosen by primary network in a previous pass
                next_states_action_one_hot = pysc2_common_net_funcs.get_action_one_hot(self._actions_next, self._config)
            else:
                # in DQN, choosing actions based on target network qvals for next states
                actions_next = {}
                for name, q_vals in self._q_target.items():
                    actions_next[name] = tf.argmax(q_vals, axis=1)
                next_states_action_one_hot = pysc2_common_net_funcs.get_action_one_hot(actions_next, self._config)

        # target Q(s,a)
        with tf.variable_scope('y'):
            # holds q val of each action component in target net
            y_components = {}
            if self._config['double_DQN']:
                # Double DQN uses target network Q val of primary network next action
                for name, action in self._actions_next.items():
                    # creates tensor with [0, 1, 2, ...] with length of ???
                    row = tf.range(tf.shape(action)[0])
                    combined = tf.stack([row, action], axis=1)
                    max_q_next = tf.gather_nd(self._q_target[name], combined)
                    y_components[name] = (1 - self._terminal) * (self._config['discount'] ** self._config['bootstrapping_steps']) * max_q_next
            else:
                # DQN uses target network max Q val
                for name, q_vals in self._q_target.items():
                    max_q_next_by_target = tf.reduce_max(q_vals, axis=-1, name=name)
                    y_components[name] = (1 - self._terminal) * (self._config['discount'] ** self._config['bootstrapping_steps']) * max_q_next_by_target
            # q val of components not used for the action actually chosen in 'function' will be set to zero
            y_components_masked = []
            # get vector of 0s of correct length
            num_components = self._rewards * 0
            for name in y_components:
                argument_mask = tf.reduce_max(next_states_action_one_hot['function'] * argument_masks[name], axis=-1)
                # keep track of number of components used in this action
                num_components = num_components + argument_mask
                y_components_masked.append(y_components[name] * argument_mask)
            y_parts_stacked = tf.stack(y_components_masked, axis=1)
            # single y value is: r + avg(q vals of components used)
            # stop gradient here because we don't back prop on target net
            y = tf.stop_gradient(self._rewards + (tf.reduce_sum(y_parts_stacked, axis=1) / num_components))

        if self._config['loss_formula'] == 'avg_y_compared_to_avg_prediction':
            with tf.variable_scope('predict_q'):
                prediction_components_masked = []
                # get vector of 0s of correct length
                num_components = self._rewards * 0
                for name in training_action_q:
                    argument_mask = tf.reduce_max(next_states_action_one_hot['function'] * argument_masks[name], axis=-1)
                    # keep track of number of components used in this action
                    num_components = num_components + argument_mask
                    prediction_components_masked.append(training_action_q[name] * argument_mask)
                prediction_parts_stacked = tf.stack(prediction_components_masked, axis=1)
                prediction_q = tf.reduce_sum(prediction_parts_stacked, axis=1) / num_components

        # calculate losses (average of y compared to each component of prediction action)
        with tf.variable_scope('losses'):
            losses = []
            td = []
            num_components = self._rewards * 0
            if self._config['loss_formula'] == 'avg_y_compared_to_components':
                for name in training_action_q:
                    # argument mask is scalar 1 if this argument is used for the transition action, 0 otherwise
                    argument_mask = tf.reduce_max(action_one_hot['function'] * argument_masks[name], axis=-1)
                    num_components = num_components + argument_mask
                    training_action_q_masked = training_action_q[name] * argument_mask
                    y_masked = y * argument_mask
                    # we compare the q value of each component to the target y; y is masked if training q is masked
                    loss = tf.losses.huber_loss(training_action_q_masked, y_masked, weights=self._per_weights)
                    td.append(tf.abs(training_action_q_masked - y_masked))
                    losses.append(loss)
            elif self._config['loss_formula'] == 'avg_y_compared_to_avg_prediction':
                # we compare the avg q value of the prediction components to the target y
                loss = tf.losses.huber_loss(prediction_q, y, weights=self._per_weights)
                td.append(tf.abs(prediction_q - y))
                losses.append(loss)
            elif self._config['loss_formula'] == 'pairwise_component_comparison':
                for name in training_action_q:
                    # argument mask is scalar 1 if this argument is used for the transition action, 0 otherwise
                    argument_mask = tf.reduce_max(action_one_hot['function'] * argument_masks[name], axis=-1)
                    num_components = num_components + argument_mask
                    training_action_q_masked = training_action_q[name] * argument_mask
                    y_masked = tf.stop_gradient((self._rewards + y_components[name]) * argument_mask)
                    # we compare the q value of each component to the y value for the same component,
                    # regardless if that component is used in the y action
                    loss = tf.losses.huber_loss(training_action_q_masked, y_masked, weights=self._per_weights)
                    td.append(tf.abs(training_action_q_masked - y_masked))
                    losses.append(loss)
            training_losses = tf.reduce_sum(tf.stack(losses), name='training_losses')
            reg_loss = tf.losses.get_regularization_loss()
            final_loss = training_losses + reg_loss
            tf.summary.scalar('training_loss', training_losses)
            tf.summary.scalar('regularization_loss', reg_loss)
            if self._config['loss_formula'] == 'avg_y_compared_to_avg_prediction' or not self._config['avg_component_tds']:
                # for loss forumals where we're adding up different components
                # we have the option to divide by number of components
                self._td_abs = tf.reduce_sum(tf.stack(td, axis=1), axis=1)
            else:
                self._td_abs = tf.reduce_sum(tf.stack(td, axis=1), axis=1) / num_components

        self._global_step = tf.placeholder(shape=[], dtype=tf.int32, name='global_step')
        if self._config['learning_rate_decay_method'] == 'exponential':
            lr = tf.train.exponential_decay(self._learning_rate, self._global_step, self._config['learning_rate_decay_steps'], self._config['learning_rate_decay_param'])
        elif self._config['learning_rate_decay_method'] == 'polynomial':
            lr = tf.train.polynomial_decay(self._learning_rate, self._global_step, self._config['learning_rate_decay_steps'], self._config['learning_rate_decay_param'])
        else:
            lr = self._learning_rate

        # must run this op to do batch norm
        self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(final_loss)
        self._optimizer = tf.group([self._optimizer, self._update_ops])

        # tensorboard summaries
        self._train_summaries = tf.summary.merge_all(scope='losses|Q_primary/advantage')
        if self._use_histograms:
            self._weight_summaries = tf.summary.merge_all(scope='Q_primary/.*_network|Q_primary/.*_stream')

        with tf.variable_scope('episode_summaries'):
            # score here might be a sparse win/loss +1/-1, or it might be a shaped reward signal
            self._episode_score = tf.placeholder(shape=[], dtype=tf.float32, name='episode_score')
            tf.summary.scalar('episode_score', self._episode_score)
            self._avg_episode_score = tf.placeholder(shape=[], dtype=tf.float32, name='avg_episode_score')
            tf.summary.scalar('avg_episode_score', self._avg_episode_score)
            # episode_win is always going to be 1/-1/0, for win/loss/draw
            self._episode_win = tf.placeholder(shape=[], dtype=tf.float32, name='episode_win')
            tf.summary.scalar('episode_win', self._episode_win)
            self._avg_episode_win = tf.placeholder(shape=[], dtype=tf.float32, name='avg_episode_win')
            tf.summary.scalar('avg_episode_win', self._avg_episode_win)

            self._epsilon = tf.placeholder(shape=[], dtype=tf.float32, name='episode_ending_epsilon')
            tf.summary.scalar('episode_ending_epsilon', self._epsilon)
        self._episode_summaries = tf.summary.merge_all(scope='episode_summaries')

        with tf.variable_scope('evaluation_episode_summaries'):
            # the same ones as above, but for evaluation episodes only
            self._eval_episode_score = tf.placeholder(shape=[], dtype=tf.float32, name='eval_episode_score')
            tf.summary.scalar('eval_episode_score', self._eval_episode_score)
            self._eval_avg_episode_score = tf.placeholder(shape=[], dtype=tf.float32, name='eval_avg_episode_score')
            tf.summary.scalar('eval_avg_episode_score', self._eval_avg_episode_score)
            self._eval_episode_win = tf.placeholder(shape=[], dtype=tf.float32, name='eval_episode_win')
            tf.summary.scalar('eval_episode_win', self._eval_episode_win)
            self._eval_avg_episode_win = tf.placeholder(shape=[], dtype=tf.float32, name='eval_avg_episode_win')
            tf.summary.scalar('eval_avg_episode_win', self._eval_avg_episode_win)
        self._eval_episode_summaries = tf.summary.merge_all(scope='evaluation_episode_summaries')

        with tf.variable_scope('predict_summaries'):
            # log average q-val of components used in action
            predict_actions = {}
            for name, q_vals in self._q.items():
                predict_actions[name] = tf.argmax(q_vals, axis=1)
            predict_action_one_hot = pysc2_common_net_funcs.get_action_one_hot(predict_actions, self._config)
            predict_q_vals = []
            count = tf.Variable(tf.zeros([], dtype=np.float32), trainable=False)
            for name, q_vals in self._q.items():
                argument_mask = tf.reduce_max(predict_action_one_hot['function'] * argument_masks[name], axis=-1)
                predict_action_q_masked = tf.reduce_max(q_vals) * argument_mask
                predict_q_vals.append(predict_action_q_masked)
                count = count + argument_mask
            predicted_q_val_avg = tf.reduce_sum(tf.stack(predict_q_vals)) / count
            tf.summary.scalar('predicted_q_val', tf.squeeze(predicted_q_val_avg))

            # also log state value if using dueling net
            if self._config['dueling_network']:
                tf.summary.scalar('state_value', tf.squeeze(self._value))
        self._predict_summaries = tf.summary.merge_all(scope='predict_summaries')

        # variable initializer
        self.var_init = tf.global_variables_initializer()

        # saver
        self.saver = tf.train.Saver(
            max_to_keep=self._config['model_checkpoint_max'],
            keep_checkpoint_every_n_hours=self._config['model_checkpoint_every_n_hours']
        )

        num_trainable_params = np.sum([np.prod([y.value for y in x.get_shape()]) for x in tf.all_variables()])
        print('Num trainable params (millions):', '{:.2f}'.format(num_trainable_params / 1e6))

    def episode_summary(self, sess, score, avg_score_all_episodes, win, avg_win, epsilon):
        return sess.run(
            self._episode_summaries,
            feed_dict={
                self._episode_score: score,
                self._avg_episode_score: avg_score_all_episodes,
                self._episode_win: win,
                self._avg_episode_win: avg_win,
                self._epsilon: epsilon
            }
        )

    def eval_episode_summary(self, sess, score, avg_score_all_episodes, win, avg_win):
        return sess.run(
            self._eval_episode_summaries,
            feed_dict={
                self._eval_episode_score: score,
                self._eval_avg_episode_score: avg_score_all_episodes,
                self._eval_episode_win: win,
                self._eval_avg_episode_win: avg_win
            }
        )

    def update_target_q_net(self, sess):
        sess.run([v_t.assign(v) for v_t, v in zip(self._q_target_vars, self._q_vars)])

    def predict_one(self, sess, state):
        feed_dict = {self._training: False}
        for name in self._states:
            # newaxis adds a new dimension of length 1 at the beginning (the batch size)
            feed_dict[self._states[name]] = np.expand_dims(state[name], axis=0)
        summaries, q, value, advantage = sess.run([self._predict_summaries, self._q, self._value, self._advantage], feed_dict=feed_dict)
        return summaries, q

    def train_batch(self, sess, global_step, states, actions, rewards, next_states, terminal, weights):
        # need batch size to reshape actions
        batch_size = actions['function'].shape[0]

        # everything else is a dictionary, so we need to loop through them
        feed_dict = {
            self._rewards: rewards,
            self._terminal: terminal,
            self._global_step: global_step,
            self._training: True
        }

        if weights is not None:
            feed_dict[self._per_weights] = weights
        else:
            feed_dict[self._per_weights] = np.ones(batch_size, dtype=np.float32)

        if self._config['double_DQN']:
            actions_next_feed_dict = {self._training: False}
            for name in self._states:
                actions_next_feed_dict[self._states[name]] = next_states[name]
            actions_next = sess.run(self._actions_selected_by_q, feed_dict=actions_next_feed_dict)
            for name, using in self._action_components.items():
                if using:
                    feed_dict[self._actions_next[name]] = actions_next[name].reshape(batch_size)

        for name, _ in self._states.items():
            feed_dict[self._states[name]] = states[name]
            feed_dict[self._next_states[name]] = next_states[name]

        for name, using in self._action_components.items():
            if using:
                feed_dict[self._actions[name]] = actions[name].reshape(batch_size)
        if self._use_histograms:
            td_abs, _, *summaries = sess.run([self._td_abs, self._optimizer, self._train_summaries, self._weight_summaries], feed_dict=feed_dict)
        else:
            td_abs, _, *summaries = sess.run(
                [self._td_abs, self._optimizer, self._train_summaries], feed_dict=feed_dict)

        return summaries, td_abs

