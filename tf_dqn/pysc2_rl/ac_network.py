import tensorflow as tf
import numpy as np
import math

from tf_dqn.common import network_utils
from . import pysc2_common_net_funcs


class ACNetwork:
    def __init__(
            self,
            config
    ):
        self._config = config

        if config['network_structure']['shared_actor_critic_net']:
            if config['reg_type'] == 'l1':
                self._regularizer = tf.contrib.layers.l1_regularizer(scale=config['reg_scale'])
            elif config['reg_type'] == 'l2':
                self._regularizer = tf.contrib.layers.l2_regularizer(scale=config['reg_scale'])
        else:
            if config['reg_type'] == 'l1':
                self._actor_regularizer = tf.contrib.layers.l1_regularizer(scale=config['actor_reg_scale'])
                self._critic_regularizer = tf.contrib.layers.l1_regularizer(scale=config['critic_reg_scale'])
            elif config['reg_type'] == 'l2':
                self._actor_regularizer = tf.contrib.layers.l2_regularizer(scale=config['actor_reg_scale'])
                self._critic_regularizer = tf.contrib.layers.l2_regularizer(scale=config['critic_reg_scale'])

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
        self._td_targets = None
        self._training = None

        # the output operations

        # now setup the model
        self._define_model()

    def _get_networks(self, inputs, use_histograms=False):
        screen_features = pysc2_common_net_funcs.preprocess_state_input(inputs, self._config)
        actor_input = screen_features
        critic_input = screen_features

        # optional shared part of actor and critic
        if self._config['network_structure']['shared_actor_critic_net']:
            with tf.variable_scope('shared_actor_critic_net'):
                shared_ac = network_utils.get_layers(
                    screen_features,
                    self._config['network_structure']['shared_ac_network'],
                    self._config['network_structure']['default_activation'],
                    self._training,
                    use_histograms=use_histograms
                )
            if self._config['network_structure']['scale_gradients_at_shared_ac_split']:
                with tf.variable_scope('shared_ac_scale'):
                    # scale the gradients entering last shared layer, as in original Dueling DQN paper
                    scale = 1 / math.sqrt(2)
                    shared_ac = (1 - scale) * tf.stop_gradient(shared_ac) + scale * shared_ac
            actor_input = shared_ac
            critic_input = shared_ac

        # critic net
        reg = None
        if not self._config['network_structure']['shared_actor_critic_net']:
            reg = self._critic_regularizer
        with tf.variable_scope('critic_net', regularizer=reg):
            fc_critic = network_utils.get_layers(
                critic_input,
                self._config['network_structure']['critic_network'],
                self._config['network_structure']['default_activation'],
                self._training,
                use_histograms=use_histograms
            )
            value = tf.layers.dense(
                fc_critic,
                1,
                activation=None,
                kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
                name='value'
            )
            value = tf.squeeze(value, axis=-1)

        # actor net
        reg = None
        if not self._config['network_structure']['shared_actor_critic_net']:
            reg = self._actor_regularizer
        with tf.variable_scope('actor_net', regularizer=reg):
            with tf.variable_scope('shared_spatial_network'):
                shared_spatial_net = network_utils.get_layers(
                    actor_input,
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
            action_probs = {}
            # action choices
            action_choices = {}
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
                        action_probs[c] = tf.nn.softmax(component_streams[c])
                    if self._use_histograms:
                        tf.summary.histogram('advantage_' + c, component_streams[c])

                    # filter out actions ('function') that are illegal for this state
                    if c == 'function':
                        with tf.variable_scope('available_actions_mask'):
                            # available actions mask; avoids using negative infinity, and is the right size
                            all_nans = action_probs['function'] * np.NaN
                            action_probs['function'] = tf.where(inputs['available_actions'], action_probs['function'], all_nans)

                    # same as np.random_choice() - chooses according to probabilities. Needs log probs.
                    action_choices[c] = tf.multinomial(tf.log(action_probs[c]), 1)
                    action_choices[c] = tf.squeeze(action_choices[c], axis=-1)

                    if self._config['network_structure']['use_stream_outputs_as_inputs_to_other_streams']:
                        with tf.variable_scope('stream_action_one_hot'):
                            found_dependency = False
                            for stream, dependencies in self._config['network_structure']['stream_dependencies'].items():
                                if self._action_components[stream] and c in dependencies:
                                    found_dependency = True
                                    break
                            if found_dependency:
                                if c == 'screen':
                                    # special handling for screen->screen2 only
                                    action_one_hot = tf.one_hot(action_choices[c], num_options[c])
                                    action_one_hot = tf.reshape(action_one_hot, [-1, self._config['env']['screen_size'], self._config['env']['screen_size'], 1])
                                    component_one_hots_or_embeddings[c] = tf.stop_gradient(action_one_hot)
                                if num_options[c] <= 10:
                                    action_one_hot = tf.one_hot(action_choices[c], num_options[c])
                                    # argmax should be non-differentiable but just to remind myself use stop_gradient
                                    component_one_hots_or_embeddings[c] = tf.stop_gradient(action_one_hot)
                                else:
                                    component_one_hots_or_embeddings[c] = tf.keras.layers.Embedding(
                                        input_dim=num_options[c],
                                        output_dim=math.ceil(num_options[c] ** (1 / 4.0))
                                    )(action_choices[c])

        # return actor action probabilities and choices, and critic value
        return component_streams, action_choices, value

    def _define_model(self):
        # placeholders for (s, v(s'), r, terminal)
        # P(a|s) generated from s
        with tf.variable_scope('states_placeholders'):
            self._states = pysc2_common_net_funcs.get_state_placeholder(self._config)
        with tf.variable_scope('action_placeholders'):
            self._actions = {}
            for name, using in self._action_components.items():
                if using:
                    self._actions[name] = tf.placeholder(shape=[None, ], dtype=tf.int32, name=name)
        self._td_targets = tf.placeholder(shape=[None, ], dtype=tf.float32, name='td_target')
        self._training = tf.placeholder(shape=[], dtype=tf.bool, name='training_placeholder')

        # actor and critic nets
        reg = None
        if self._config['network_structure']['shared_actor_critic_net']:
            reg = self._regularizer
        with tf.variable_scope('networks', regularizer=reg):
            self._action_logits, self._action_choices, self._value = self._get_networks(self._states, self._use_histograms)

        # one hot the actions from experiences
        with tf.variable_scope('action_one_hot'):
            action_one_hot = pysc2_common_net_funcs.get_action_one_hot(self._actions, self._config)

        # these mask out the arguments that aren't used for the selected function from the loss calculation
        with tf.variable_scope('argument_masks'):
            argument_masks = pysc2_common_net_funcs.get_argument_masks(self._config)

        with tf.variable_scope('training'):
            critic_loss = tf.losses.huber_loss(self._td_targets, self._value)
            # critic_loss = tf.reduce_mean(tf.square(td_errors))
            td_errors = self._td_targets - self._value
            tf.summary.scalar('mean_td_error', tf.reduce_mean(td_errors))
            tf.summary.scalar('critic_loss', critic_loss)

            # calc actor error
            actor_losses = []
            td_errors = tf.stop_gradient(td_errors)

            for name in self._actions:
                # argument mask is scalar 1 if this argument is used for the transition action, 0 otherwise
                argument_mask = tf.reduce_max(action_one_hot['function'] * argument_masks[name], axis=-1)
                log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._action_logits[name], labels=self._actions[name])
                tf.summary.scalar('mean_actor_logprob_' + name, tf.reduce_mean(log_prob))
                loss = argument_mask * td_errors * log_prob
                actor_losses.append(loss)
            # add up component log-probs
            stacked = tf.stack(actor_losses, axis=1)
            actor_loss_summed = tf.reduce_sum(stacked, axis=-1)
            # take mean of losses across batch
            actor_loss = tf.reduce_mean(actor_loss_summed)
            tf.summary.scalar('actor_loss', actor_loss)

            # regularization loss and tensorboard summaries
            critic_reg_loss = tf.losses.get_regularization_loss('networks/critic_net')
            actor_reg_loss = tf.losses.get_regularization_loss('networks/actor_net')
            tf.summary.scalar('critic_reg_loss', critic_reg_loss)
            tf.summary.scalar('actor_reg_loss', actor_reg_loss)
            if self._config['network_structure']['shared_actor_critic_net']:
                shared_reg_loss = tf.losses.get_regularization_loss('networks/shared_actor_critic_net')
                total_reg_loss = shared_reg_loss + critic_reg_loss + actor_reg_loss
                tf.summary.scalar('shared_reg_loss', shared_reg_loss)
                tf.summary.scalar('reg_loss', total_reg_loss)

        shared_lr = self._config['shared_learning_rate']
        actor_lr = self._config['actor_learning_rate']
        critic_lr = self._config['critic_learning_rate']
        self._global_step = tf.placeholder(shape=[], dtype=tf.int32, name='global_step')
        if self._config['learning_rate_decay_method'] == 'exponential':
            if self._config['network_structure']['shared_actor_critic_net']:
                shared_lr = tf.train.exponential_decay(self._config['shared_learning_rate'], self._global_step, self._config['learning_rate_decay_steps'], self._config['learning_rate_decay_param'])
            else:
                actor_lr = tf.train.exponential_decay(self._config['actor_learning_rate'], self._global_step, self._config['learning_rate_decay_steps'], self._config['learning_rate_decay_param'])
                critic_lr = tf.train.exponential_decay(self._config['critic_learning_rate'], self._global_step, self._config['learning_rate_decay_steps'], self._config['learning_rate_decay_param'])
        elif self._config['learning_rate_decay_method'] == 'polynomial':
            if self._config['network_structure']['shared_actor_critic_net']:
                shared_lr = tf.train.polynomial_decay(self._config['shared_learning_rate'], self._global_step, self._config['learning_rate_decay_steps'], self._config['learning_rate_decay_param'])
            else:
                actor_lr = tf.train.polynomial_decay(self._config['actor_learning_rate'], self._global_step,                                                       self._config['learning_rate_decay_steps'], self._config['learning_rate_decay_param'])
                critic_lr = tf.train.polynomial_decay(self._config['critic_learning_rate'], self._global_step,                                                       self._config['learning_rate_decay_steps'], self._config['learning_rate_decay_param'])

        # must run this op to do batch norm
        self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if self._config['network_structure']['shared_actor_critic_net']:
            optimizer = tf.train.AdamOptimizer(learning_rate=shared_lr).minimize(actor_loss + critic_loss + total_reg_loss)
            self._optimizers = tf.group([optimizer, self._update_ops])
        else:
            critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr).minimize(critic_loss + critic_reg_loss)
            actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr).minimize(actor_loss + actor_reg_loss)
            self._optimizers = tf.group([critic_optimizer, actor_optimizer, self._update_ops])

        # tensorboard summaries
        self._train_summaries = tf.summary.merge_all(scope='training|networks/actor_net/advantage')
        if self._use_histograms:
            self._weight_summaries = tf.summary.merge_all(scope='networks/(actor_net/.*(_network|_stream)|shared_actor_critic_net|critic_net)')

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

    def episode_summary(self, sess, score, avg_score_all_episodes, win, avg_win):
        return sess.run(
            self._episode_summaries,
            feed_dict={
                self._episode_score: score,
                self._avg_episode_score: avg_score_all_episodes,
                self._episode_win: win,
                self._avg_episode_win: avg_win
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

    def predict_one(self, sess, state):
        feed_dict = {self._training: False}
        for name in self._states:
            # newaxis adds a new dimension of length 1 at the beginning (the batch size)
            feed_dict[self._states[name]] = np.expand_dims(state[name], axis=0)
        summaries, action_choices, state_value = sess.run([self._predict_summaries, self._action_choices, self._value], feed_dict=feed_dict)
        return summaries, action_choices, state_value

    def train_batch(self, sess, global_step, states, actions, td_targets):
        # need batch size to reshape actions
        batch_size = td_targets.shape[0]

        # everything else is a dictionary, so we need to loop through them
        feed_dict = {
            self._td_targets: td_targets,
            self._global_step: global_step,
            self._training: True
        }

        for name, _ in self._states.items():
            feed_dict[self._states[name]] = states[name]

        for name, using in self._action_components.items():
            if using:
                feed_dict[self._actions[name]] = actions[name].reshape(batch_size)

        if self._use_histograms:
            _, *summaries = sess.run([self._optimizers, self._train_summaries, self._weight_summaries], feed_dict=feed_dict)
        else:
            _, *summaries = sess.run([self._optimizers, self._train_summaries], feed_dict=feed_dict)

        return summaries

