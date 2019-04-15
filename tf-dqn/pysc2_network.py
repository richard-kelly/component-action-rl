import tensorflow as tf
import numpy as np
import math

class SC2Network:
    def __init__(
            self,
            double_dqn,
            dueling,
            learning_rate,
            learning_decay_mode,
            learning_decay_steps,
            learning_decay_param,
            discount,
            max_checkpoints,
            checkpoint_hours,
            reg_type,
            reg_scale,
            environment_properties
    ):

        self._double_dqn = double_dqn
        self._dueling = dueling
        self._learning_rate = learning_rate
        self._learning_decay = learning_decay_mode
        self._learning_decay_steps = learning_decay_steps
        self._learning_decay_factor = learning_decay_param
        self._discount = discount
        self._max_checkpoints = max_checkpoints
        self._checkpoint_hours = checkpoint_hours

        if reg_type == 'l1':
            self._regularizer = tf.contrib.layers.l1_regularizer(scale=reg_scale)
        elif reg_type == 'l2':
            self._regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)

        self._screen_size = environment_properties['screen_size']
        self._action_components = environment_properties['action_components']
        self._action_list = environment_properties['action_list']

        # define the placeholders
        self._global_step = None
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._terminal = None

        # the output operations
        self._q = None
        self._actions_selected_by_q = None
        self._optimizer = None
        self.var_init = None
        self.saver = None

        # now setup the model
        self._define_model()

    def _get_network(self, inputs):
        # concat parts of input
        screen_player_relative_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['screen_player_relative'],
            num_classes=5
        )[:, :, :, 1:]

        screen_selected_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['screen_player_relative'],
            num_classes=2
        )[:, :, :, 1:]
        screen = tf.concat([screen_player_relative_one_hot, screen_selected_one_hot], axis=-1, name='screen_input')

        # begin shared conv layers
        conv1_spatial = tf.layers.conv2d(
            inputs=screen,
            filters=16,
            kernel_size=5,
            padding='same',
            name='conv1_spatial',
            activation=tf.nn.relu
        )

        conv2_spatial = tf.layers.conv2d(
            inputs=conv1_spatial,
            filters=32,
            kernel_size=3,
            padding='same',
            name='conv2_spatial',
            activation=tf.nn.relu
        )

        with tf.variable_scope('spatial_gradient_scale'):
            # scale because multiple action component streams are meeting here
            # TODO: come up with better scaling based on which action components are used in training.
            scale = 1 / math.sqrt(2)
            conv2_spatial = (1 - scale) * tf.stop_gradient(conv2_spatial) + scale * conv2_spatial

        # spatial policy splits off before max pooling
        max_pool = tf.layers.max_pooling2d(
            inputs=conv2_spatial,
            pool_size=3,
            strides=3,
            padding='valid',
            name='max_pool'
        )

        # MUST flatten conv or pooling layers before sending to dense layer
        non_spatial_flat = tf.reshape(
            max_pool,
            shape=[-1, int(self._screen_size * self._screen_size / 9 * 32)],
            name='conv2_spatial_flat'
        )

        if self._dueling:
            with tf.variable_scope('dueling_gradient_scale'):
                # scale the gradients entering last shared layer, as in original Dueling DQN paper
                scale = 1 / math.sqrt(2)
                non_spatial_flat = (1 - scale) * tf.stop_gradient(non_spatial_flat) + scale * non_spatial_flat

        # for dueling net, split here
        if self._dueling:
            fc_value1 = tf.layers.dense(
                non_spatial_flat,
                1024,
                activation=tf.nn.relu,
                kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
                name='fc_value_1'
            )
            fc_value2 = tf.layers.dense(
                fc_value1,
                1024,
                activation=tf.nn.relu,
                kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
                name='fc_value_2'
            )
            value = tf.layers.dense(
                fc_value2,
                1,
                activation=None,
                kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
                name='value'
            )

        fc_non_spatial_1 = tf.layers.dense(
            non_spatial_flat,
            1024,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
            name='fc_non_spatial_1'
        )

        fc_non_spatial_2 = tf.layers.dense(
            fc_non_spatial_1,
            1024,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
            name='fc_non_spatial_2'
        )

        with tf.variable_scope('non_spatial_gradient_scale'):
            # scale because multiple action component streams are meeting here
            # TODO: come up with better scaling based on which action components are used in training.
            scale = 1 / math.sqrt(2)
            fc_non_spatial_2 = (1 - scale) * tf.stop_gradient(fc_non_spatial_2) + scale * fc_non_spatial_2

        spatial_policy_1 = tf.layers.conv2d(
            inputs=conv2_spatial,
            filters=1,
            kernel_size=1,
            padding='same',
            name='spatial_policy_1'
        )

        if self._action_components['screen2']:
            spatial_policy_2 = tf.layers.conv2d(
                inputs=conv2_spatial,
                filters=1,
                kernel_size=1,
                padding='same',
                name='spatial_policy_2'
            )
        else:
            spatial_policy_2 = None

        n = self._screen_size
        comp = self._action_components
        action_q_vals = dict(
            function=tf.layers.dense(fc_non_spatial_2, 4, name='function'),
            screen=tf.reshape(spatial_policy_1, [-1, n * n], name='screen') if comp['screen'] else None,
            screen2=tf.reshape(spatial_policy_2, [-1, n * n], name='screen2') if comp['screen2'] else None,
            queued=tf.layers.dense(fc_non_spatial_2, 2, name='queued') if comp['queued'] else None,
            select_point_act=tf.layers.dense(fc_non_spatial_2, 4, name='select_point_act') if comp['select_point_act'] else None,
            select_add=tf.layers.dense(fc_non_spatial_2, 2, name='select_add') if comp['select_add'] else None
        )

        action_q_vals_filtered = {}
        for name, val in action_q_vals.items():
            if val is not None:
                action_q_vals_filtered[name] = val

        if self._dueling:
            # action_q_vals_filtered is A(s,a), value is V(s)
            # Q(s,a) = V(s) + A(s,a) - 1/|A| * SUM_a(A(s,a))
            with tf.variable_scope('q_vals'):
                for name, advantage in action_q_vals_filtered.items():
                    action_q_vals_filtered[name] = tf.add(value, (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)), name=name)

        with tf.variable_scope('available_actions_mask'):
            # available actions mask; avoids using negative infinity, and is the right size
            action_neg_inf_q_vals = action_q_vals_filtered['function'] * 0 - 1000000
            action_q_vals_filtered['function'] = tf.where(inputs['available_actions'], action_q_vals_filtered['function'], action_neg_inf_q_vals)

        return action_q_vals_filtered

    def _get_state_placeholder(self):
        return dict(
                screen_player_relative=tf.placeholder(
                    shape=[None, self._screen_size, self._screen_size],
                    dtype=tf.int32,
                    name='screen_player_relative'
                ),
                screen_selected=tf.placeholder(
                    shape=[None, self._screen_size, self._screen_size],
                    dtype=tf.int32,
                    name='screen_selected'
                ),
                available_actions=tf.placeholder(
                    shape=[None, len(self._action_list['function'])],
                    dtype=tf.bool,
                    name='available_actions'
                )
            )

    def _get_argument_masks(self):
        # these would have to change for different pysc2 action functions...
        # they should be generated by a function based on data in pysc2.lib.actions
        return dict(
                function=tf.constant([1, 1, 1, 1], dtype=tf.float32, name='function'),
                screen=tf.constant([0, 1, 1, 1], dtype=tf.float32, name='screen'),
                screen2=tf.constant([0, 0, 1, 0], dtype=tf.float32, name='screen2'),
                select_point_act=tf.constant([0, 1, 0, 0], dtype=tf.float32, name='select_point_act'),
                select_add=tf.constant([0, 0, 1, 0], dtype=tf.float32, name='select_add'),
                queued=tf.constant([0, 0, 0, 1], dtype=tf.float32, name='queued'),
            )

    def _get_action_one_hot(self, actions):
        # action components we are using.
        comp = self._action_components
        n = self._screen_size * self._screen_size
        # number of options for some function args hard coded here
        return dict(
            function=tf.one_hot(actions['function'], len(self._action_list['function']), 1.0, 0.0, name='function'),
            screen=tf.one_hot(actions['screen'], n, 1.0, 0.0, name='screen') if comp['screen'] else None,
            screen2=tf.one_hot(actions['screen2'], n, 1.0, 0.0, name='screen2') if comp['screen2'] else None,
            select_point_act=tf.one_hot(actions['select_point_act'], 4, 1.0, 0.0, name='select_point_act') if comp['select_point_act'] else None,
            select_add=tf.one_hot(actions['select_add'], 2, 1.0, 0.0, name='select_add') if comp['select_add'] else None,
            queued=tf.one_hot(actions['queued'], 2, 1.0, 0.0, name='queued') if comp['queued'] else None
        )

    def _define_model(self):
        # placeholders for (s, a, s', r, terminal)
        with tf.variable_scope('states_placeholders'):
            self._states = self._get_state_placeholder()
        with tf.variable_scope('action_placeholders'):
            self._actions = {}
            for name, using in self._action_components.items():
                if using:
                    self._actions[name] = tf.placeholder(shape=[None, ], dtype=tf.int32, name=name)
        with tf.variable_scope('next_states_placeholders'):
            self._next_states = self._get_state_placeholder()
        self._rewards = tf.placeholder(shape=[None, ], dtype=tf.float32, name='reward_placeholder')
        self._terminal = tf.placeholder(shape=[None, ], dtype=tf.float32, name='terminal_placeholder')

        # primary and target Q nets
        with tf.variable_scope('Q_primary', regularizer=self._regularizer):
            self._q = self._get_network(self._states)
        with tf.variable_scope('Q_target'):
            self._q_target = self._get_network(self._next_states)
        # used for copying parameters from primary to target net
        self._q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_primary")
        self._q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_target")

        if self._double_dqn:
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
            action_one_hot = self._get_action_one_hot(self._actions)

        # The q value by the primary Q network for the actual actions taken in an experience
        with tf.variable_scope('prediction'):
            training_action_q = {}
            for name, q_vals in self._q.items():
                training_action_q[name] = tf.reduce_sum(q_vals * action_one_hot[name], reduction_indices=-1, name=name)

        # one hot the actions from next states
        with tf.variable_scope('next_states_action_one_hot'):
            if self._double_dqn:
                # in DDQN, actions have been chosen by primary network in a previous pass
                next_states_action_one_hot = self._get_action_one_hot(self._actions_next)
            else:
                # in DQN, choosing actions based on target network qvals for next states
                actions_next = {}
                for name, q_vals in self._q_target.items():
                    actions_next[name] = tf.argmax(q_vals, axis=1)
                next_states_action_one_hot = self._get_action_one_hot(actions_next)

        # these mask out the arguments that aren't used for the selected function from the loss calculation
        with tf.variable_scope('argument_masks'):
            argument_masks = self._get_argument_masks()

        # target Q(s,a)
        with tf.variable_scope('y'):
            y_components = {}
            if self._double_dqn:
                # Double DQN uses target network Q val of primary network next action
                for name, action in self._actions_next.items():
                    row = tf.range(tf.shape(action)[0])
                    combined = tf.stack([row, action], axis=1)
                    max_q_next = tf.gather_nd(self._q_target[name], combined)
                    y_components[name] = (1 - self._terminal) * self._discount * max_q_next
            else:
                # DQN uses target network max Q val
                for name, q_vals in self._q_target.items():
                    max_q_next_by_target = tf.reduce_max(q_vals, axis=-1, name=name)
                    y_components[name] = (1 - self._terminal) * self._discount * max_q_next_by_target
            y_components_masked = []
            # get vector of 0s of correct length
            num_components = self._rewards * 0
            for name in y_components:
                argument_mask = tf.reduce_max(next_states_action_one_hot['function'] * argument_masks[name], axis=-1)
                # keep track of number of components used in this action
                num_components = num_components + argument_mask
                y_components_masked.append(y_components[name] * argument_mask)
            y_parts_stacked = tf.stack(y_components_masked, axis=1)
            y = tf.stop_gradient(self._rewards + tf.reduce_sum(y_parts_stacked, axis=1) / num_components)

        # calculate losses (average of y compared to each component of prediction action)
        with tf.variable_scope('losses'):
            losses = []
            for name in training_action_q:
                # argument mask is scalar 1 if this argument is used for the transition action, 0 otherwise
                argument_mask = tf.reduce_max(action_one_hot['function'] * argument_masks[name], axis=-1)
                training_action_q_masked = training_action_q[name] * argument_mask
                y_masked = y * argument_mask
                # we compare the q value of each component to the target y; y is masked if training q is masked
                loss = tf.losses.huber_loss(training_action_q_masked, y_masked)
                losses.append(loss)
            # TODO: This average is kind of meaningless, maybe it should be sum instead
            losses_avg = tf.reduce_mean(tf.stack(losses), name='losses_avg')
            reg_loss = tf.losses.get_regularization_loss()
            final_loss = losses_avg + reg_loss
            tf.summary.scalar('training_loss_avg', losses_avg)
            tf.summary.scalar('training_loss_reg', reg_loss)

        self._global_step = tf.placeholder(shape=[], dtype=tf.int32, name='global_step')
        if self._learning_decay == 'exponential':
            lr = tf.train.exponential_decay(self._learning_rate, self._global_step, self._learning_decay_steps, self._learning_decay_factor)
        elif self._learning_decay == 'polynomial':
            lr = tf.train.polynomial_decay(self._learning_rate, self._global_step, self._learning_decay_steps, self._learning_decay_factor)
        else:
            lr = self._learning_rate
        self._optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(final_loss)

        # tensorboard summaries
        self._train_summaries = tf.summary.merge_all(scope='losses')

        with tf.variable_scope('episode_summaries'):
            self._episode_score = tf.placeholder(shape=[], dtype=tf.float32, name='episode_score')
            tf.summary.scalar('episode_score', self._episode_score)
            self._epsilon = tf.placeholder(shape=[], dtype=tf.float32, name='episode_ending_epsilon')
            tf.summary.scalar('episode_ending_epsilon', self._epsilon)
        self._episode_summaries = tf.summary.merge_all(scope='episode_summaries')

        with tf.variable_scope('predict_summaries'):
            predict_actions = {}
            for name, q_vals in self._q.items():
                predict_actions[name] = tf.argmax(q_vals, axis=1)
            predict_action_one_hot = self._get_action_one_hot(predict_actions)
            predict_q_vals = []
            count = tf.Variable(tf.zeros([], dtype=np.float32), trainable=False)
            for name, q_vals in self._q.items():
                argument_mask = tf.reduce_max(predict_action_one_hot['function'] * argument_masks[name], axis=-1)
                predict_action_q_masked = tf.reduce_max(q_vals) * argument_mask
                predict_q_vals.append(predict_action_q_masked)
                count = count + argument_mask
            predicted_q_val_avg = tf.reduce_sum(tf.stack(predict_q_vals)) / count
            tf.summary.scalar('predicted_q_val', tf.squeeze(predicted_q_val_avg))
        self._predict_summaries = tf.summary.merge_all(scope='predict_summaries')

        # variable initializer
        self.var_init = tf.global_variables_initializer()

        # saver
        self.saver = tf.train.Saver(
            max_to_keep=self._max_checkpoints,
            keep_checkpoint_every_n_hours=self._checkpoint_hours
        )

    def episode_summary(self, sess, score, epsilon):
        return sess.run(
            self._episode_summaries,
            feed_dict={
                self._episode_score: score,
                self._epsilon: epsilon
            }
        )

    def update_target_q_net(self, sess):
        sess.run([v_t.assign(v) for v_t, v in zip(self._q_target_vars, self._q_vars)])

    def predict_one(self, sess, state):
        feed_dict = {}
        for name in self._states:
            # newaxis adds a new dimension of length 1 at the beginning (the batch size)
            feed_dict[self._states[name]] = np.expand_dims(state[name], axis=0)
        return sess.run([self._predict_summaries, self._q], feed_dict=feed_dict)

    def train_batch(self, sess, global_step, states, actions, rewards, next_states, terminal):
        # need batch size to reshape actions
        batch_size = actions['function'].shape[0]

        # everything else is a dictionary, so we need to loop through them
        feed_dict = {
            self._rewards: rewards,
            self._terminal: terminal,
            self._global_step: global_step
        }

        if self._double_dqn:
            actions_next_feed_dict = {}
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

        summary, _ = sess.run([self._train_summaries, self._optimizer], feed_dict=feed_dict)

        return summary

