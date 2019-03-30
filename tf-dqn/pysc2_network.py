import tensorflow as tf
import numpy as np


class SC2Network:
    def __init__(
            self,
            learning_rate,
            discount,
            max_checkpoints,
            checkpoint_hours,
            reg_type,
            reg_scale,
            environment_properties
    ):

        self._learning_rate = learning_rate
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
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._terminal = None

        # the output operations
        self._q = None
        self._optimizer = None
        self.var_init = None
        self.saver = None

        # now setup the model
        self._define_model()

    def _get_network(self, inputs):
        screen_player_relative_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['screen_player_relative'],
            num_classes=5
        )[:, :, :, 1:]

        screen_selected_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['screen_player_relative'],
            num_classes=2
        )[:, :, :, 1:]
        screen = tf.concat([screen_player_relative_one_hot, screen_selected_one_hot], axis=-1, name='screen_input')

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
        fc_non_spatial = tf.layers.dense(
            non_spatial_flat,
            1024,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
            name='fc_1'
        )

        fc_non_spatial2 = tf.layers.dense(
            fc_non_spatial,
            1024,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
            name='fc_2'
        )

        spatial_policy_1 = tf.layers.conv2d(
            inputs=conv2_spatial,
            filters=1,
            kernel_size=1,
            padding='same',
            name='spatial_policy_1'
        )

        comp = self._action_components
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
        logits = dict(
            function=tf.layers.dense(fc_non_spatial2, 4, name='function'),
            screen=tf.reshape(spatial_policy_1, [-1, n * n], name='screen') if comp['screen'] else None,
            screen2=tf.reshape(spatial_policy_2, [-1, n * n], name='screen2') if comp['screen2'] else None,
            queued=tf.layers.dense(fc_non_spatial, 2, name='queued') if comp['queued'] else None,
            select_point_act=tf.layers.dense(fc_non_spatial, 4, name='select_point_act') if comp['select_point_act'] else None,
            select_add=tf.layers.dense(fc_non_spatial, 2, name='select_add') if comp['select_add'] else None
        )

        logits_filtered = {}
        for name, val in logits.items():
            if val is not None:
                logits_filtered[name] = val
        return logits_filtered

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
        # action components we are using. Function is always included (for times when we iterate over action part names)
        comp = self._action_components
        n = self._screen_size * self._screen_size
        # number of options for some function args hard coded here
        return dict(
            function=tf.one_hot(actions['function'], len(self._action_list), 1.0, 0.0, name='function'),
            screen=tf.one_hot(actions['screen'], n, 1.0, 0.0, name='screen') if comp['screen'] else None,
            screen2=tf.one_hot(actions['screen2'], n, 1.0, 0.0, name='screen2') if comp['screen2'] else None,
            select_point_act=tf.one_hot(actions['select_point_act'], 4, 1.0, 0.0, name='select_point_act') if comp['select_point_act'] else None,
            select_add=tf.one_hot(actions['select_add'], 2, 1.0, 0.0, name='select_add') if comp['select_add'] else None,
            queued=tf.one_hot(actions['queued'], 2, 1.0, 0.0, name='queued') if comp['queued'] else None
        )

    def _define_model(self):
        # placeholders for (s, s', r, terminal) [a is below]
        with tf.variable_scope('states_placeholders'):
            self._states = self._get_state_placeholder()
        with tf.variable_scope('next_states_placeholders'):
            self._next_states = self._get_state_placeholder()
        self._rewards = tf.placeholder(shape=[None, ], dtype=tf.float32, name='reward_placeholder')
        self._terminal = tf.placeholder(shape=[None, ], dtype=tf.float32, name='terminal_placeholder')

        # primary and target Q nets
        with tf.variable_scope('Q_primary', regularizer=self._regularizer):
            self._q = self._get_network(self._states)
            # available actions mask; avoids using negative infinity, and is the right size
            action_neg_inf_q_vals = self._q['function'] * 0 - 1000000
            self._q['function'] = tf.where(self._states['available_actions'], self._q['function'], action_neg_inf_q_vals)
        with tf.variable_scope('Q_target'):
            self._q_target = self._get_network(self._next_states)
            # available actions mask; avoids using negative infinity, and is the right size
            action_neg_inf_q_vals = self._q_target['function'] * 0 - 1000000
            self._q_target['function'] = tf.where(self._next_states['available_actions'], self._q_target['function'], action_neg_inf_q_vals)

        # used for copying parameters from primary to target net
        self._q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_primary")
        self._q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_target")

        # action placeholders
        with tf.variable_scope('action_placeholders'):
            self._actions = {}
            for name, val in self._q.items():
                if val is not None:
                    self._actions[name] = tf.placeholder(shape=[None, ], dtype=tf.int32, name=name)

        # one hot the actions from experiences
        with tf.variable_scope('action_one_hot'):
            action_one_hot = self._get_action_one_hot(self._actions)

        # The prediction by the primary Q network for the actual actions
        with tf.variable_scope('prediction'):
            training_action_q = {}
            for name, q_vals in self._q.items():
                training_action_q[name] = tf.reduce_sum(q_vals * action_one_hot[name], reduction_indices=-1, name=name)

        # Next state q value
        with tf.variable_scope('optimization_target'):
            max_q_next_by_target = {}
            for name, q_vals in self._q_target.items():
                max_q_next_by_target[name] = tf.reduce_max(q_vals, axis=-1, name=name)

        # target Q(s,a)
        with tf.variable_scope('y'):
            y = {}
            for name, max_q_next in max_q_next_by_target.items():
                y[name] = self._rewards + (1 - self._terminal) * self._discount * max_q_next

        # these mask out the arguments that aren't used for the selected function from the loss calculation
        with tf.variable_scope('argument_masks'):
            argument_masks = self._get_argument_masks()

        # calculate losses (average of all args to the action function (including the function) compared pairwise)
        with tf.variable_scope('losses'):
            losses = []
            for name in y.keys():
                argument_mask = tf.reduce_max(action_one_hot['function'] * argument_masks[name], axis=-1)
                training_action_q_masked = training_action_q[name] * argument_mask
                y_masked = tf.stop_gradient(y[name]) * argument_mask
                # these masks work because we are doing MSE loss, so if both the predicted q and target q are 0,
                # the loss is zero. (unlike the available actions mask which has to make the values effectively -inf)
                loss = tf.losses.huber_loss(training_action_q_masked, y_masked)
                tf.summary.scalar('training_loss_' + name, loss)
                losses.append(loss)
            losses_avg = tf.reduce_mean(tf.stack(losses), name='losses_avg')
            reg_loss = tf.losses.get_regularization_loss()
            final_loss = losses_avg + reg_loss
            tf.summary.scalar('training_loss_avg', losses_avg)
            tf.summary.scalar('training_loss_reg', reg_loss)
            tf.summary.scalar('training_loss_final', final_loss)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(final_loss)

        # tensorboard summaries
        self._train_summaries = tf.summary.merge_all(scope='losses')

        with tf.variable_scope('episode_summaries'):
            self._episode_score = tf.placeholder(shape=[], dtype=tf.float32, name='episode_score')
            tf.summary.scalar('episode_score', self._episode_score)
            self._epsilon = tf.placeholder(shape=[], dtype=tf.float32, name='episode_ending_epsilon')
            tf.summary.scalar('episode_ending_epsilon', self._epsilon)
        self._episode_summaries = tf.summary.merge_all(scope='episode_summaries')

        with tf.variable_scope('predict_summaries'):
            for name, q_vals in self._q.items():
                action_q_val = tf.reduce_max(q_vals, name=name)
                tf.summary.scalar('step_q_' + name, action_q_val)
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

    def train_batch(self, sess, states, actions, rewards, next_states, terminal):
        batch = actions['function'].shape[0]
        feed_dict = {
            self._actions['function']: actions['function'].reshape(batch),
            self._rewards: rewards,
            self._terminal: terminal
        }

        for name, _ in self._states.items():
            feed_dict[self._states[name]] = states[name]
            feed_dict[self._next_states[name]] = next_states[name]

        for name, using in self._action_components.items():
            if using:
                feed_dict[self._actions[name]] = actions[name].reshape(batch)

        summary, _ = sess.run([self._train_summaries, self._optimizer], feed_dict=feed_dict)

        return summary

