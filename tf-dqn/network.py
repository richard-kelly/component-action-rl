import tensorflow as tf


class Network:
    def __init__(self, learning_rate, discount, max_checkpoints, checkpoint_hours, screen_size):

        self._learning_rate = learning_rate
        self._discount = discount
        self._max_checkpoints = max_checkpoints
        self._checkpoint_hours = checkpoint_hours
        self._screen_size = screen_size

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

    def _get_network(self, inputs, scope_name):
        with tf.variable_scope(scope_name):
            conv1_spatial = tf.layers.conv2d(
                inputs=inputs,
                filters=16,
                kernel_size=5,
                padding='same',
                name=scope_name + '_conv1_spatial'
            )

            conv2_spatial = tf.layers.conv2d(
                inputs=conv1_spatial,
                filters=32,
                kernel_size=3,
                padding='same',
                name=scope_name + '_conv2_spatial'
            )

            max_pool = tf.layers.max_pooling2d(
                inputs=conv2_spatial,
                pool_size=3,
                strides=3,
                padding='valid',
                name=scope_name + '_max_pool'
            )

            # MUST flatten conv or pooling layers before sending to dense layer
            non_spatial_flat = tf.reshape(
                max_pool,
                shape=[-1, int(self._screen_size * self._screen_size / 9 * 32)],
                name=scope_name + '_conv2_spatial_flat'
            )
            fc_non_spatial = tf.layers.dense(
                non_spatial_flat,
                512,
                activation=tf.nn.relu,
                name=scope_name + '_fc_spatial'
            )

            spatial_policy_1 = tf.layers.conv2d(
                inputs=conv2_spatial,
                filters=1,
                kernel_size=1,
                padding='same',
                name=scope_name + '_spatial_policy_1'
            )

            spatial_policy_2 = tf.layers.conv2d(
                inputs=conv2_spatial,
                filters=1,
                kernel_size=1,
                padding='same',
                name=scope_name + '_spatial_policy_2'
            )

            logits = dict(
                function=tf.layers.dense(fc_non_spatial, 4, name=scope_name + '_function'),
                screen=tf.reshape(spatial_policy_1, [-1, self._screen_size * self._screen_size], name=scope_name + '_screen_policy'),
                screen2=tf.reshape(spatial_policy_2, [-1, self._screen_size * self._screen_size], name=scope_name + '_screen2_policy'),
                select_point_act=tf.layers.dense(fc_non_spatial, 4, name=scope_name + '_select_point_act'),
                select_add=tf.layers.dense(fc_non_spatial, 2, name=scope_name + '_select_add'),
                queued=tf.layers.dense(fc_non_spatial, 2, name=scope_name + '_queued')
            )

            return logits

    def _define_model(self):

        self._states = tf.placeholder(shape=[None, self._screen_size, self._screen_size, 5], dtype=tf.float32, name='states_placeholder')
        self._actions = dict(
            function=tf.placeholder(shape=[None, ], dtype=tf.int32, name='action_function_placeholder'),
            screen=tf.placeholder(shape=[None, ], dtype=tf.int32, name='action_screen_placeholder'),
            screen2=tf.placeholder(shape=[None, ], dtype=tf.int32, name='action_screen2_placeholder'),
            select_point_act=tf.placeholder(shape=[None, ], dtype=tf.int32, name='action_select_point_act_placeholder'),
            select_add=tf.placeholder(shape=[None, ], dtype=tf.int32, name='action_select_add_placeholder'),
            queued=tf.placeholder(shape=[None, ], dtype=tf.int32, name='action_queued_placeholder')
        )
        self._rewards = tf.placeholder(shape=[None, ], dtype=tf.float32, name='reward_placeholder')
        self._next_states = tf.placeholder(shape=[None, self._screen_size, self._screen_size, 5], dtype=tf.float32, name='next_states_placeholder')
        self._not_terminal = tf.placeholder(shape=[None, ], dtype=tf.float32, name='not_terminal_placeholder')

        self._q = self._get_network(self._states, 'Q_primary')
        self._q_target = self._get_network(self._next_states, 'Q_target')

        self._q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_primary")
        self._q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_target")

        action_one_hot = dict(
            function=tf.one_hot(self._actions['function'], 4, 1.0, 0.0, name='function_one_hot'),
            screen=tf.one_hot(self._actions['screen'], self._screen_size * self._screen_size, 1.0, 0.0, name='screen_one_hot'),
            screen2=tf.one_hot(self._actions['screen2'], self._screen_size * self._screen_size, 1.0, 0.0, name='screen2_one_hot'),
            select_point_act=tf.one_hot(self._actions['select_point_act'], 4, 1.0, 0.0, name='select_point_act_one_hot'),
            select_add=tf.one_hot(self._actions['select_add'], 2, 1.0, 0.0, name='select_add_one_hot'),
            queued=tf.one_hot(self._actions['queued'], 2, 1.0, 0.0, name='queued_one_hot')
        )
        # The prediction by the primary Q network for the actual actions
        pred = {}
        for name, q_vals in self._q.items():
            pred[name] = tf.reduce_sum(q_vals * action_one_hot[name], reduction_indices=-1, name='q_acted_' + name)

        # The optimization target
        max_q_next_by_target = {}
        for name, q_vals in self._q_target.items():
            max_q_next_by_target[name] = tf.reduce_max(q_vals, axis=-1)

        y = {}
        for name, max_q_next in max_q_next_by_target.items():
            y[name] = self._rewards + self._not_terminal * self._discount * max_q_next

        losses = []
        for name in y.keys():
            print(name)
            print(pred[name].shape)
            print(y[name].shape)
            losses.append(tf.losses.mean_squared_error(pred[name], tf.stop_gradient(y[name])))

        loss = tf.add_n(losses, name='losses_sum')

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(loss)

        self.var_init = tf.global_variables_initializer()

        self.saver = tf.train.Saver(
            max_to_keep=self._max_checkpoints,
            keep_checkpoint_every_n_hours=self._checkpoint_hours
        )

    def update_target_q_net(self, sess):
        sess.run([v_t.assign(v) for v_t, v in zip(self._q_target_vars, self._q_vars)])

    def predict_one(self, state, sess):
        return sess.run(
            self._q,
            feed_dict={self._states: state['screen'].reshape(1, self._screen_size, self._screen_size, 5)}
        )

    def train_batch(self, sess, states, actions, rewards, next_states, not_terminal):
        batch = actions['function'].shape[0]
        sess.run(
            self._optimizer,
            feed_dict={
                self._states: states['screen'],
                self._actions['function']: actions['function'].reshape(batch),
                self._actions['screen']: actions['screen'].reshape(batch),
                self._actions['screen2']: actions['screen2'].reshape(batch),
                self._actions['select_point_act']: actions['select_point_act'].reshape(batch),
                self._actions['select_add']: actions['select_add'].reshape(batch),
                self._actions['queued']: actions['queued'].reshape(batch),
                self._rewards: rewards,
                self._next_states: next_states['screen'],
                self._not_terminal: not_terminal
            }
        )

