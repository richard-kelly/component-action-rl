import tensorflow as tf


class Network:
    def __init__(self, learning_rate, max_checkpoints, checkpoint_hours):

        self._learning_rate = learning_rate
        self._max_checkpoints = max_checkpoints
        self._checkpoint_hours = checkpoint_hours

        # define the placeholders
        self._states = None
        self._actions = None

        # the output operations
        self._logits = None
        self._optimizer = None
        self.var_init = None
        self.saver = None

        # now setup the model
        self._define_model()

    def _define_model(self):

        self.screen_input = tf.placeholder(shape=[None, 84, 84, 5], dtype=tf.float32)
        self._q_s_a = dict(
            function=tf.placeholder(shape=[None, 4], dtype=tf.float32),
            screen_x=tf.placeholder(shape=[None, 84], dtype=tf.float32),
            screen_y=tf.placeholder(shape=[None, 84], dtype=tf.float32),
            screen2_x=tf.placeholder(shape=[None, 84], dtype=tf.float32),
            screen2_y=tf.placeholder(shape=[None, 84], dtype=tf.float32),
            select_point_act=tf.placeholder(shape=[None, 4], dtype=tf.float32),
            select_add=tf.placeholder(shape=[None, 2], dtype=tf.float32),
            queued=tf.placeholder(shape=[None, 2], dtype=tf.float32)
        )

        # conv1_non_spatial = tf.layers.conv2d(
        #     inputs=self.screen_input,
        #     filters=16,
        #     kernel_size=8,
        #     strides=4,
        #     padding='valid',
        #     name='conv1_non_spatial'
        # )
        #
        # max_1 = tf.layers.max_pooling2d(
        #     inputs=conv1_non_spatial,
        #     pool_size=3,
        #     strides=1,
        #     name='max_1'
        # )
        #
        # conv2_non_spatial = tf.layers.conv2d(
        #     inputs=max_1,
        #     filters=32,
        #     kernel_size=4,
        #     stride=2,
        #     padding='valid',
        #     name='conv2_spatial'
        # )
        #
        # max_2 = tf.layers.max_pooling2d(
        #     inputs=conv2_non_spatial,
        #     pool_size=3,
        #     strides=1,
        #     name='max_2'
        # )

        conv1_spatial = tf.layers.conv2d(
            inputs=self.screen_input,
            filters=16,
            kernel_size=5,
            padding='same',
            name='conv1_spatial'
        )

        conv2_spatial = tf.layers.conv2d(
            inputs=conv1_spatial,
            filters=32,
            kernel_size=3,
            padding='same',
            name='conv2_spatial'
        )

        # MUST flatten conv or pooling layers before sending to dense layer
        conv2_spatial_flat = tf.reshape(conv2_spatial, [-1, 84 * 84 * 32], name='conv2_spatial_flat')
        fc_spatial = tf.layers.dense(conv2_spatial_flat, 1024, activation=tf.nn.relu, name='fc_spatial')

        spatial_policy_1 = tf.layers.conv2d(
            inputs=conv2_spatial,
            filters=1,
            kernel_size=1,
            padding='same',
            name='spatial_policy_1'
        )

        spatial_policy_2 = tf.layers.conv2d(
            inputs=conv2_spatial,
            filters=1,
            kernel_size=1,
            padding='same',
            name='spatial_policy_2'
        )

        self._logits = dict(
            function=tf.layers.dense(fc_spatial, 4, name='function'),
            screen_x=tf.reshape(tf.reduce_max(spatial_policy_1, axis=2), [-1, 84], name='screen_x'),
            screen_y=tf.reshape(tf.reduce_max(spatial_policy_1, axis=1), [-1, 84], name='screen_y'),
            screen2_x=tf.reshape(tf.reduce_max(spatial_policy_2, axis=2), [-1, 84], name='screen2_x'),
            screen2_y=tf.reshape(tf.reduce_max(spatial_policy_2, axis=1), [-1, 84], name='screen2_y'),
            select_point_act=tf.layers.dense(fc_spatial, 4, name='select_point_act'),
            select_add=tf.layers.dense(fc_spatial, 2, name='select_add'),
            queued=tf.layers.dense(fc_spatial, 2, name='queued')
        )

        losses = []
        for key in self._logits.keys():
            # print(key)
            # print(self._q_s_a[key].shape)
            # print(self._logits[key].shape)
            losses.append(tf.losses.mean_squared_error(self._q_s_a[key], self._logits[key]))

        loss = tf.add_n(losses, name='losses')

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(loss)

        self.var_init = tf.global_variables_initializer()

        self.saver = tf.train.Saver(
            max_to_keep=self._max_checkpoints,
            keep_checkpoint_every_n_hours=self._checkpoint_hours
        )

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self.screen_input: state['screen'].reshape(1, 84, 84, 5)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self.screen_input: states['screen']})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(
            self._optimizer,
            feed_dict={
                self.screen_input: x_batch['screen'],
                self._q_s_a['function']: y_batch['function'],
                self._q_s_a['screen_x']: y_batch['screen_x'],
                self._q_s_a['screen_y']: y_batch['screen_y'],
                self._q_s_a['screen2_x']: y_batch['screen2_x'],
                self._q_s_a['screen2_y']: y_batch['screen2_y'],
                self._q_s_a['select_point_act']: y_batch['select_point_act'],
                self._q_s_a['select_add']: y_batch['select_add'],
                self._q_s_a['queued']: y_batch['queued']
            }
        )

