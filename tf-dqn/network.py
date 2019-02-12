import tensorflow as tf


class Network:
    def __init__(self, learning_rate):

        self._learning_rate = learning_rate

        # define the placeholders
        self._states = None
        self._actions = None

        # the output operations
        self._logits = None
        self._optimizer = None
        self.var_init = None

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

        conv1 = tf.layers.conv2d(
            inputs=self.screen_input,
            filters=16,
            kernel_size=5,
            padding='same'
        )

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=16,
            kernel_size=3,
            padding='same'
        )

        # MUST flatten conv or pooling layers before sending to dense layer
        conv2_flat = tf.reshape(conv2, [-1, 84 * 84 * 16])
        fc = tf.layers.dense(conv2_flat, 1024, activation=tf.nn.relu)
        print(self._q_s_a['screen_x'].shape)
        self._logits = dict(
            function=tf.layers.dense(fc, 4),
            screen_x=tf.layers.dense(fc, 84),
            screen_y=tf.layers.dense(fc, 84),
            screen2_x=tf.layers.dense(fc, 84),
            screen2_y=tf.layers.dense(fc, 84),
            select_point_act=tf.layers.dense(fc, 4),
            select_add=tf.layers.dense(fc, 2),
            queued=tf.layers.dense(fc, 2)
        )
        print(self._logits['screen_x'].shape)
        losses = []
        for key in self._logits.keys():
            print(key)
            losses.append(tf.losses.mean_squared_error(self._q_s_a[key], self._logits[key]))

        loss = tf.add_n(losses)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(loss)

        self.var_init = tf.global_variables_initializer()

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

