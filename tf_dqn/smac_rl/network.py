import tensorflow as tf


class Network:
    def __init__(self, learning_rate, discount, max_checkpoints, checkpoint_hours, env_info):

        self._learning_rate = learning_rate
        self._discount = discount
        self._max_checkpoints = max_checkpoints
        self._checkpoint_hours = checkpoint_hours

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

        # environment properties
        self._num_actions = env_info["n_actions"]
        self._state_shape = env_info["obs_shape"]

        # now setup the model
        self._define_model()

    def _get_network(self, inputs):
        fc_1 = tf.layers.dense(
            inputs,
            512,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
            name='fc_1'
        )

        fc_2 = tf.layers.dense(
            fc_1,
            512,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
            name='fc_2'
        )

        logits = dict(
            action=tf.layers.dense(fc_2, self._num_actions, name='action'),
        )

        return logits

    def _define_model(self):
        self._states = tf.placeholder(
            shape=[None, self._state_shape],
            dtype=tf.float32,
            name='states_placeholder'
        )
        with tf.variable_scope('action_placeholders'):
            self._actions = dict(
                action=tf.placeholder(shape=[None, ], dtype=tf.int32, name='action'),
            )
        self._rewards = tf.placeholder(shape=[None, ], dtype=tf.float32, name='reward_placeholder')
        self._next_states = tf.placeholder(
            shape=[None, self._state_shape],
            dtype=tf.float32,
            name='next_states_placeholder'
        )
        self._not_terminal = tf.placeholder(shape=[None, ], dtype=tf.float32, name='not_terminal_placeholder')

        with tf.variable_scope('Q_primary'):
            self._q = self._get_network(self._states)
        with tf.variable_scope('Q_target'):
            self._q_target = self._get_network(self._next_states)

        self._q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_primary")
        self._q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_target")

        with tf.variable_scope('action_one_hot'):
            action_one_hot = dict(
                action=tf.one_hot(self._actions['action'], self._num_actions, 1.0, 0.0, name='action'),
            )

        with tf.variable_scope('prediction'):
            # The prediction by the primary Q network for the actual actions
            pred = {}
            for name, q_vals in self._q.items():
                pred[name] = tf.reduce_sum(q_vals * action_one_hot[name], reduction_indices=-1, name=name)

        with tf.variable_scope('optimization_target'):
            # The optimization target
            max_q_next_by_target = {}
            for name, q_vals in self._q_target.items():
                max_q_next_by_target[name] = tf.reduce_max(q_vals, axis=-1, name=name)

        with tf.variable_scope('y'):
            y = {}
            for name, max_q_next in max_q_next_by_target.items():
                y[name] = self._rewards + self._not_terminal * self._discount * max_q_next

        with tf.variable_scope('losses'):
            losses = []
            for name in y.keys():
                loss = tf.losses.huber_loss(pred[name], tf.stop_gradient(y[name]))
                tf.summary.scalar('training_loss_' + name, loss)
                losses.append(loss)
            losses_sum = tf.add_n(losses, name='losses_sum')
            tf.summary.scalar('training_loss_total', losses_sum)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(losses_sum)

        # tensorboard
        self._train_summaries = tf.summary.merge_all(scope='losses')

        with tf.variable_scope('episode_summaries'):
            self._episode_score = tf.placeholder(shape=[], dtype=tf.float32, name='episode_score')
            tf.summary.scalar('episode_score', self._episode_score)
        self._episode_summaries = tf.summary.merge_all(scope='episode_summaries')

        with tf.variable_scope('predict_summaries'):
            for name, q_vals in self._q.items():
                action_q_val = tf.reduce_max(q_vals, name=name)
                tf.summary.scalar('step_q_' + name, action_q_val)
        self._predict_summaries = tf.summary.merge_all(scope='predict_summaries')

        self.var_init = tf.global_variables_initializer()

        self.saver = tf.train.Saver(
            max_to_keep=self._max_checkpoints,
            keep_checkpoint_every_n_hours=self._checkpoint_hours
        )

    def episode_summary(self, sess, score):
        return sess.run(self._episode_summaries, feed_dict={self._episode_score: score})

    def update_target_q_net(self, sess):
        sess.run([v_t.assign(v) for v_t, v in zip(self._q_target_vars, self._q_vars)])

    def predict_one(self, sess, state):
        return sess.run(
            [self._predict_summaries, self._q],
            feed_dict={self._states: state['state'].reshape(1, self._state_shape)}
        )

    def train_batch(self, sess, states, actions, rewards, next_states, not_terminal):
        batch = actions['action'].shape[0]
        summary, _ = sess.run(
            [self._train_summaries, self._optimizer],
            feed_dict={
                self._states: states['state'],
                self._actions['action']: actions['action'].reshape(batch),
                self._rewards: rewards,
                self._next_states: next_states['state'],
                self._not_terminal: not_terminal
            }
        )

        return summary

