import tensorflow as tf
import numpy as np
import math


class MRTSNetwork:
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

        # define the placeholders
        self._global_step = None
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._terminal = None
        self._per_weights = None

        # the output operations
        self._q = None
        self._actions_selected_by_q = None
        self._td_abs = None
        self._optimizer = None
        self.var_init = None
        self.saver = None

        # now setup the model
        self._define_model()

    def _get_network(self, inputs):
        # num_classes should be one more than actual, because it includes 0, which we ignore in the slice
        terrain = tf.contrib.layers.one_hot_encoding(
            labels=inputs['terrain'],
            num_classes=2
        )[:, :, :, 1:]

        units_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['units'],
            num_classes=8
        )[:, :, :, 1:]

        health_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['health'],
            num_classes=6
        )[:, :, :, 1:]

        players_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['players'],
            num_classes=3
        )[:, :, :, 1:]

        eta_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['eta'],
            num_classes=8
        )[:, :, :, 1:]

        resources_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['resources'],
            num_classes=8
        )[:, :, :, 1:]

        screen = tf.concat(
            [
                terrain,
                units_one_hot,
                health_one_hot,
                players_one_hot,
                eta_one_hot,
                resources_one_hot
            ],
            axis=-1,
            name='screen_input'
        )

        # begin shared conv layers
        conv1_spatial = tf.layers.conv2d(
            inputs=screen,
            filters=64,
            kernel_size=3,
            padding='same',
            name='conv1_spatial',
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0)
        )

        conv2_spatial = tf.layers.conv2d(
            inputs=conv1_spatial,
            filters=32,
            kernel_size=3,
            padding='same',
            name='conv2_spatial',
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0)
        )

        with tf.variable_scope('spatial_gradient_scale'):
            # scale because multiple action component streams are meeting here
            # TODO: come up with better scaling based on which action components are used in training.
            scale = 1 / math.sqrt(2)
            conv2_spatial = (1 - scale) * tf.stop_gradient(conv2_spatial) + scale * conv2_spatial

        # MUST flatten conv or pooling layers before sending to dense layer
        non_spatial_flat = tf.reshape(
            conv2_spatial,
            shape=[-1, int(self._screen_size * self._screen_size * 32)],
            name='non_spatial_flat'
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
                512,
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
            512,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
            name='fc_non_spatial_2'
        )

        with tf.variable_scope('non_spatial_gradient_scale'):
            # scale because multiple action component streams are meeting here
            # TODO: come up with better scaling based on which action components are used in training.
            scale = 1 / math.sqrt(2)
            fc_non_spatial_2 = (1 - scale) * tf.stop_gradient(fc_non_spatial_2) + scale * fc_non_spatial_2

        select = tf.layers.conv2d(
            inputs=conv2_spatial,
            filters=1,
            kernel_size=1,
            padding='same',
            name='select'
        )

        param = tf.layers.conv2d(
            inputs=conv2_spatial,
            filters=1,
            kernel_size=1,
            padding='same',
            name='param'
        )

        n = self._screen_size
        action_q_vals = dict(
            select=tf.reshape(select, [-1, n * n], name='select'),
            type=tf.layers.dense(fc_non_spatial_2, 6, name='type'),
            param=tf.reshape(param, [-1, n * n], name='param'),
            unit_type=tf.layers.dense(fc_non_spatial_2, 6, name='unit_type')
        )

        if self._dueling:
            # action_q_vals_filtered is A(s,a), value is V(s)
            # Q(s,a) = V(s) + A(s,a) - 1/|A| * SUM_a(A(s,a))
            with tf.variable_scope('q_vals'):
                for name, advantage in action_q_vals.items():
                    action_q_vals[name] = tf.add(value, (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)), name=name)

        return action_q_vals

    def _get_state_placeholder(self):
        return dict(
            terrain=tf.placeholder(
                shape=[None, self._screen_size, self._screen_size],
                dtype=tf.int32,
                name='terrain'
            ),
            units=tf.placeholder(
                shape=[None, self._screen_size, self._screen_size],
                dtype=tf.int32,
                name='units'
            ),
            health=tf.placeholder(
                shape=[None, self._screen_size, self._screen_size],
                dtype=tf.int32,
                name='health'
            ),
            players=tf.placeholder(
                shape=[None, self._screen_size, self._screen_size],
                dtype=tf.int32,
                name='players'
            ),
            eta=tf.placeholder(
                shape=[None, self._screen_size, self._screen_size],
                dtype=tf.int32,
                name='eta'
            ),
            resources=tf.placeholder(
                shape=[None, self._screen_size, self._screen_size],
                dtype=tf.int32,
                name='resources'
            ),
            # available_resources=tf.placeholder(
            #     shape=[None, ],
            #     dtype=tf.int32,
            #     name='available_resources'
            # )
        )

    def _get_action_placeholders(self):
        # number of options for some function args hard coded here
        return dict(
            select=tf.placeholder(shape=[None, ], dtype=tf.int32, name='select'),
            type=tf.placeholder(shape=[None, ], dtype=tf.int32, name='type'),
            param=tf.placeholder(shape=[None, ], dtype=tf.int32, name='param'),
            unit_type=tf.placeholder(shape=[None, ], dtype=tf.int32, name='unit_type'),
        )

    def _get_argument_masks(self):
        return dict(
            select=tf.constant([1, 1, 1, 1, 1, 1], dtype=tf.float32, name='select'),
            type=tf.constant([1, 1, 1, 1, 1, 1], dtype=tf.float32, name='type'),
            param=tf.constant([0, 1, 1, 1, 1, 1], dtype=tf.float32, name='param'),
            unit_type=tf.constant([0, 0, 0, 0, 1, 0], dtype=tf.float32, name='unit_type')
        )

    def _get_action_one_hot(self, actions):
        # number of options for some function args hard coded here
        return dict(
            select=tf.one_hot(actions['select'], self._screen_size * self._screen_size, 1.0, 0.0, name='select'),
            type=tf.one_hot(actions['type'], 6, 1.0, 0.0, name='type'),
            param=tf.one_hot(actions['param'], self._screen_size * self._screen_size, 1.0, 0.0, name='param'),
            unit_type=tf.one_hot(actions['unit_type'], 6, 1.0, 0.0, name='unit_type')
        )

    def _define_model(self):
        # placeholders for (s, a, s', r, terminal)
        with tf.variable_scope('states_placeholders'):
            self._states = self._get_state_placeholder()
        with tf.variable_scope('action_placeholders'):
            self._actions = self._get_action_placeholders()
        with tf.variable_scope('next_states_placeholders'):
            self._next_states = self._get_state_placeholder()
        self._rewards = tf.placeholder(shape=[None, ], dtype=tf.float32, name='reward_placeholder')
        self._terminal = tf.placeholder(shape=[None, ], dtype=tf.float32, name='terminal_placeholder')
        self._per_weights = tf.placeholder(shape=[None, ], dtype=tf.float32, name='per_weights_placeholder')

        # primary and target Q nets
        with tf.variable_scope('Q_primary', regularizer=self._regularizer):
            self._q = self._get_network(self._states)
        with tf.variable_scope('Q_target'):
            self._q_target = self._get_network(self._next_states)
        # used for copying parameters from primary to target net
        self._q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_primary")
        self._q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Q_target")

        if self._double_dqn:
            # next action placeholders for Double DQN
            with tf.variable_scope('action_next_placeholders'):
                self._actions_next = self._get_action_placeholders()

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
            # TODO: update this later, for now just assume always double DQN
            # else:
                # in DQN, choosing actions based on target network qvals for next states
                # actions_next = {}
                # for name, q_vals in self._q_target.items():
                #     actions_next[name] = tf.argmax(q_vals, axis=1)
                # next_states_action_one_hot = self._get_action_one_hot(actions_next)

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
            # else:
            #     # DQN uses target network max Q val
            #     for name, q_vals in self._q_target.items():
            #         max_q_next_by_target = tf.reduce_max(q_vals, axis=-1, name=name)
            #         y_components[name] = (1 - self._terminal) * self._discount * max_q_next_by_target
            y_components_masked = []
            # get vector of 0s of correct length
            num_components = self._rewards * 0
            for name in y_components:
                argument_mask = tf.reduce_max(next_states_action_one_hot['type'] * argument_masks[name], axis=-1)
                # keep track of number of components used in this action
                num_components = num_components + argument_mask
                y_components_masked.append(y_components[name] * argument_mask)
            y_parts_stacked = tf.stack(y_components_masked, axis=1)
            y = tf.stop_gradient(self._rewards + tf.reduce_sum(y_parts_stacked, axis=1) / num_components)

        # calculate losses (average of y compared to each component of prediction action)
        with tf.variable_scope('losses'):
            losses = []
            td = []
            num_components = self._rewards * 0
            for name in training_action_q:
                # argument mask is scalar 1 if this argument is used for the transition action, 0 otherwise
                argument_mask = tf.reduce_max(action_one_hot['type'] * argument_masks[name], axis=-1)
                training_action_q_masked = training_action_q[name] * argument_mask
                y_masked = y * argument_mask
                # we compare the q value of each component to the target y; y is masked if training q is masked
                loss = tf.losses.huber_loss(training_action_q_masked, y_masked)
                td.append(tf.abs(training_action_q_masked - y_masked))
                num_components = num_components + argument_mask
                losses.append(loss)
            # TODO: The following might make more sense as a sum instead of mean,
            #  but then probably the learning rate should come down
            training_losses = tf.reduce_mean(tf.stack(losses), name='training_losses')
            reg_loss = tf.losses.get_regularization_loss()
            final_loss = training_losses + reg_loss
            tf.summary.scalar('training_loss', training_losses)
            tf.summary.scalar('regularization_loss', reg_loss)
            self._td_abs = tf.reduce_sum(tf.stack(td, axis=1), axis=1) / num_components

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
                argument_mask = tf.reduce_max(predict_action_one_hot['type'] * argument_masks[name], axis=-1)
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

    def train_batch(self, sess, global_step, states, actions, rewards, next_states, terminal, weights):
        # need batch size to reshape actions
        batch_size = actions['type'].shape[0]

        # everything else is a dictionary, so we need to loop through them
        feed_dict = {
            self._rewards: rewards,
            self._terminal: terminal,
            self._global_step: global_step
        }

        if weights is not None:
            feed_dict[self._per_weights] = weights
        else:
            feed_dict[self._per_weights] = np.ones(batch_size, dtype=np.float32)

        if self._double_dqn:
            q_next_feed_dict = {}
            for name in self._states:
                q_next_feed_dict[self._states[name]] = next_states[name]
            q_next = sess.run(self._q, feed_dict=q_next_feed_dict)
            actions_next = self.choose_valid_action(next_states, q_next)
            for name in actions_next:
                feed_dict[self._actions_next[name]] = actions_next[name].reshape(batch_size)

        for name in self._states:
            feed_dict[self._states[name]] = states[name]
            feed_dict[self._next_states[name]] = next_states[name]

        for name in actions:
            feed_dict[self._actions[name]] = actions[name].reshape(batch_size)

        summary, td_abs, _ = sess.run([self._train_summaries, self._td_abs, self._optimizer], feed_dict=feed_dict)

        return summary, td_abs

    def _flattened_to_grid(self, elem):
        # returns in (x, y) order
        return elem % self._screen_size, elem // self._screen_size

    def _grid_to_flattened(self, tile):
        return tile[1] * self._screen_size + tile[0]

    def _get_neighbours(self, x, y):
        neighbours = []
        if x > 0:
            neighbours.append((x - 1, y))
        if x < self._screen_size - 1:
            neighbours.append((x + 1, y))
        if y > 0:
            neighbours.append((x, y - 1))
        if y < self._screen_size - 1:
            neighbours.append((x, y + 1))
        return neighbours

    def _get_ranged_targets(self, x, y):
        # hard-coded for range 3
        # corners of 5x5 square around origin
        corners = [(a, b) for a in [x - 2, x - 1, x + 1, x + 2] for b in [y - 2, y - 1, y + 1, y + 2]]
        # horizontal and vertical lines extending from the origin
        lines_h = [(a, y) for a in [x - 3, x - 2, x - 1, x + 1, x + 2, x + 3]]
        lines_v = [(x, b) for b in [y - 3, y - 2, y - 1, y + 1, y + 2, y + 3]]
        targets = corners + lines_h + lines_v
        valid_targets = []
        for target in targets:
            a, b, = target
            if a >= 0 and a < self._screen_size and b >= 0 and b < self._screen_size:
                valid_targets.append(target)
        return valid_targets

    def choose_valid_action(self, states, q_vals):
        batch = states['units'].shape[0]
        n = self._screen_size
        costs = dict(
            worker=1,
            light=2,
            heavy=3,
            ranged=2,
            barracks=5,
            base=10
        )

        # SELECT = unit that is doing the action
        # mask out tiles that don't have our units
        q_vals['select'] = np.where(states['players'].reshape((batch, n * n)) == 1, q_vals['select'], np.nan)
        # mask out tiles with units that already have actions
        q_vals['select'] = np.where(states['eta'].reshape((batch, n * n)) == 0, q_vals['select'], np.nan)
        actions = dict(select=np.nanargmax(q_vals['select'], axis=1))

        # TYPE = type of action: [no-op, move, harvest, return, produce, attack]
        unit_types = np.reshape(states['units'], (batch, n * n))[np.arange(batch), actions['select']]
        for i in range(batch):
            x, y = self._flattened_to_grid(actions['select'][i])
            neighbours = self._get_neighbours(x, y)
            if unit_types[i] == 1:
                # bases can only do no_op and produce (0 and 4)
                q_vals['type'][i][np.array([1, 2, 3, 5], dtype=np.int32)] = np.nan
                # are there enough resources to produce? Bases only make workers
                can_produce = False
                if states['available_resources'][i] >= costs['worker']:
                    # is there a space to produce? first get x,y coords of selected unit
                    params_for_produce = np.zeros(n * n, dtype=np.int32)
                    for neighbour in neighbours:
                        # check for empty tiles (no terrain, no units)
                        x_n, y_n = neighbour
                        flat_n = self._grid_to_flattened(neighbour)
                        if states['terrain'][i][y_n, x_n] == 1:
                            continue
                        if states['units'][i][y_n, x_n] != 0:
                            continue
                        # this is a valid neighbour
                        params_for_produce[flat_n] = 1
                        can_produce = True
                    q_vals['param'][i] = np.where(params_for_produce, q_vals['param'][i], np.nan)
                if not can_produce:
                    q_vals['type'][i][4] = np.nan
                else:
                    # mask out units that can't be produced by Base
                    q_vals['unit_type'][i][np.array([0, 1, 3, 4, 5], dtype=np.int32)] = np.nan
            elif unit_types[i] == 2:
                # Barracks, can only do no_op and Produce L/H/R
                q_vals['type'][i][np.array([1, 2, 3, 5], dtype=np.int32)] = np.nan
                # are there enough resources to produce?
                can_produce = False
                if states['available_resources'][i] >= min([costs['light'], costs['heavy'], costs['ranged']]):
                    # is there a space to produce? first get x,y coords of selected unit
                    params_for_produce = np.zeros(n * n, dtype=np.int32)
                    for neighbour in neighbours:
                        # check for empty tiles (no terrain, no units)
                        x_n, y_n = neighbour
                        flat_n = self._grid_to_flattened(neighbour)
                        if states['terrain'][i][y_n, x_n] == 1:
                            continue
                        if states['units'][i][y_n, x_n] != 0:
                            continue
                        # this is a valid neighbour
                        params_for_produce[flat_n] = 1
                        can_produce = True
                    q_vals['param'][i] = np.where(params_for_produce, q_vals['param'][i], np.nan)
                if not can_produce:
                    q_vals['type'][i][4] = np.nan
                else:
                    # mask out units that can't be produced by barracks given current resources
                    mask = [0, 1, 2]
                    if states['available_resources'][i] < costs['light']:
                        mask.append(3)
                    if states['available_resources'][i] < costs['heavy']:
                        mask.append(4)
                    if states['available_resources'][i] < costs['ranged']:
                        mask.append(5)
                    q_vals['unit_type'][i][np.array(mask, dtype=np.int32)] = np.nan
            elif unit_types[i] == 3:
                # Worker can do every action
                # are there enough resources to produce?
                if states['available_resources'][i] < min([costs['barracks'], costs['base']]):
                    # no producing
                    q_vals['type'][i][4] = np.nan
                # is there a space to produce or move?
                can_produce_or_move = False
                params_for_produce_move = np.zeros(n * n, dtype=np.int32)
                for neighbour in neighbours:
                    # check for empty tiles (no terrain, no units)
                    x_n, y_n = neighbour
                    flat_n = self._grid_to_flattened(neighbour)
                    if states['terrain'][i][y_n, x_n] == 1:
                        continue
                    if states['units'][i][y_n, x_n] != 0:
                        continue
                    # this is a valid neighbour
                    params_for_produce_move[flat_n] = 1
                    can_produce_or_move = True
                if not can_produce_or_move:
                    q_vals['type'][i][np.array([1, 4], dtype=np.int32)] = np.nan
                else:
                    # mask out units that can't be produced by worker given current resources
                    mask = [2, 3, 4, 5]
                    if states['available_resources'][i] < costs['base']:
                        mask.append(0)
                    if states['available_resources'][i] < costs['barracks']:
                        mask.append(1)
                    q_vals['unit_type'][i][np.array(mask, dtype=np.int32)] = np.nan
                # is there a space to attack?
                can_attack = False
                params_for_attack = np.zeros(n * n, dtype=np.int32)
                for neighbour in neighbours:
                    # check for enemy units
                    x_n, y_n = neighbour
                    flat_n = self._grid_to_flattened(neighbour)
                    if states['players'][i][y_n, x_n] != 2:
                        continue
                    # this is a valid neighbour with enemy unit
                    params_for_attack[flat_n] = 1
                    can_attack = True
                if not can_attack:
                    q_vals['type'][i][5] = np.nan
                # can the worker harvest or return?
                params_for_harvest = np.zeros(n * n, dtype=np.int32)
                params_for_return = np.zeros(n * n, dtype=np.int32)
                can_harvest = False
                can_return = False
                if states['resources'][i][y][x] == 0:
                    # worker not holding resources; can't return but can harvest
                    # check for neighbour to harvest from
                    for neighbour in neighbours:
                        x_n, y_n = neighbour
                        flat_n = self._grid_to_flattened(neighbour)
                        if states['units'][i][y_n, x_n] != 7:
                            continue
                        # this is a valid neighbour resource patch
                        params_for_harvest[flat_n] = 1
                        can_harvest = True
                else:
                    # worker IS holding resources; can return but can't harvest
                    # check for neighbour to return to
                    for neighbour in neighbours:
                        x_n, y_n = neighbour
                        flat_n = self._grid_to_flattened(neighbour)
                        if not (states['units'][i][y_n, x_n] == 1 and states['players'][i][y_n, x_n] == 1):
                            continue
                        # this is a valid base to return to
                        params_for_return[flat_n] = 1
                        can_return = True
                if not can_harvest:
                    q_vals['type'][i][2] = np.nan
                if not can_return:
                    q_vals['type'][i][3] = np.nan
                # FINALLY choose which action this worker is taking
                action = np.nanargmax(q_vals['type'][i])
                if action == 1 or action == 4:
                    q_vals['param'][i] = np.where(params_for_produce_move, q_vals['param'][i], np.nan)
                elif action == 2:
                    q_vals['param'][i] = np.where(params_for_harvest, q_vals['param'][i], np.nan)
                elif action == 3:
                    q_vals['param'][i] = np.where(params_for_return, q_vals['param'][i], np.nan)
                elif action == 5:
                    q_vals['param'][i] = np.where(params_for_attack, q_vals['param'][i], np.nan)
            elif unit_types[i] == 4 or unit_types[i] == 5 or unit_types[i] == 6:
                # combat unit; only Ranged is slightly different because it can attack more tiles
                # can only do no_op, move, and attack (0, 1, 5)
                q_vals['type'][i][np.array([2, 3, 4], dtype=np.int32)] = np.nan
                # is there a space to move?
                can_move = False
                params_for_move = np.zeros(n * n, dtype=np.int32)
                for neighbour in neighbours:
                    # check for empty tiles (no terrain, no units)
                    x_n, y_n = neighbour
                    flat_n = self._grid_to_flattened(neighbour)
                    if states['terrain'][i][y_n, x_n] == 1:
                        continue
                    if states['units'][i][y_n, x_n] != 0:
                        continue
                    # this is a valid neighbour
                    params_for_move[flat_n] = 1
                    can_move = True
                if not can_move:
                    q_vals['type'][i][1] = np.nan
                # is there a space to attack?
                can_attack = False
                params_for_attack = np.zeros(n * n, dtype=np.int32)
                if unit_types[i] == 6:
                    attackable = self._get_ranged_targets(x, y)
                else:
                    attackable = neighbours
                for target in attackable:
                    # check for enemy units
                    x_n, y_n = target
                    flat_n = self._grid_to_flattened(target)
                    if states['players'][i][y_n, x_n] != 2:
                        continue
                    # this is a valid tile with enemy unit
                    params_for_attack[flat_n] = 1
                    can_attack = True
                if not can_attack:
                    q_vals['type'][i][5] = np.nan

                # choose which action this unit is taking
                action = np.nanargmax(q_vals['type'][i])
                if action == 1:
                    q_vals['param'][i] = np.where(params_for_move, q_vals['param'][i], np.nan)
                elif action == 5:
                    q_vals['param'][i] = np.where(params_for_attack, q_vals['param'][i], np.nan)

            # make sure all unit_types and params are valid for the nanargmax function
            # if they're all NaN they won't actually be used.
            if np.all(np.isnan(q_vals['unit_type'][i])):
                q_vals['unit_type'][i][0] = 1
            if np.all(np.isnan(q_vals['param'][i])):
                q_vals['param'][i][0] = 1

        # Now all invalid actions have had their corresponding q_val set to NaN
        # (excluding some components that won't be used for this action)
        actions['type'] = np.nanargmax(q_vals['type'], axis=1)
        actions['param'] = np.nanargmax(q_vals['param'], axis=1)
        actions['unit_type'] = np.nanargmax(q_vals['unit_type'], axis=1)

        return actions

