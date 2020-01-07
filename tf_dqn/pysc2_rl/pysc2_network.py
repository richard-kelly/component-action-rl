import tensorflow as tf
import numpy as np
import math

from pysc2.lib import actions as pysc2_actions
from pysc2.lib import static_data as pysc2_static_data


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

    def _get_network(self, inputs):
        # all processed screen input will be added to this list
        to_concat = []

        screen_player_relative_one_hot = tf.contrib.layers.one_hot_encoding(
            labels=inputs['screen_player_relative'],
            num_classes=5
        )
        # we only want self and enemy:
        # NONE = 0
        # SELF = 1
        # ALLY = 2
        # NEUTRAL = 3
        # ENEMY = 4
        screen_player_relative_self = screen_player_relative_one_hot[:, :, :, 1]
        screen_player_relative_self = tf.expand_dims(screen_player_relative_self, axis=-1)
        to_concat.append(screen_player_relative_self)
        screen_player_relative_enemy = screen_player_relative_one_hot[:, :, :, 4]
        screen_player_relative_enemy = tf.expand_dims(screen_player_relative_enemy, axis=-1)
        to_concat.append(screen_player_relative_enemy)

        # observation is in int, but network uses floats
        # selected is binary, just 1 or 0, so is already in one hot form
        screen_selected_one_hot = tf.cast(inputs['screen_selected'], dtype=tf.float32)
        screen_selected_one_hot = tf.expand_dims(screen_selected_one_hot, axis=-1)
        to_concat.append(screen_selected_one_hot)

        if self._config['env']['use_hp_shield_log_values']:
            # scale hit points (0-?) logarithmically (add 1 to avoid undefined) since they can be so high
            screen_unit_hit_points = tf.math.log1p(tf.cast(inputs['screen_unit_hit_points'], dtype=tf.float32))
            # add a dimension (depth)
            screen_unit_hit_points = tf.expand_dims(screen_unit_hit_points, axis=-1)
            to_concat.append(screen_unit_hit_points)

            screen_unit_shields = tf.math.log1p(tf.cast(inputs['screen_unit_shields'], dtype=tf.float32))
            screen_unit_shields = tf.expand_dims(screen_unit_shields, axis=-1)
            to_concat.append(screen_unit_shields)

        if self._config['env']['use_hp_shield_ratios']:
            # ratio goes up to 255 max
            screen_unit_hit_points_ratio = tf.cast(inputs['screen_unit_hit_points_ratio'] / 255, dtype=tf.float32)
            screen_unit_hit_points_ratio = tf.expand_dims(screen_unit_hit_points_ratio, axis=-1)
            to_concat.append(screen_unit_hit_points_ratio)

            screen_unit_shields_ratio = tf.cast(inputs['screen_unit_shields_ratio'] / 255, dtype=tf.float32)
            screen_unit_shields_ratio = tf.expand_dims(screen_unit_shields_ratio, axis=-1)
            to_concat.append(screen_unit_shields_ratio)

        if self._config['env']['use_hp_shield_cats']:
            # hit point and shield categories
            ones = tf.ones(tf.shape(inputs['screen_unit_hit_points']))
            zeros = tf.zeros(tf.shape(inputs['screen_unit_hit_points']))
            hp = inputs['screen_unit_hit_points']

            # add a dimension (depth) to each
            to_concat.append(tf.expand_dims(tf.where(hp < 15, ones, zeros), axis=-1))
            to_concat.append(tf.expand_dims(tf.where(tf.logical_and(hp >= 15, hp < 30), ones, zeros), axis=-1))
            to_concat.append(tf.expand_dims(tf.where(tf.logical_and(hp >= 30, hp < 50), ones, zeros), axis=-1))
            to_concat.append(tf.expand_dims(tf.where(tf.logical_and(hp >= 50, hp < 100), ones, zeros), axis=-1))
            to_concat.append(tf.expand_dims(tf.where(tf.logical_and(hp >= 100, hp < 200), ones, zeros), axis=-1))
            sh = inputs['screen_unit_shields']
            to_concat.append(tf.expand_dims(tf.where(sh < 15, ones, zeros), axis=-1))
            to_concat.append(tf.expand_dims(tf.where(tf.logical_and(sh >= 15, sh < 30), ones, zeros), axis=-1))
            to_concat.append(tf.expand_dims(tf.where(tf.logical_and(sh >= 30, sh < 50), ones, zeros), axis=-1))
            to_concat.append(tf.expand_dims(tf.where(tf.logical_and(sh >= 50, sh < 100), ones, zeros), axis=-1))
            to_concat.append(tf.expand_dims(tf.where(tf.logical_and(sh >= 100, sh < 200), ones, zeros), axis=-1))

        if self._config['env']['use_all_unit_types']:
            # pysc2 has a list of known unit types, and the max unit id is around 2000 but there are 259 units (v3.0)
            # 4th root of 259 is ~4 (Google rule of thumb for ratio of embedding dimensions to number of categories)
            # src: https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            # embedding output: [batch_size, screen y, screen x, output_dim]
            screen_unit_type = tf.keras.layers.Embedding(
                input_dim=len(pysc2_static_data.UNIT_TYPES),
                output_dim=4
            )(inputs['screen_unit_type'])
            to_concat.append(screen_unit_type)
        elif self._config['env']['use_specific_unit_types']:
            screen_unit_type = tf.contrib.layers.one_hot_encoding(
                labels=inputs['screen_unit_type'],
                num_classes=len(self._config['env']['specific_unit_types'])
            )[:, :, :, 1:]
            # above throws away first layer that has zeros
            to_concat.append(screen_unit_type)

        screen = tf.concat(to_concat, axis=-1, name='screen_input')

        with tf.variable_scope('spatial_network'):
            conv_spatial_num_filters, conv_spatial = self._get_conv_layers(screen, self._config['network_structure']['spatial_network'])

        with tf.variable_scope('spatial_gradient_scale'):
            # scale because multiple action component streams are meeting here
            # TODO: come up with better scaling based on which action components are used in training.
            scale = 1 / math.sqrt(2)
            conv_spatial = (1 - scale) * tf.stop_gradient(conv_spatial) + scale * conv_spatial

        # spatial policy splits off before max pooling
        max_pool = tf.layers.max_pooling2d(
            inputs=conv_spatial,
            pool_size=3,
            strides=3,
            padding='valid',
            name='max_pool'
        )

        # MUST flatten conv or pooling layers before sending to dense layer
        non_spatial_flat = tf.reshape(
            max_pool,
            shape=[-1, int(self._config['env']['screen_size'] * self._config['env']['screen_size'] / 9 * conv_spatial_num_filters)],
            name='conv2_spatial_flat'
        )

        if self._config['dueling_network']:
            with tf.variable_scope('dueling_gradient_scale'):
                # scale the gradients entering last shared layer, as in original Dueling DQN paper
                scale = 1 / math.sqrt(2)
                non_spatial_flat = (1 - scale) * tf.stop_gradient(non_spatial_flat) + scale * non_spatial_flat

        # for dueling net, split here
        if self._config['dueling_network']:
            with tf.variable_scope('value_network'):
                fc_value = self._get_dense_layers(non_spatial_flat, self._config['network_structure']['value_network'])
                value = tf.layers.dense(
                    fc_value,
                    1,
                    activation=None,
                    kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
                    name='value'
                )
                value = tf.layers.batch_normalization(value, training=self._training)

        with tf.variable_scope('non_spatial_network'):
            fc_non_spatial = self._get_dense_layers(non_spatial_flat, self._config['network_structure']['non_spatial_network'])

        with tf.variable_scope('non_spatial_gradient_scale'):
            # scale because multiple action component streams are meeting here
            # TODO: come up with better scaling based on which action components are used in training.
            scale = 1 / math.sqrt(2)
            fc_non_spatial = (1 - scale) * tf.stop_gradient(fc_non_spatial) + scale * fc_non_spatial

        spatial_policy_1 = tf.layers.conv2d(
            inputs=conv_spatial,
            filters=1,
            kernel_size=1,
            padding='same',
            name='spatial_policy_1'
        )

        if self._action_components['screen2']:
            spatial_policy_2 = tf.layers.conv2d(
                inputs=conv_spatial,
                filters=1,
                kernel_size=1,
                padding='same',
                name='spatial_policy_2'
            )
        else:
            spatial_policy_2 = None

        comp = self._action_components
        num_options = self._get_num_options_per_function()
        action_q_vals = dict(
            function=tf.layers.dense(fc_non_spatial, num_options['function'], name='function'),
            screen=tf.reshape(spatial_policy_1, [-1, num_options['screen']], name='screen') if comp['screen'] else None,
            minimap=tf.reshape(spatial_policy_1, [-1, num_options['minimap']], name='minimap') if comp['minimap'] else None,
            screen2=tf.reshape(spatial_policy_2, [-1, num_options['screen2']], name='screen2') if comp['screen2'] else None,
            queued=tf.layers.dense(fc_non_spatial, num_options['queued'], name='queued') if comp['queued'] else None,
            control_group_act=tf.layers.dense(fc_non_spatial, num_options['control_group_act'], name='control_group_act') if comp['control_group_act'] else None,
            control_group_id=tf.layers.dense(fc_non_spatial, num_options['control_group_id'], name='control_group_id') if comp['control_group_id'] else None,
            select_point_act=tf.layers.dense(fc_non_spatial, num_options['select_point_act'], name='select_point_act') if comp['select_point_act'] else None,
            select_add=tf.layers.dense(fc_non_spatial, num_options['select_add'], name='select_add') if comp['select_add'] else None,
            select_unit_act=tf.layers.dense(fc_non_spatial, num_options['select_unit_act'], name='select_unit_act') if comp['select_unit_act'] else None,
            select_unit_id=tf.layers.dense(fc_non_spatial, num_options['select_unit_id'], name='select_unit_id') if comp['select_unit_id'] else None,
            select_worker=tf.layers.dense(fc_non_spatial, num_options['select_worker'], name='select_worker') if comp['select_worker'] else None,
            build_queue_id=tf.layers.dense(fc_non_spatial, num_options['build_queue_id'], name='build_queue_id') if comp['build_queue_id'] else None,
            unload_id=tf.layers.dense(fc_non_spatial, num_options['unload_id'], name='unload_id') if comp['unload_id'] else None,
        )

        num_active_args = 0
        self._arg_weight_association = dict()
        for name, val in action_q_vals.items():
            if val is not None:
                self._arg_weight_association[name] = num_active_args
                num_active_args += 1

        # function_selected =
        # q-q_val_weights =
        q_val_weights = tf.layers.dense(fc_non_spatial, num_active_args)
        q_val_weights = tf.nn.softmax(q_val_weights, name='q_val_weights')

        action_q_vals_filtered = {}
        for name, val in action_q_vals.items():
            if val is not None:
                action_q_vals_filtered[name] = val

        if self._config['dueling_network']:
            # action_q_vals_filtered is A(s,a), value is V(s)
            # Q(s,a) = V(s) + A(s,a) - 1/|A| * SUM_a(A(s,a))
            with tf.variable_scope('q_vals'):
                for name, advantage in action_q_vals_filtered.items():
                    action_q_vals_filtered[name] = tf.add(value, (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True)), name=name)

        with tf.variable_scope('available_actions_mask'):
            # available actions mask; avoids using negative infinity, and is the right size
            action_neg_inf_q_vals = action_q_vals_filtered['function'] * 0 - 1000000
            action_q_vals_filtered['function'] = tf.where(inputs['available_actions'], action_q_vals_filtered['function'], action_neg_inf_q_vals)

        return action_q_vals_filtered, q_val_weights

    def _get_state_placeholder(self):
        screen_shape = [None, self._config['env']['screen_size'], self._config['env']['screen_size']]
        state_placeholder = dict(
            screen_player_relative=tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_player_relative'
            ),
            screen_selected=tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_selected'
            ),
            available_actions=tf.placeholder(
                shape=[None, len(self._action_list)],
                dtype=tf.bool,
                name='available_actions'
            )
        )

        if self._config['env']['use_hp_shield_log_values'] or self._config['env']['use_hp_shield_cats']:
            state_placeholder['screen_unit_hit_points'] = tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_unit_hit_points'
            )
            state_placeholder['screen_unit_shields'] = tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_unit_shields'
            )

        if self._config['env']['use_hp_shield_ratios']:
            state_placeholder['screen_unit_hit_points_ratio'] = tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_unit_hit_points_ratio'
            )
            state_placeholder['screen_unit_shields_ratio'] = tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_unit_shields_ratio'
            )

        if self._config['env']['use_all_unit_types'] or self._config['env']['use_specific_unit_types']:
            state_placeholder['screen_unit_type'] =tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_unit_type'
            )

        return state_placeholder

    def _get_argument_masks(self):
        masks = dict(function=tf.constant([1] * len(self._action_list), dtype=tf.float32, name='function'))

        for arg_type in pysc2_actions.TYPES:
            if self._action_components[arg_type.name]:
                mask = []
                for func in pysc2_actions.FUNCTIONS:
                    if int(func.id) not in self._action_list:
                        continue
                    found = False
                    for arg in func.args:
                        if arg_type.name == arg.name:
                            found = True
                    if found:
                        mask.append(1)
                    else:
                        mask.append(0)
                masks[arg_type.name] = tf.constant(mask, dtype=tf.float32, name=arg_type.name)

        return masks

    def _get_num_options_per_function(self):
        screen_size = self._config['env']['screen_size']
        minimap_size = self._config['env']['minimap_size']
        # this is hopefully the only place this has to be hard coded
        return dict(
            function=len(self._action_list),
            screen=screen_size ** 2,
            minimap=minimap_size ** 2,
            screen2=screen_size ** 2,
            queued=2,
            control_group_act=5,
            control_group_id=self._num_control_groups,
            select_point_act=4,
            select_add=2,
            select_unit_act=4,
            select_unit_id=500,
            select_worker=4,
            build_queue_id=10,
            unload_id=500
        )

    def _get_action_one_hot(self, actions):
        # action components we are using.
        comp = self._action_components
        # number of options for function args hard coded here... probably won't change in pysc2
        num_options = self._get_num_options_per_function()

        action_one_hot = {}
        for name, using in comp.items():
            if using:
                action_one_hot[name] = tf.one_hot(actions[name], num_options[name], 1.0, 0.0, name=name)

        return action_one_hot


    def _get_conv_layers(self, inputs, spec):
        # expecting spec to be a list of lists of dicts.
        # each inner list is a list of conv layers using the same input to be concatenated
        # each dict gives the number of filters and kernel size of a conv layer
        num_output_layers = 0
        for conv_unit in spec:
            num_output_layers = 0
            conv_layers = []
            for conv in conv_unit:
                conv_layer = tf.layers.conv2d(
                    inputs=inputs,
                    filters=conv['filters'],
                    kernel_size=conv['kernel_size'],
                    padding='same',
                    activation=tf.nn.leaky_relu
                )
                conv_layer = tf.layers.batch_normalization(conv_layer, training=self._training)
                conv_layers.append(conv_layer)
                num_output_layers += conv['filters']
            inputs = tf.concat(conv_layers, axis=-1)
        return num_output_layers, inputs


    def _get_dense_layers(self, inputs, spec):
        # expecting spec to be a list of ints
        for num_units in spec:
            dense = tf.layers.dense(
                inputs,
                units=num_units,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.variance_scaling_initializer(scale=2.0)
            )
            dense = tf.layers.batch_normalization(dense, training=self._training)
            inputs = dense
        return inputs


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
        self._per_weights = tf.placeholder(shape=[None, ], dtype=tf.float32, name='per_weights_placeholder')
        self._training = tf.placeholder(shape=[], dtype=tf.bool, name='training_placeholder')

        # primary and target Q nets
        with tf.variable_scope('Q_primary', regularizer=self._regularizer):
            self._q, self._q_weights = self._get_network(self._states)
        with tf.variable_scope('Q_target'):
            self._q_target, self._q_target_weights = self._get_network(self._next_states)
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
            action_one_hot = self._get_action_one_hot(self._actions)

        # these mask out the arguments that aren't used for the selected function from the loss calculation
        with tf.variable_scope('argument_masks'):
            argument_masks = self._get_argument_masks()
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
                next_states_action_one_hot = self._get_action_one_hot(self._actions_next)
            else:
                # in DQN, choosing actions based on target network qvals for next states
                actions_next = {}
                for name, q_vals in self._q_target.items():
                    actions_next[name] = tf.argmax(q_vals, axis=1)
                next_states_action_one_hot = self._get_action_one_hot(actions_next)

        # modify q_val weights based on used arguments
        # all_masks = []
        # for name, i in self._arg_weight_association.items():
        #     all_masks.append(argument_masks[name])
        # all_masks = tf.stack(all_masks, axis=1)
        # tf.reduce_max(next_states_action_one_hot['function'] * all_masks, axis=-1)
            # # TODO: THIS IS WRONG!
            # adjusted_weights = self._q_target_weights[self._arg_weight_association[name]] * argument_masks[name]
            # self._q_target_weights[name] = tf.nn.softmax(adjusted_weights, name='q_val_weights_masked')

        # target Q(s,a)
        with tf.variable_scope('y'):
            y_components = {}
            if self._config['double_DQN']:
                # Double DQN uses target network Q val of primary network next action
                for name, action in self._actions_next.items():
                    row = tf.range(tf.shape(action)[0])
                    combined = tf.stack([row, action], axis=1)
                    max_q_next = tf.gather_nd(self._q_target[name], combined)
                    y_components[name] = (1 - self._terminal) * (self._config['discount'] ** self._config['bootstrapping_steps']) * max_q_next
            else:
                # DQN uses target network max Q val
                for name, q_vals in self._q_target.items():
                    max_q_next_by_target = tf.reduce_max(q_vals, axis=-1, name=name)
                    y_components[name] = (1 - self._terminal) * (self._config['discount'] ** self._config['bootstrapping_steps']) * max_q_next_by_target
            y_components_masked = []
            # get vector of 0s of correct length
            num_components = self._rewards * 0
            for name in y_components:
                argument_mask = tf.reduce_max(next_states_action_one_hot['function'] * argument_masks[name], axis=-1)
                # keep track of number of components used in this action
                num_components = num_components + argument_mask
                if self._config['use_component_weights']:
                    y_components[name] = y_components[name] * self._q_target_weights[:, self._arg_weight_association[name]]
                y_components_masked.append(y_components[name] * argument_mask)
            y_parts_stacked = tf.stack(y_components_masked, axis=1)
            if self._config['use_component_weights']:
                y = tf.stop_gradient(self._rewards + tf.reduce_sum(y_parts_stacked, axis=1))
            else:
                y = tf.stop_gradient(self._rewards + (tf.reduce_sum(y_parts_stacked, axis=1) / num_components))

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
            precidtion_q = tf.reduce_sum(prediction_parts_stacked, axis=1) / num_components

        # methods:
            # 1) average of y compared to each component of prediction action
            # 2) average of y compared to average of prediction
            # 3) each component compared pairwise, only if used in prediction [not implemented... what to do with reward?]
        loss_method = 1

        # calculate losses (average of y compared to each component of prediction action)
        with tf.variable_scope('losses'):
            if loss_method == 1:
                losses = []
                td = []
                num_components = self._rewards * 0
                for name in training_action_q:
                    # argument mask is scalar 1 if this argument is used for the transition action, 0 otherwise
                    argument_mask = tf.reduce_max(action_one_hot['function'] * argument_masks[name], axis=-1)
                    training_action_q_masked = training_action_q[name] * argument_mask
                    y_masked = y * argument_mask
                    if self._config['use_component_weights']:
                        training_action_q_masked = training_action_q_masked * self._q_weights[:, self._arg_weight_association[name]]
                    # we compare the q value of each component to the target y; y is masked if training q is masked
                    loss = tf.losses.huber_loss(training_action_q_masked, y_masked, weights=self._per_weights)
                    td.append(tf.abs(training_action_q_masked - y_masked))
                    num_components = num_components + argument_mask
                    losses.append(loss)
                # TODO: Switched to sum instead of mean, so maybe the learning rate should come down
                training_losses = tf.reduce_sum(tf.stack(losses), name='training_losses')
                reg_loss = tf.losses.get_regularization_loss()
                final_loss = training_losses + reg_loss
                tf.summary.scalar('training_loss', training_losses)
                tf.summary.scalar('regularization_loss', reg_loss)
                # self._td_abs = tf.reduce_sum(tf.stack(td, axis=1), axis=1) / num_components
                self._td_abs = tf.reduce_sum(tf.stack(td, axis=1), axis=1)
            # elif loss_method == 2:

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
        self._train_summaries = tf.summary.merge_all(scope='losses')

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
            max_to_keep=self._config['model_checkpoint_max'],
            keep_checkpoint_every_n_hours=self._config['model_checkpoint_every_n_hours']
        )

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

    def update_target_q_net(self, sess):
        sess.run([v_t.assign(v) for v_t, v in zip(self._q_target_vars, self._q_vars)])

    def predict_one(self, sess, state):
        feed_dict = {self._training: False}
        for name in self._states:
            # newaxis adds a new dimension of length 1 at the beginning (the batch size)
            feed_dict[self._states[name]] = np.expand_dims(state[name], axis=0)
        return sess.run([self._predict_summaries, self._q], feed_dict=feed_dict)

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

        summary, td_abs, _ = sess.run([self._train_summaries, self._td_abs, self._optimizer], feed_dict=feed_dict)

        return summary, td_abs

