import tensorflow as tf
import numpy as np
import math

from tf_dqn.common import network_utils

from pysc2.lib import actions as pysc2_actions
from pysc2.lib import static_data as pysc2_static_data

spatial_components = ['screen', 'screen2', 'minimap']
all_components = dict(
        function=True,
        screen=False,
        minimap=False,
        screen2=False,
        queued=False,
        control_group_act=False,
        control_group_id=False,
        select_point_act=False,
        select_add=False,
        select_unit_act=False,
        select_unit_id=False,
        select_worker=False,
        build_queue_id=False,
        unload_id=False
    )
component_order = ['function', 'queued', 'control_group_act', 'control_group_id', 'select_point_act', 'select_add',
                   'select_unit_act', 'select_unit_id', 'select_worker', 'build_queue_id', 'unload_id',
                   'screen', 'screen2', 'minimap']

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
        with tf.variable_scope('input_processing'):
            # all processed screen input will be added to this list
            to_concat = []

            screen_player_relative_one_hot = tf.contrib.layers.one_hot_encoding(
                labels=inputs['screen_player_relative'],
                num_classes=5
            )
            # we only want self and enemy:
            # NONE = 0, SELF = 1, ALLY = 2, NEUTRAL = 3, ENEMY = 4
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

            if self._config['env']['use_hp_log_values']:
                # scale hit points (0-?) logarithmically (add 1 to avoid undefined) since they can be so high
                screen_unit_hit_points = tf.math.log1p(tf.cast(inputs['screen_unit_hit_points'], dtype=tf.float32))
                # add a dimension (depth)
                screen_unit_hit_points = tf.expand_dims(screen_unit_hit_points, axis=-1)
                to_concat.append(screen_unit_hit_points)
            if self._config['env']['use_shield_log_values']:
                screen_unit_shields = tf.math.log1p(tf.cast(inputs['screen_unit_shields'], dtype=tf.float32))
                screen_unit_shields = tf.expand_dims(screen_unit_shields, axis=-1)
                to_concat.append(screen_unit_shields)

            if self._config['env']['use_hp_ratios']:
                # ratio goes up to 255 max
                screen_unit_hit_points_ratio = tf.cast(inputs['screen_unit_hit_points_ratio'] / 255, dtype=tf.float32)
                screen_unit_hit_points_ratio = tf.expand_dims(screen_unit_hit_points_ratio, axis=-1)
                to_concat.append(screen_unit_hit_points_ratio)
            if self._config['env']['use_shield_ratios']:
                screen_unit_shields_ratio = tf.cast(inputs['screen_unit_shields_ratio'] / 255, dtype=tf.float32)
                screen_unit_shields_ratio = tf.expand_dims(screen_unit_shields_ratio, axis=-1)
                to_concat.append(screen_unit_shields_ratio)

            if self._config['env']['use_hp_cats']:
                ones = tf.ones(tf.shape(inputs['screen_unit_hit_points']))
                zeros = tf.zeros(tf.shape(inputs['screen_unit_hit_points']))
                hp = inputs['screen_unit_hit_points']
                # add a dimension (depth) to each
                vals = self._config['env']['hp_cats_values']
                to_concat.append(tf.expand_dims(tf.where(hp <= vals[0], ones, zeros), axis=-1))
                for i in range(1, len(vals)):
                    to_concat.append(
                        tf.expand_dims(tf.where(tf.logical_and(hp > vals[i-1], hp <= vals[i]), ones, zeros), axis=-1)
                    )
                to_concat.append(tf.expand_dims(tf.where(hp > vals[-1], ones, zeros), axis=-1))
            if self._config['env']['use_shield_cats']:
                ones = tf.ones(tf.shape(inputs['screen_unit_hit_points']))
                zeros = tf.zeros(tf.shape(inputs['screen_unit_hit_points']))
                sh = inputs['screen_unit_shields']
                vals = self._config['env']['hp_cats_values']
                to_concat.append(tf.expand_dims(tf.where(sh <= vals[0], ones, zeros), axis=-1))
                for i in range(1, len(vals)):
                    to_concat.append(
                        tf.expand_dims(tf.where(tf.logical_and(sh > vals[i - 1], sh <= vals[i]), ones, zeros), axis=-1)
                    )
                to_concat.append(tf.expand_dims(tf.where(sh > vals[-1], ones, zeros), axis=-1))

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

        with tf.variable_scope('shared_spatial_network'):
            shared_spatial_net = network_utils.get_layers(
                screen,
                self._config['network_structure']['shared_spatial_network'],
                self._config['network_structure']['activation'],
                self._training
            )

        if self._config['network_structure']['scale_gradients_at_shared_spatial_split']:
            with tf.variable_scope('spatial_gradient_scale'):
                # scale because multiple action component streams are meeting here
                # (always one more branch than number of spatial components)
                spatial_count = 1
                for name, using in self._action_components.items():
                    if using and name in spatial_components:
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
                    self._config['network_structure']['activation'],
                    self._training
                )
                value = tf.layers.dense(
                    fc_value,
                    1,
                    activation=None,
                    kernel_initializer=tf.variance_scaling_initializer(scale=2.0),
                    name='value'
                )

        with tf.variable_scope('shared_non_spatial_network'):
            fc_non_spatial = network_utils.get_layers(
                shared_spatial_net,
                self._config['network_structure']['shared_non_spatial_network'],
                self._config['network_structure']['activation'],
                self._training
            )

        if self._config['network_structure']['scale_gradients_at_shared_non_spatial_split']:
            with tf.variable_scope('non_spatial_gradient_scale'):
                # scale because multiple action component streams are meeting here
                non_spatial_count = 0
                for name, using in self._action_components.items():
                    if using and name not in spatial_components:
                        non_spatial_count += 1
                scale = 1 / non_spatial_count
                fc_non_spatial = (1 - scale) * tf.stop_gradient(fc_non_spatial) + scale * fc_non_spatial

        num_options = self._get_num_options_per_function()

        # create each component stream
        component_streams = {}
        action_q_vals = {}
        action_one_hots = {}
        for c in component_order:
            # are we using this component?
            if self._action_components[c]:
                with tf.variable_scope(c + '_branch'):
                    if c in spatial_components and not self._config['network_structure']['use_dense_layers_for_spatial']:
                        spatial_policy = tf.layers.conv2d(
                            inputs=shared_spatial_net,
                            filters=1,
                            kernel_size=1,
                            padding='same'
                        )
                        component_streams[c] = tf.reshape(spatial_policy, [-1, num_options[c]], name=c)
                    else:
                        # optionally one stream of fully connected layers per component
                        spec = self._config['network_structure']['component_stream_default']
                        if c in self._config['network_structure']['component_stream_specs']:
                            spec = self._config['network_structure']['component_stream_specs'][c]
                        if c in spatial_components:
                            stream_input = tf.reshape(
                                shared_spatial_net,
                                shape=[-1, np.prod(shared_spatial_net.shape[1:])]
                            )
                        else:
                            stream_input = fc_non_spatial

                        # optionally feed one hot versions of earlier stream outputs to this stream
                        if self._config['network_structure']['use_stream_outputs_as_inputs_to_other_streams']:
                            if c in self._config['network_structure']['stream_dependencies']:
                                dependencies = [stream_input]
                                for d in self._config['network_structure']['stream_dependencies'][c]:
                                    dependencies.append(action_one_hots[d])
                                stream_input = tf.concat(dependencies, axis=-1)
                        component_fc = network_utils.get_dense_layers(
                            stream_input,
                            spec,
                            self._config['network_structure']['activation'],
                            self._training
                        )
                        # for non-spatial components make a dense layer with width equal to number of possible actions
                        component_streams[c] = tf.layers.dense(component_fc, num_options[c], name=c)

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
                            action_one_hot = tf.one_hot(action_index, num_options[c])
                            # argmax should be non-differentiable but just to remind myself use stop_gradient
                            action_one_hots[c] = tf.stop_gradient(action_one_hot)

        # return action_q_vals
        return action_q_vals

    def _get_state_placeholder(self):
        screen_shape = [None, self._config['env']['screen_size'], self._config['env']['screen_size']]

        # things that always go in
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

        # hp and shield categories that are optional
        if self._config['env']['use_hp_log_values'] or self._config['env']['use_hp_cats']:
            state_placeholder['screen_unit_hit_points'] = tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_unit_hit_points'
            )
        if self._config['env']['use_shield_log_values'] or self._config['env']['use_shield_cats']:
            state_placeholder['screen_unit_shields'] = tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_unit_shields'
            )

        if self._config['env']['use_hp_ratios']:
            state_placeholder['screen_unit_hit_points_ratio'] = tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_unit_hit_points_ratio'
            )
        if self._config['env']['use_shield_ratios']:
            state_placeholder['screen_unit_shields_ratio'] = tf.placeholder(
                shape=screen_shape,
                dtype=tf.int32,
                name='screen_unit_shields_ratio'
            )

        # unit types are optional
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
            # self._q, self._q_weights = self._get_network(self._states)
            self._q = self._get_network(self._states)
        with tf.variable_scope('Q_target'):
            # self._q_target, self._q_target_weights = self._get_network(self._next_states)
            self._q_target = self._get_network(self._next_states)
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

