import tensorflow as tf
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


def preprocess_state_input(inputs, config):
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

        if config['env']['use_hp_log_values']:
            # scale hit points (0-?) logarithmically (add 1 to avoid undefined) since they can be so high
            screen_unit_hit_points = tf.math.log1p(tf.cast(inputs['screen_unit_hit_points'], dtype=tf.float32))
            # add a dimension (depth)
            screen_unit_hit_points = tf.expand_dims(screen_unit_hit_points, axis=-1)
            to_concat.append(screen_unit_hit_points)
        if config['env']['use_shield_log_values']:
            screen_unit_shields = tf.math.log1p(tf.cast(inputs['screen_unit_shields'], dtype=tf.float32))
            screen_unit_shields = tf.expand_dims(screen_unit_shields, axis=-1)
            to_concat.append(screen_unit_shields)

        if config['env']['use_hp_ratios']:
            # ratio goes up to 255 max
            screen_unit_hit_points_ratio = tf.cast(inputs['screen_unit_hit_points_ratio'] / 255, dtype=tf.float32)
            screen_unit_hit_points_ratio = tf.expand_dims(screen_unit_hit_points_ratio, axis=-1)
            to_concat.append(screen_unit_hit_points_ratio)
        if config['env']['use_shield_ratios']:
            screen_unit_shields_ratio = tf.cast(inputs['screen_unit_shields_ratio'] / 255, dtype=tf.float32)
            screen_unit_shields_ratio = tf.expand_dims(screen_unit_shields_ratio, axis=-1)
            to_concat.append(screen_unit_shields_ratio)

        if config['env']['use_hp_cats']:
            ones = tf.ones(tf.shape(inputs['screen_unit_hit_points']))
            zeros = tf.zeros(tf.shape(inputs['screen_unit_hit_points']))
            hp = inputs['screen_unit_hit_points']
            # add a dimension (depth) to each
            vals = config['env']['hp_cats_values']
            to_concat.append(tf.expand_dims(tf.where(hp <= vals[0], ones, zeros), axis=-1))
            for i in range(1, len(vals)):
                to_concat.append(
                    tf.expand_dims(tf.where(tf.logical_and(hp > vals[i - 1], hp <= vals[i]), ones, zeros), axis=-1)
                )
            to_concat.append(tf.expand_dims(tf.where(hp > vals[-1], ones, zeros), axis=-1))
        if config['env']['use_shield_cats']:
            ones = tf.ones(tf.shape(inputs['screen_unit_hit_points']))
            zeros = tf.zeros(tf.shape(inputs['screen_unit_hit_points']))
            sh = inputs['screen_unit_shields']
            vals = config['env']['hp_cats_values']
            to_concat.append(tf.expand_dims(tf.where(sh <= vals[0], ones, zeros), axis=-1))
            for i in range(1, len(vals)):
                to_concat.append(
                    tf.expand_dims(tf.where(tf.logical_and(sh > vals[i - 1], sh <= vals[i]), ones, zeros), axis=-1)
                )
            to_concat.append(tf.expand_dims(tf.where(sh > vals[-1], ones, zeros), axis=-1))

        if config['env']['use_all_unit_types']:
            # pysc2 has a list of known unit types, and the max unit id is around 2000 but there are 259 units (v3.0)
            # 4th root of 259 is ~4 (Google rule of thumb for ratio of embedding dimensions to number of categories)
            # src: https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
            # embedding output: [batch_size, screen y, screen x, output_dim]
            screen_unit_type = tf.keras.layers.Embedding(
                input_dim=len(pysc2_static_data.UNIT_TYPES),
                output_dim=4
            )(inputs['screen_unit_type'])
            to_concat.append(screen_unit_type)
        elif config['env']['use_specific_unit_types']:
            screen_unit_type = tf.contrib.layers.one_hot_encoding(
                labels=inputs['screen_unit_type'],
                num_classes=len(config['env']['specific_unit_types'])
            )[:, :, :, 1:]
            # above throws away first layer that has zeros
            to_concat.append(screen_unit_type)

        if config['env']['use_buffs']:
            screen_buffs = tf.contrib.layers.one_hot_encoding(
                labels=inputs['screen_buffs'],
                num_classes=len(config['env']['buff_ids'])
            )[:, :, :, 1:]
            # above throws away first layer that has zeros
            to_concat.append(screen_buffs)

        screen = tf.concat(to_concat, axis=-1, name='screen_input')
        return screen


def get_state_placeholder(config):
    screen_shape = [None, config['env']['screen_size'], config['env']['screen_size']]

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
            shape=[None, len(config['env']['computed_action_list'])],
            dtype=tf.bool,
            name='available_actions'
        )
    )

    # hp and shield categories that are optional
    if config['env']['use_hp_log_values'] or config['env']['use_hp_cats']:
        state_placeholder['screen_unit_hit_points'] = tf.placeholder(
            shape=screen_shape,
            dtype=tf.int32,
            name='screen_unit_hit_points'
        )
    if config['env']['use_shield_log_values'] or config['env']['use_shield_cats']:
        state_placeholder['screen_unit_shields'] = tf.placeholder(
            shape=screen_shape,
            dtype=tf.int32,
            name='screen_unit_shields'
        )

    if config['env']['use_hp_ratios']:
        state_placeholder['screen_unit_hit_points_ratio'] = tf.placeholder(
            shape=screen_shape,
            dtype=tf.int32,
            name='screen_unit_hit_points_ratio'
        )
    if config['env']['use_shield_ratios']:
        state_placeholder['screen_unit_shields_ratio'] = tf.placeholder(
            shape=screen_shape,
            dtype=tf.int32,
            name='screen_unit_shields_ratio'
        )

    # unit types are optional
    if config['env']['use_all_unit_types'] or config['env']['use_specific_unit_types']:
        state_placeholder['screen_unit_type'] =tf.placeholder(
            shape=screen_shape,
            dtype=tf.int32,
            name='screen_unit_type'
        )

    # buffs are optional
    if config['env']['use_buffs']:
        state_placeholder['screen_buffs'] =tf.placeholder(
            shape=screen_shape,
            dtype=tf.int32,
            name='screen_buffs'
        )

    return state_placeholder


def get_argument_masks(config):
    masks = dict(function=tf.constant([1] * len(config['env']['computed_action_list']), dtype=tf.float32, name='function'))

    for arg_type in pysc2_actions.TYPES:
        if config['env']['computed_action_components'][arg_type.name]:
            mask = []
            for func in pysc2_actions.FUNCTIONS:
                if int(func.id) not in config['env']['computed_action_list']:
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


def get_num_options_per_function(config):
    # this is hopefully the only place this has to be hard coded
    return dict(
        function=len(config['env']['computed_action_list']),
        screen=config['env']['screen_size'] ** 2,
        minimap=config['env']['minimap_size'] ** 2,
        screen2=config['env']['screen_size'] ** 2,
        queued=2,
        control_group_act=5,
        control_group_id=config['env']['num_control_groups'],
        select_point_act=4,
        select_add=2,
        select_unit_act=4,
        select_unit_id=500,
        select_worker=4,
        build_queue_id=10,
        unload_id=500
    )


def get_action_one_hot(actions, config):
    # action components we are using.
    # number of options for function args hard coded here... probably won't change in pysc2
    num_options = get_num_options_per_function(config)

    action_one_hot = {}
    for name, using in config['env']['computed_action_components'].items():
        if using:
            action_one_hot[name] = tf.one_hot(actions[name], num_options[name], 1.0, 0.0, name=name)

    return action_one_hot
