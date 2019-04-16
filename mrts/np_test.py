import numpy as np

n = 2


def flattened_to_grid(elem):
    # x, y
    return elem % n, elem // n


def grid_to_flattened(tile):
    return tile[1] * n + tile[0]


def get_neighbours(x, y):
    neighbours = []
    if x > 0:
        neighbours.append((x - 1, y))
    if x < n - 1:
        neighbours.append((x + 1, y))
    if y > 0:
        neighbours.append((x, y - 1))
    if y < n - 1:
        neighbours.append((x, y + 1))
    return neighbours


def get_ranged_targets(x, y):
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
        if a >= 0 and a < n and b >= 0 and b < n:
            valid_targets.append(target)
    return valid_targets


def choose_valid_action(states, q_vals, costs):
    batch = states['units'].shape[0]

    # SELECT = unit that is doing the action
    # mask out tiles that don't have our units
    q_vals['select'] = np.where(states['player'].reshape((batch, n * n)) == 1, q_vals['select'], np.nan)
    # mask out tiles with units that already have actions
    q_vals['select'] = np.where(states['eta'].reshape((batch, n * n)) == 0, q_vals['select'], np.nan)
    actions = dict(select=np.nanargmax(q_vals['select'], axis=1))

    # TYPE = type of action: [no-op, move, harvest, return, produce, attack]
    unit_types = np.reshape(states['units'], (batch, n * n))[np.arange(batch), actions['select']]
    for i in range(batch):
        x, y = flattened_to_grid(actions['select'][i])
        neighbours = get_neighbours(x, y)
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
                    flat_n = grid_to_flattened(neighbour)
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
                    flat_n = grid_to_flattened(neighbour)
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
                flat_n = grid_to_flattened(neighbour)
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
                flat_n = grid_to_flattened(neighbour)
                if states['player'][i][y_n, x_n] != 2:
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
            if states['resources'][i][x][y] == 0:
                # worker not holding resources; can't return but can harvest
                # check for neighbour to harvest from
                for neighbour in neighbours:
                    x_n, y_n = neighbour
                    flat_n = grid_to_flattened(neighbour)
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
                    flat_n = grid_to_flattened(neighbour)
                    if not (states['units'][i][y_n, x_n] == 0 and states['player'][i][y_n, x_n] == 1):
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
                flat_n = grid_to_flattened(neighbour)
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
                attackable = get_ranged_targets(x, y)
            else:
                attackable = neighbours
            for target in attackable:
                # check for enemy units
                x_n, y_n = target
                flat_n = grid_to_flattened(target)
                if states['player'][i][y_n, x_n] != 2:
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

    # Now all invalid actions have had their corresponding q_val set to NAN
    # (excluding some components that won't be used for this action)
    actions['type'] = np.nanargmax(q_vals['type'], axis=1)
    actions['param'] = np.nanargmax(q_vals['param'], axis=1)
    actions['unit_type'] = np.nanargmax(q_vals['unit_type'], axis=1)
    return actions


def main():
    # screen is 2x2
    q_vals = dict(
        select=np.array([[3, 4, 5, 2]], dtype=np.float32),
        type=np.array([[3.4, 5.6, 20, 9.0, 15, 11]], dtype=np.float32),
        param=np.array([[3, 4, 5, 2]], dtype=np.float32),
        unit_type=np.array([[3.4, 5.6, 7.8, 9.0, 1, 2]], dtype=np.float32)
    )

    states = dict(
        terrain=np.array([[[0, 0], [0, 0]]], dtype=np.int8),
        player=np.array([[[0, 2], [1, 1]]], dtype=np.int8),
        eta=np.array([[[0, 0], [0, 0]]], dtype=np.int8),
        units=np.array([[[0, 6], [3, 1]]], dtype=np.int8),
        resources=np.array([[[0, 0], [1, 0]]], dtype=np.int8),
        available_resources=np.array([20]), dtype=np.int32
    )

    costs = dict(
        worker=1,
        light=2,
        heavy=3,
        ranged=2,
        barracks=5,
        base=10
    )

    actions = choose_valid_action(states, q_vals, costs)
    print(actions)
    interpret_actions(actions)


def interpret_actions(actions):
    for i in range(actions['select'].shape[0]):
        x, y = flattened_to_grid(actions['select'][i])
        select_coords = '(' + str(x) + ',' + str(y) + ')'
        a, b = flattened_to_grid(actions['param'][i])
        param_coords = '(' + str(a) + ',' + str(b) + ')'
        action_type = actions['type'][i]
        action_names = ['no_op', 'move', 'harvest', 'return', 'produce', 'attack']
        produce_types = ['base', 'barracks', 'worker', 'light', 'heavy', 'ranged']
        print('Unit at', select_coords, 'doing', action_names[action_type], '/', param_coords, '/', produce_types[actions['unit_type'][i]])


if __name__ == '__main__':
    main()



