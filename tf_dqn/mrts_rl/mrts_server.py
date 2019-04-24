import os
import asyncio
import json
import re
import numpy as np
import tensorflow as tf

from tf_dqn.mrts_rl.dqn_agent import DQNAgent
import tf_dqn.common.utils as utils

HOST = '127.0.0.1'
PORT = 9898

# connection counter
conn_count = 0

# remember the player for each connection
conn_player = {}

# game details
budgets = None
unit_types = None
move_conflict_resolution_strategy = None

rl_agent = None
step = 0


def get_conn_count():
    global conn_count
    conn_count += 1
    return conn_count


def handle_game_over(winner, conn_num):
    global rl_agent
    if winner == -1:
        reward = 0
        print('Connection', conn_num, ': GAME OVER - DRAW')
    elif winner == conn_player[conn_num]:
        reward = 1
        print('Connection', conn_num, ': GAME OVER - WON')
    else:
        reward = -1
        print('Connection', conn_num, ': GAME OVER - LOST')
    rl_agent.observe(terminal=True, reward=reward)


def handle_unit_type_table(utt):
    # This gets sent (sometimes?) twice per game, because reset() gets called twice in the SocketAI implementation
    global move_conflict_resolution_strategy
    global unit_types

    # handles when units try to move into the same space on the same frame:
    # 1 - cancel both; 2 - cancel random; 3 - cancel alternating
    move_conflict_resolution_strategy = utt['moveConflictResolutionStrategy']

    # a list of unit types, each of which is a dict with:
    #   ID (int)   : just equivalent to place in list, no relation to other uses of 'ID'
    #   name (str) : e.g. 'Resource'
    #   cost (int) : only meaningful for things that can be built
    #   hp (int)   : max hp
    #   minDamage (int), maxDamage (int), attackRange (int)
    #   produceTime (int) : The time it takes to produce this unit
    #   moveTime (int), attackTime (int), harvestTime (int), returnTime (int) : time cost to perform these actions
    #   harvestAmount (int) : only workers harvest and they harvest 1
    #   sightRadius (int) : for use in the partially observable ruleset
    #   isResource (bool), isStockpile (bool) [place a resource can be returned to - only Bases]
    #   canHarvest (bool), canMove (bool), canAttack (bool)
    #   produces (list of str): e.g. ['Base', 'Barracks']
    #   producedBy (list of str): e.g. ['Base']

    # convert to dict indexed by unit names
    unit_types = {}
    for unit_type in utt['unitTypes']:
        unit_types[unit_type['name']] = unit_type


def handle_pre_game_analysis(state, ms, conn_num):
    # nothing to do with this for now
    # ms is the number of ms we have to exit this function (and send back a response, which happens elsewhere)
    # state is the staring state of the game (t=0)
    print('Connection', conn_num, ': Received pre-game analysis state for', ms, 'ms.')


def handle_get_action(state, player, conn_num):
    global conn_player
    global rl_agent
    global step
    conn_player[conn_num] = player
    state_for_rl = {}
    # state:
    #   map (0, 0) is top left
    game_frame = state['time']
    map_w = state['pgs']['width']
    map_h = state['pgs']['height']
    biggest = max([map_w, map_h])
    if biggest <= 8:
        map_size = 8
    elif biggest <= 16:
        map_size = 16
    elif biggest <= 32:
        map_size = 32
    elif biggest <= 64:
        map_size = 64
    elif biggest <= 128:
        map_size = 128
    else:
        print('Map dimension', biggest, 'is too big.')
        return

    # terrain can be 0 (empty) or 1 (wall); in row major order
    terrain = np.array([int(i) for i in state['pgs']['terrain']], dtype=np.int8).reshape((map_h, map_w))
    # for now pad non-square maps with 'walls' in the bottom and right
    terrain = np.pad(terrain, ((0, map_size - map_h), (0, map_size - map_w)), 'constant', constant_values=1)
    state_for_rl['terrain'] = terrain

    # list, each player is dict with ints "ID" and "resources"
    # the ID here is the same as player parameter to this function and "player" in unit object below
    players = state['pgs']['players']
    state_for_rl['available_resources'] = np.array(players[player]['resources'], dtype=np.int32)

    # list, each unit is dict with string "type" and
    # ints "ID", "player", "x", "y", "resources", "hitpoints"
    # ID here is a unique id for each unit in the game
    # resources is same for all bases belonging to a player, and equal to players[x]['resources']
    # convert to dict indexed by ID
    units = {}
    friendly_unit_id_by_coordinates = {}
    for unit in state['pgs']['units']:
        units[unit['ID']] = unit
        if player == unit['player']:
            friendly_unit_id_by_coordinates[(unit['x'], unit['y'])] = unit['ID']

    unit_type_names = ['Base', 'Barracks', 'Worker', 'Light', 'Heavy', 'Ranged', 'Resource']
    units_feature = np.zeros((map_size, map_size), dtype=np.int8)
    for _, unit in units.items():
        units_feature[unit['y'], unit['x']] = unit_type_names.index(unit['type']) + 1
    state_for_rl['units'] = units_feature

    # TODO: try other representations of health... normalized real number? thermometer encoding?
    health_feature = np.zeros((map_size, map_size), dtype=np.int8)
    for _, unit in units.items():
        if unit['hitpoints'] == 1:
            health_feature[unit['y'], unit['x']] = 1
        elif unit['hitpoints'] == 2:
            health_feature[unit['y'], unit['x']] = 2
        elif unit['hitpoints'] == 3:
            health_feature[unit['y'], unit['x']] = 3
        elif unit['hitpoints'] == 4:
            health_feature[unit['y'], unit['x']] = 4
        elif unit['hitpoints'] >= 5:
            health_feature[unit['y'], unit['x']] = 5
    state_for_rl['health'] = health_feature

    # switch player ownership feature based on which player we are
    players_feature = np.zeros((map_size, map_size), dtype=np.int8)
    for _, unit in units.items():
        if unit['player'] == 0:
            if player == 0:
                players_feature[unit['y'], unit['x']] = 1
            else:
                players_feature[unit['y'], unit['x']] = 2
        elif unit['player'] == 1:
            if player == 0:
                players_feature[unit['y'], unit['x']] = 2
            else:
                players_feature[unit['y'], unit['x']] = 1
    state_for_rl['players'] = players_feature

    # actions ongoing for both players. A list with dicts:
    #   "ID":       int [unit ID],
    #   "time":     int [game frame when the action was given]
    #   "action":   {type, parameter, unitType, etc.]
    # convert to dict indexed by ID
    current_actions = {}
    for ongoing_action in state['actions']:
        current_actions[ongoing_action['ID']] = ongoing_action
        # for now mark any space where something is moving or producing as a "wall" so we won't try to use it
        if ongoing_action['action']['type'] == 1 or ongoing_action['action']['type'] == 4:
            param = ongoing_action['action']['parameter']
            x = units[ongoing_action['ID']]['x']
            y = units[ongoing_action['ID']]['y']
            if param == 0:
                # up
                state_for_rl['terrain'][y - 1][x] = 1
            elif param == 1:
                # right
                state_for_rl['terrain'][y][x + 1] = 1
            elif param == 2:
                # down
                state_for_rl['terrain'][y + 1][x] = 1
            elif param == 3:
                # left
                state_for_rl['terrain'][y][x - 1] = 1


    # TODO: add more information about current actions (type, target, etc.)

    eta_feature = np.zeros((map_size, map_size), dtype=np.int8)
    for ID, current_action in current_actions.items():
        unit = units[current_action['ID']]
        eta = get_eta(game_frame, current_action, unit['type'])
        eta_cat = get_eta_category(eta)
        eta_feature[unit['y'], unit['x']] = eta_cat
    state_for_rl['eta'] = eta_feature

    resources_feature = np.zeros((map_size, map_size), dtype=np.int8)
    for _, unit in units.items():
        if unit['resources'] == 1:
            resources_feature[unit['y'], unit['x']] = 1
        elif unit['resources'] == 2:
            resources_feature[unit['y'], unit['x']] = 2
        elif unit['resources'] == 2:
            resources_feature[unit['y'], unit['x']] = 3
        elif unit['resources'] == 3:
            resources_feature[unit['y'], unit['x']] = 4
        elif unit['resources'] == 4:
            resources_feature[unit['y'], unit['x']] = 5
        elif unit['resources'] == 5:
            resources_feature[unit['y'], unit['x']] = 6
        elif unit['resources'] <= 9:
            resources_feature[unit['y'], unit['x']] = 7
        elif unit['resources'] >= 10:
            resources_feature[unit['y'], unit['x']] = 8
    state_for_rl['resources'] = resources_feature

    # An action for a turn is a list with actions for each unit: a dict with
    # "unitID": int,
    # "unitAction": {
    #     "type":      int [one of 0 (none/wait), 1 (move), 2 (harvest), 3 (return), 4 (produce), 5 (attack_location)],
    #     // used for direction (of move, harvest, return, produce) and duration (wait)
    #     "parameter": int [one of -1 (none), 0 (up), 1 (right), 2 (down), 3 (left) OR any positive int for wait],
    #     "x":         int [x coordinate of attack],
    #     "y":         int [y coordinate of attack],
    #     "unitType":  str [the name of the type of unit to produce with a produce action]
    # }

    # Actions have to have a target / be legal at the time they are issued.
    # So even though actions take time to complete, there has to be a target at the time an attack is issued,
    # or a space has to be empty in order to issue a move action.
    # This is probably why attacks are all faster than moves.
    # some things we have to avoid in an action:
    #   don't send an action for a unit that is already doing an action
    #   don't try to produce two things in the same place

    actions = []

    friendly_units_without_actions = []
    for _, unit_id in friendly_unit_id_by_coordinates.items():
        if unit_id not in current_actions:
            friendly_units_without_actions.append(unit_id)

    while len(friendly_units_without_actions) > 0:
        if step > 0:
            rl_agent.observe(terminal=False, reward=0)
        step += 1
        action = rl_agent.act(state_for_rl)

        mrts_action = {}
        x, y = utils.flattened_to_grid(map_size, action['select'])
        x_p, y_p = utils.flattened_to_grid(map_size, action['param'])
        unit_id = friendly_unit_id_by_coordinates[(x, y)]
        mrts_action['unitID'] = unit_id

        unit_action = {'type': int(action['type'])}
        if unit_action['type'] == 0:
            # no_op/wait; always wait for only one frame
            unit_action['parameter'] = 1
        elif 1 <= unit_action['type'] <= 4:
            # one of move, harvest, return, produce
            if y_p == y - 1:
                # up
                unit_action['parameter'] = 0
            elif x_p == x + 1:
                # right
                unit_action['parameter'] = 1
            elif y_p == y + 1:
                # down
                unit_action['parameter'] = 2
            elif x_p == x - 1:
                # left
                unit_action['parameter'] = 3
            else:
                print('Invalid parameter for action type', action['type'])
        else:
            # attack
            unit_action['parameter'] = -1
            unit_action['x'] = int(x_p)
            unit_action['y'] = int(y_p)
        # handle produce
        if unit_action['type'] == 4:
            unit_action['unitType'] = unit_type_names[action['unit_type']]

        if unit_action['type'] == 0 or unit_action['type'] == 4:
            # mark spaces used for move or produce as a 'wall' so that it can't be chosen again
            state_for_rl['terrain'][y_p][x_p] = 1

        # modify state so that this unit is now doing an action
        current_action = {'time': game_frame, 'action': unit_action}
        unit = units[unit_id]
        eta = get_eta(game_frame, current_action, unit['type'])
        eta_cat = get_eta_category(eta)
        state_for_rl['eta'][unit['y'], unit['x']] = eta_cat

        mrts_action['unitAction'] = unit_action
        actions.append(mrts_action)
        friendly_units_without_actions.remove(unit_id)

    return json.dumps(actions)


def get_eta(game_frame, current_action, unit_type):
    action = current_action['action']
    time = current_action['time']
    time_elapsed = game_frame - time
    if action['type'] == 0:
        # no_op
        action_duration = action['parameter']
    elif action['type'] == 1:
        # move
        action_duration = unit_types[unit_type]['moveTime']
    elif action['type'] == 2:
        # harvest
        action_duration = unit_types[unit_type]['harvestTime']
    elif action['type'] == 3:
        # return
        action_duration = unit_types[unit_type]['returnTime']
    elif action['type'] == 4:
        # produce
        action_duration = unit_types[action['unitType']]['produceTime']
    elif action['type'] == 5:
        # attack
        action_duration = unit_types[unit_type]['attackTime']
    else:
        # error?
        action_duration = time_elapsed

    eta = action_duration - time_elapsed
    return eta


def get_eta_category(eta):
    if eta <= 5:
        return 1
    elif eta <= 10:
        return 2
    elif eta <= 25:
        return 3
    elif eta <= 50:
        return 4
    elif eta <= 80:
        return 5
    elif eta <= 120:
        return 6
    else:
        return 7


async def handle_client(reader, writer):
    global budgets
    global conn_player
    count = get_conn_count()
    print('Connection', count, ': OPEN')
    writer.write(b"ack\n")
    while True:
        try:
            data = (await reader.read(16384))
        except ConnectionResetError as e:
            print(e)
            break
        if not data:
            break
        decoded = data.decode('utf-8')
        # decide what to do based on first word
        if re.search("^budget", decoded):
            budgets = [int(i) for i in decoded.split()[1:]]
            writer.write(b"ack\n")
        elif re.search("^utt", decoded):
            handle_unit_type_table(json.loads(decoded.split('\n')[1]))
            writer.write(b"ack\n")
        elif re.search("^preGameAnalysis", decoded):
            lines = decoded.split('\n')
            ms = int(lines[0].split()[1])
            handle_pre_game_analysis(json.loads(lines[1]), ms, count)
            writer.write(b"ack\n")
        elif re.search("^getAction", decoded):
            lines = decoded.split('\n')
            player = int(lines[0].split()[1])
            action = handle_get_action(json.loads(lines[1]), player, count)
            message = action + '\n'
            writer.write(message.encode('utf-8'))
        elif re.search("^gameOver", decoded):
            handle_game_over(int(decoded.split()[1]), count)
            writer.write(b"ack\n")
        else:
            print('Message type not recognized!')
            print(decoded)

    # if the loop exits then the connection is closed
    print('Connection', count, ': CLOSED')
    conn_player.pop(count, None)


def main():
    global rl_agent
    # TODO: for now just load the config once, no batches
    # load configuration
    with open('mrts_config.json', 'r') as fp:
        config = json.load(fp=fp)

    # save a copy of the configuration file being used for a run in the run's folder (first time only)
    restore = True
    if not os.path.exists(config['model_dir']):
        restore = False
    if not restore:
        os.makedirs(config['model_dir'])
        with open(config['model_dir'] + '/config.json', 'w+') as fp:
            fp.write(json.dumps(config, indent=4))

    with tf.Session() as sess:
        rl_agent = DQNAgent(sess, config, restore)
        print('Ready to accept incoming connections')
        loop = asyncio.get_event_loop()
        loop.create_task(asyncio.start_server(handle_client, HOST, PORT))
        loop.run_forever()


if __name__ == '__main__':
    main()
