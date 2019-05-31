import os
import asyncio
import subprocess
import datetime
import json
import re
import random
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
unit_type_names = ['Base', 'Barracks', 'Worker', 'Light', 'Heavy', 'Ranged', 'Resource']
one_obs_per_turn = False

rl_agent = None
config = None
loop = None
env = None
step = 0
episodes = 0
eval_step = 0
eval_episodes = 0
eval_mode = False

output_file = None
num_eps = 200
max_ep_score = None
all_ep_scores = []
last_n_ep_score = []


def get_conn_count():
    global conn_count
    conn_count += 1
    return conn_count


def handle_game_over(winner, conn_num):
    global rl_agent
    global all_ep_scores, last_n_ep_score, max_ep_score, episodes, eval_episodes

    episodes += 1
    eval_episodes += 1

    if winner == -1:
        reward = 0
        print('Connection', conn_num, ': GAME OVER - DRAW')
    elif winner == conn_player[conn_num]:
        reward = 1
        print('Connection', conn_num, ': GAME OVER - WON')
    else:
        reward = -1
        print('Connection', conn_num, ': GAME OVER - LOST')

    if len(last_n_ep_score) == num_eps:
        last_n_ep_score.pop(0)
    last_n_ep_score.append(reward)
    all_ep_scores.append(reward)
    if max_ep_score is None or reward > max_ep_score:
        max_ep_score = reward
    rl_agent.observe(conn_num, terminal=True, reward=reward, record=eval_mode)


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


def handle_pre_game_analysis(unused_state, ms, conn_num):
    # nothing to do with this for now
    # ms is the number of ms we have to exit this function (and send back a response, which happens elsewhere)
    # state is the starting state of the game (t=0)
    print('Connection', conn_num, ': Received pre-game analysis state for', ms, 'ms.')


def handle_get_action(state, player, state_eval, conn_num):
    global conn_player
    global step, eval_step

    eval_step += 1

    conn_player[conn_num] = player
    state_for_rl = {}
    # state:
    #   map (0, 0) is top left
    game_frame = state['time']
    if game_frame == 0:
        rl_agent.reset(conn_num)
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

    # list, each player is dict with ints "ID" and "resources"
    # the ID here is the same as player parameter to this function and "player" in unit object below
    players = state['pgs']['players']
    our_resources = players[player]['resources']
    their_resources = players[(player + 1) % 2]['resources']

    # list, each unit is dict with string "type" and
    # ints "ID", "player", "x", "y", "resources", "hitpoints"
    # ID here is a unique id for each unit in the game
    # resources is resources that a worker is carrying or that are in a resource patch
    # convert to dict indexed by ID
    units = {}
    friendly_unit_id_by_coordinates = {}
    for unit in state['pgs']['units']:
        units[unit['ID']] = unit
        if player == unit['player']:
            friendly_unit_id_by_coordinates[(unit['x'], unit['y'])] = unit['ID']

    # actions ongoing for both players. A list with dicts:
    #   "ID":       int [unit ID],
    #   "time":     int [game frame when the action was given]
    #   "action":   {type, parameter, unitType, etc.]
    # convert to dict indexed by ID
    current_actions = {}
    for ongoing_action in state['actions']:
        current_actions[ongoing_action['ID']] = ongoing_action

    state_for_rl['terrain'] = terrain
    state_for_rl['available_resources'] = np.array(our_resources, dtype=np.int32)
    state_for_rl['player_resources'] = get_players_resources_array(our_resources, their_resources)
    state_for_rl['players'] = get_players_feature(units, map_size, player)

    eta_feature = np.zeros((map_size, map_size), dtype=np.int8)
    for ID, current_action in current_actions.items():
        unit = units[current_action['ID']]
        eta = get_eta(game_frame, current_action, unit['type'])
        eta_cat = get_eta_category(eta)
        eta_feature[unit['y'], unit['x']] = eta_cat
    state_for_rl['eta'] = eta_feature

    resources_feature = np.zeros((map_size, map_size), dtype=np.int8)
    for _, unit in units.items():
        resources_feature[unit['y'], unit['x']] = get_resource_category(unit['resources'])
    state_for_rl['resources'] = resources_feature

    units_feature = np.zeros((map_size, map_size), dtype=np.int8)
    for _, unit in units.items():
        units_feature[unit['y'], unit['x']] = unit_type_names.index(unit['type']) + 1
    state_for_rl['units'] = units_feature

    # TODO: try other representations of health... normalized real number? thermometer encoding?
    health_feature = np.zeros((map_size, map_size), dtype=np.int8)
    for _, unit in units.items():
        health_feature[unit['y'], unit['x']] = get_health_category(unit['hitpoints'])
    state_for_rl['health'] = health_feature

    for _, ongoing_action in current_actions.items():
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
        # resources are not deducted until a unit is finished being built.
        # so we deduct them to make sure we don't overbuild
        # TODO: do for enemy as well?
        if ongoing_action['action']['type'] == 4 and units[ongoing_action['ID']]['player'] == player:
            cost = unit_types[ongoing_action['action']['unitType']]['cost']
            state_for_rl['available_resources'] -= cost
            our_resources -= cost
            state_for_rl['player_resources'] = get_players_resources_array(our_resources, their_resources)
    # TODO: add more information about current actions (type, target, etc.)

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

    num_actions = len(friendly_units_without_actions)
    if num_actions == 0:
        return json.dumps(actions)
    remembered_action = random.randrange(num_actions)
    for i in range(num_actions):
        remember = i == remembered_action or not one_obs_per_turn
        if step > 0:
            if remember:
                reward = state_eval if config['env']['use_shaped_rewards'] else 0
                rl_agent.observe(conn_num, terminal=False, reward=reward)
        step += 1
        if eval_mode:
            force_eps = config['self_play_epsilon']
        else:
            force_eps = None
        action = rl_agent.act(conn_num, state_for_rl, remember, force_eps)

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
            # reduce available resources so that we don't try to over build in a turn
            cost = unit_types[unit_action['unitType']]['cost']
            state_for_rl['available_resources'] -= cost
            our_resources -= cost
            state_for_rl['player_resources'] = get_players_resources_array(our_resources, their_resources)

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


def get_players_feature(units, map_size, player):
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
    return players_feature


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


def get_health_category(hitpoints):
    if hitpoints <= 4:
        return hitpoints
    else:
        return 5


def get_resource_category(resources):
        if resources <= 5:
            return resources
        elif resources <= 9:
            return 6
        else:
            return 7


def get_players_resources_array(self_resources, enemy_resources):
    # should be highest cat number plus one (includes zero)
    num_cats = 8
    self_cat = get_resource_category(self_resources)
    enemy_cat = get_resource_category(enemy_resources)
    array = np.zeros(2 * num_cats, dtype=np.int32)
    array[self_cat] = 1
    array[num_cats + enemy_cat] = 1
    return array


async def handle_client(reader, writer):
    global budgets
    global conn_player
    global env, eval_step, eval_episodes, eval_mode
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

        # check if time to quit this training session
        over_steps = config['max_steps'] != 0 and rl_agent.get_global_step() >= config['max_steps']
        over_episodes = config['max_episodes'] != 0 and episodes >= config['max_episodes']
        if over_steps or over_episodes:
            if output_file is not None:
                with open(output_file, 'a+') as f:
                    avg_last = sum(last_n_ep_score) / len(last_n_ep_score)
                    avg = sum(all_ep_scores) / len(all_ep_scores)
                    f.write(config['model_dir'] + ' ' + str(max_ep_score) + ' ' + str(avg) + ' ' + str(avg_last) + '\n')
            loop.stop()
            break

        if config['self_play']:
            if not eval_mode:
                over_steps = config['self_play_eval_freq_steps'] != 0 and eval_step >= config['self_play_eval_freq_steps']
                over_episodes = config['self_play_eval_freq_episodes'] != 0 and eval_episodes >= config['self_play_eval_freq_episodes']
                if over_steps or over_episodes:
                    eval_mode = True
                    eval_step = 0
                    eval_episodes = 0
                    env.kill()
                    env = get_env(config['self_play_eval_env'])
                    continue
            if eval_mode:
                over_steps = config['self_play_eval_duration_steps'] != 0 and eval_step >= config['self_play_eval_duration_steps']
                over_episodes = config['self_play_eval_duration_episodes'] != 0 and eval_episodes >= config['self_play_eval_duration_episodes']
                if over_steps or over_episodes:
                    eval_mode = False
                    eval_step = 0
                    eval_episodes = 0
                    env.kill()
                    env = get_env(config['env'])
                    continue

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
            state_eval = float(lines[1])
            action = handle_get_action(json.loads(lines[2]), player, state_eval, count)
            message = action + '\n'
            writer.write(message.encode('utf-8'))
        elif re.search("^gameOver", decoded):
            handle_game_over(int(decoded.split()[1]), count)
            writer.write(b"ack\n")
            writer.close()
        else:
            print('Message type not recognized!')
            print(decoded)

    # if the loop exits then the connection is closed
    print('Connection', count, ': CLOSED')
    conn_player.pop(count, None)


def get_env(env_config):
    num_maps = 0
    maps = []
    for map in os.listdir(env_config['map_folder']):
        num_maps += 1
        maps.append(env_config['map_folder'] + '/' + map)
    # TODO: work out correct formula for iterations
    iterations = 10000
    args = [
        "java",
        "-jar",
        "microrts.jar",
        env_config['unit_type_table'],
        env_config['selected_ai'],
        str(iterations),
        str(env_config['max_game_length']),
        str(env_config['time_budget']),
        str(env_config['pre_analysis_budget']),
        'true' if env_config['full_observability'] else 'false',
        'true' if env_config['store_traces'] else 'false',
        str(len(env_config['opponents']))
    ]
    args += env_config['opponents'] + [str(num_maps)] + maps

    return subprocess.Popen(args, stdout=None, stderr=None)


def run_one_env(config, rename_if_duplicate=False, server_only=False):
    global rl_agent, loop, env, eval_mode
    global all_ep_scores, last_n_ep_score, max_ep_score, episodes, eval_episodes

    restore = True
    if not os.path.exists(config['model_dir']):
        restore = False
    elif rename_if_duplicate:
        restore = False
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        config['model_dir'] = config['model_dir'] + '_' + time
    if not restore:
        os.makedirs(config['model_dir'])
        with open(config['model_dir'] + '/config.json', 'w+') as fp:
            fp.write(json.dumps(config, indent=4))

    tf.reset_default_graph()
    with tf.Session() as sess:
        rl_agent = DQNAgent(sess, config, restore, 'normal')
        print('RL agent ready.')

        # metrics
        max_ep_score = -1
        all_ep_scores = []
        last_n_ep_score = []
        episodes = 0
        eval_episodes = 0

        # if not doing self play always record episode summaries
        eval_mode = not config['self_play']

        # start env
        if not server_only:
            env = get_env(config['env'])
        print('Environment started, about to start the loop')

        loop = asyncio.get_event_loop()
        server = asyncio.start_server(handle_client, HOST, PORT, reuse_address=True)
        loop.create_task(server)
        loop.run_forever()

        # loop exits once enough episodes or steps have passed
        server.close()
        if not server_only:
            env.kill()


def main():
    global one_obs_per_turn
    global config
    global output_file

    # TODO: for now just load the config once, no batches
    # load configuration
    with open('mrts_config.json', 'r') as fp:
        config = json.load(fp=fp)
    one_obs_per_turn = config['env']['one_obs_per_turn']

    # load batch config file
    with open('batch.json', 'r') as fp:
        batch = json.load(fp=fp)

    if batch['use']:
        base_name = config['model_dir']
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_file = base_name + '/_' + time + '_batch_summary.txt'

        if not os.path.exists(base_name):
            os.mkdir(base_name)

        if not os.path.isfile(output_file):
            with open(output_file, 'a+') as f:
                f.write('Run_Name Max_Score Avg_Score Last_' + str(num_eps) + '_Score\n')
        count = 0
        while True:
            count += 1
            name = str(count)
            for param in batch['log_random']:
                config[param] = utils.log_uniform(batch['log_random'][param]['min'], batch['log_random'][param]['max'])
                name += '_' + param + '_' + '{:.2e}'.format(config[param])
            for param in batch['random']:
                config[param] = random.uniform(batch['random'][param]['min'], batch['random'][param]['max'])
                name += '_' + param + '_' + '{:.2e}'.format(config[param])
            config['model_dir'] = base_name + '/' + name
            print('****** Starting a new run in this batch: ' + name + ' ******')
            run_one_env(config, rename_if_duplicate=True)
    else:
        if config['server_only_no_env']:
            # run just the server and don't start a mrts tournament
            run_one_env(config, rename_if_duplicate=False, server_only=True)
        else:
            run_one_env(config, rename_if_duplicate=False)


if __name__ == '__main__':
    main()
