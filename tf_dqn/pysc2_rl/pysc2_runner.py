import numpy as np
import json
import os
import sys
import datetime
import random
import re
import tensorflow as tf

from absl import flags
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import static_data
from pysc2.lib import units
from pysc2.env.environment import StepType
from pysc2.env import sc2_env

from tf_dqn.pysc2_rl.dqn_agent import DQNAgent
from tf_dqn.common import utils

# Isn't used here, but allows pysc2 to use the maps
from tf_dqn.pysc2_rl.maps import CombatMaps

# Needed to satisfy something in pysc2, though I'm not actually using the flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def get_win_loss(obs):
    # if we're not using any food, then we don't have any units (and have lost)
    if obs.observation['player'][features.Player.food_used] > 0:
        return 1
    else:
        return -1


def get_action_function(obs, action, config):
    actions_in_use = config['env']['computed_action_list']
    screen_size = config['env']['screen_size']
    half_rect = config['env']['select_rect_half_size']

    # id = action['function']
    # Masked actions instead
    func_id = actions_in_use[action['function']]

    if func_id not in obs.observation['available_actions']:
        # no_op because of bad action - should not happen
        print("Action returned by RL agent is not available. Doing no_op.")
        return actions.FunctionCall(0, [])

    args = []
    pysc2_funcs = actions.FUNCTIONS
    for i in range(len(pysc2_funcs[func_id].args)):
        name = pysc2_funcs[func_id].args[i].name
        # special case of func_id 3, select_rect, used only if 'screen2' isn't output by network
        # just select a rectangle around the point given by 'screen'
        if func_id == 3 and (name == 'screen' or name == 'screen2') and 'screen2' not in action:
            x, y = get_screen_coords(action['screen'], screen_size)
            if name == 'screen':
                args.append([max(x - half_rect, 0), max(y - half_rect, 0)])
            elif name == 'screen2':
                args.append([min(x + half_rect, screen_size - 1), min(y + half_rect, screen_size - 1)])
        else:
            if name == 'screen':
                x, y = get_screen_coords(action['screen'], screen_size)
                args.append([x, y])
            elif name == 'screen2':
                x, y = get_screen_coords(action['screen2'], screen_size)
                args.append([x, y])
            elif name == 'minimap':
                x, y = get_screen_coords(action['minimap'], screen_size)
                args.append([x, y])
            elif name not in action:
                # if network doesn't supply argument, uses first choice, which is usually default no modifier action
                args.append([0])
            else:
                args.append([action[name]])

    if config['log_action_descriptions']:
        print(pysc2_funcs[func_id].name, ':', args)
    return actions.FunctionCall(func_id, args)


def get_screen_coords(val, screen_size):
    y = val // screen_size
    x = val % screen_size
    return x, y


def get_unit_ids(unit_list):
    unit_ids = []
    for unit in unit_list:
        if unit in units.Terran.__members__:
            unit_ids.append(int(units.Terran[unit]))
        if unit in units.Protoss.__members__:
            unit_ids.append(int(units.Protoss[unit]))
        if unit in units.Zerg.__members__:
            unit_ids.append(int(units.Zerg[unit]))
    return unit_ids


def compute_action_list(rules):
    # Returns sorted list of integer action function IDs
    allowed_actions = set()
    # first process all includes
    for rule in rules:
        if rule['type'] == 'include_range':
            for i in range(rule['list'][0], rule['list'][1] + 1):
                allowed_actions.add(i)
        elif rule['type'] == 'include_list':
            for i in rule['list']:
                allowed_actions.add(i)
    # then process all excludes
    for rule in rules:
        if rule['type'] == 'exclude_range':
            for i in range(rule['list'][0], rule['list'][1] + 1):
                allowed_actions.remove(i)
        elif rule['type'] == 'exclude_list':
            for i in rule['list']:
                allowed_actions.remove(i)
    computed = list(allowed_actions)
    computed.sort()
    return computed


# Add some extra properties to the config['env'] section that will be used by the Network
def process_config_env(config):
    config['env']['computed_action_list'] = compute_action_list(config['env']['action_functions'])

    # modify the list of computed actions based on the map name for maps that pre-select units or have control groups pre-assigned
    preselected = False
    num_groups = 0
    match = re.search(r"select_all", config['env']['map_name'])
    if match:
        # all units are pre-selected, no groups
        preselected = True
    match = re.search(r"select_(\d+)", config['env']['map_name'])
    if match:
        num_groups = int(match.group(1))

    if preselected or num_groups > 0:
        # remove all selection functions
        select_functions = [2, 3, 4, 5, 6, 7, 8, 9]
        to_remove = []
        for func in config['env']['computed_action_list']:
            if func in select_functions:
                to_remove.append(func)
        for func in to_remove:
            config['env']['computed_action_list'].remove(func)
    if num_groups > 0:
        # if we have preselected groups, add back in control groups
        control_group_func_id = 4
        config['env']['computed_action_list'].append(control_group_func_id)
        # only select the pre-existing control groups
        config['env']['num_control_groups'] = num_groups
    else:
        config['env']['num_control_groups'] = 10

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

    # if some function we are using uses an argument type, we must use it
    for func in actions.FUNCTIONS:
        if int(func.id) in config['env']['computed_action_list']:
            for arg in func.args:
                if arg.name in all_components:
                    all_components[arg.name] = True

    # these are special and can be replaced with default values (or computed differently in case of screen2)
    if not config['env']['use_queue']:
        all_components['queued'] = False
    if not config['env']['use_screen2']:
        all_components['screen2'] = False

    # if we have pre selected control groups, agent should only be able to select/recall a control group (which is 0)
    if num_groups > 0:
        all_components['control_group_act'] = False

    config['env']['computed_action_components'] = all_components

    # compute list of unit ids we care about for unit types
    config['env']['specific_unit_type_ids'] = get_unit_ids(config['env']['specific_unit_types'])

    return config


def preprocess_state(obs, config):
    actions_in_use = config['env']['computed_action_list']
    avail_actions = np.in1d(actions_in_use, obs.observation['available_actions'])

    unit_types = obs.observation['feature_screen'].unit_type
    if config['env']['use_all_unit_types']:
        modified_unit_types = np.zeros(unit_types.shape, dtype=np.int32)
        for i in range(len(static_data.UNIT_TYPES)):
            # skip zero since that is the default value representing no units
            modified_unit_types[unit_types == static_data.UNIT_TYPES[i]] = i + 1
        unit_types = modified_unit_types
    elif config['env']['use_specific_unit_types']:
        modified_unit_types = np.zeros(unit_types.shape, dtype=np.int32)
        unique_units = config['env']['specific_unit_type_ids']
        for i in range(len(unique_units)):
            # skip zero since that is the default value representing no units
            modified_unit_types[unit_types == unique_units[i]] = i + 1
        unit_types = modified_unit_types
    else:
        unit_types = None

    state = dict(
        screen_player_relative=obs.observation['feature_screen'].player_relative,
        screen_selected=obs.observation['feature_screen'].selected,
        screen_unit_hit_points=obs.observation['feature_screen'].unit_hit_points,
        screen_unit_hit_points_ratio=obs.observation['feature_screen'].unit_hit_points_ratio,
        screen_unit_shields=obs.observation['feature_screen'].unit_shields,
        screen_unit_shields_ratio=obs.observation['feature_screen'].unit_shields_ratio,
        available_actions=avail_actions
    )

    if unit_types is not None:
        state['screen_unit_type'] = unit_types

    return state


def run_one_env(config, run_num=0, run_variables={}, rename_if_duplicate=False, output_file=None):
    # save a copy of the configuration file being used for a run in the run's folder (first time only)
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

    num_eps = 20
    max_ep_score = None
    all_ep_scores = []
    last_n_ep_score = []
    all_ep_wins = []
    last_n_ep_wins = []
    win_count = 0

    if output_file is not None and not os.path.isfile(output_file):
        with open(output_file, 'a+') as f:
            params = []
            for name in run_variables:
                params.append(name)
            run_info = ['Run_Name', 'Run_num'] + params
            score_info = ['Max_Score', 'Avg_Score', 'Last_' + str(num_eps) + '_Score']
            win_info = ['Avg_Win_Val', 'Last_' + str(num_eps) + '_Win_Val', 'Win_%']
            header_row = run_info + score_info + win_info
            f.write(','.join(str(val) for val in header_row) + '\n')

    with sc2_env.SC2Env(
        map_name=config['env']['map_name'],
        players=[sc2_env.Agent(sc2_env.Race['random'], None)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(config['env']['screen_size'], config['env']['minimap_size']),
            action_space=actions.ActionSpace.FEATURES
        ),
        visualize=config['env']['visualize'],
        step_mul=config['env']['step_mul'],
        realtime=config['realtime']
    ) as env:
        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            rl_agent = DQNAgent(sess, config, restore)
            # observations from the env are tuples of 1 Timestep per player
            obs = env.reset()[0]
            step = 0
            episode = 1
            episode_reward = 0

            # Rewards from the map have to be integers,
            # and some maps calculate normalized float rewards and then multiply them by some factor.
            match = re.search(r"factor_(\d+)", config['env']['map_name'])
            factor = float(match.group(1)) if match else 1.0

            # For combat micro maps we may have a shaped reward or not, but we independently want to record win/loss
            match = re.match(r"^combat", config['env']['map_name'])
            win_loss = True if match else False

            while (config['max_steps'] == 0 or step <= config['max_steps']) and (config['max_episodes'] == 0 or episode <= config['max_episodes']):
                step += 1
                state = preprocess_state(obs, config)
                available_actions = obs.observation['available_actions']

                step_reward = obs.reward / factor
                episode_reward += step_reward
                win = 0
                if obs.step_type is StepType.LAST:
                    terminal = True
                    # if this map type uses this win/loss calc
                    if win_loss:
                        win = get_win_loss(obs)
                        if win == 1:
                            win_count += 1

                    print("Episode", episode, "finished. Win:", win, "Score:", episode_reward)
                    if len(last_n_ep_score) == num_eps:
                        last_n_ep_score.pop(0)
                        last_n_ep_wins.pop(0)
                    last_n_ep_score.append(episode_reward)
                    last_n_ep_wins.append(win)
                    all_ep_scores.append(episode_reward)
                    all_ep_wins.append(win)
                    if max_ep_score is None or episode_reward > max_ep_score:
                        max_ep_score = episode_reward
                    episode_reward = 0
                    episode += 1
                else:
                    terminal = False

                if step > 1:
                    rl_agent.observe(terminal=terminal, reward=step_reward, win=win)

                action = rl_agent.act(state, available_actions)
                action_for_sc = get_action_function(obs, action, config)
                # actions passed into env.step() are in a list with one action per player
                obs = env.step([action_for_sc])[0]

    if output_file is not None:
        with open(output_file, 'a+') as f:
            avg_last = sum(last_n_ep_score) / len(last_n_ep_score)
            avg = sum(all_ep_scores) / len(all_ep_scores)
            avg_win_last = sum(last_n_ep_wins) / len(last_n_ep_wins)
            avg_wins = sum(all_ep_wins) / len(all_ep_wins)
            run_var_vals = []
            for _, val in run_variables.items():
                run_var_vals.append('{:.2e}'.format(val))
            columns = [config['model_dir'], run_num]
            columns += run_var_vals
            columns.append(max_ep_score)
            columns.append(avg)
            columns.append(avg_last)
            columns.append(avg_wins)
            columns.append(avg_win_last)
            columns.append(win_count / episode)
            f.write(','.join(str(val) for val in columns) + '\n')


def main():
    # load configuration
    config_paths = ['pysc2_config.json']
    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            # if arg is a dir, we have a dir of config files to be run
            config_files = os.listdir(sys.argv[1])
            config_paths = []
            for file in config_files:
                config_paths.append(os.path.join(sys.argv[1], file))
        else:
            # if it's a file, we just want that file
            config_paths = [sys.argv[1]]

    for config_file in config_paths:
        with open(config_file, 'r') as fp:
            config = json.load(fp=fp)

        config = process_config_env(config)

        # load batch config file
        with open(config['batch_file'], 'r') as fp:
            batch = json.load(fp=fp)

        if config['use_batch'] and not config['inference_only']:
            base_name = config['model_dir']
            summary_file_name = base_name + '/batch_summary.csv'
            count = 0

            # if we're restarting a run, continue numbering starting from number at end of file
            if os.path.isfile(summary_file_name):
                with open(summary_file_name, 'r') as summary:
                    for line in summary:
                        words = line.split()
                        if len(words) > 1 and words[0] != 'Run_Name':
                            count = int(words[1])

            for _ in range(config['batch_runs']):
                count += 1
                # for if we are restarting a run (so range above is really useless)
                if count > config['batch_runs']:
                    break
                name = str(count)
                run_variables = {}
                for param in batch['log_random']:
                    new_val = utils.log_uniform(batch['log_random'][param]['min'], batch['log_random'][param]['max'])
                    run_variables[param] = new_val
                    config[param] = new_val
                    name += '_' + param + '_' + '{:.2e}'.format(config[param])
                for param in batch['random']:
                    new_val = random.uniform(batch['random'][param]['min'], batch['random'][param]['max'])
                    run_variables[param] = new_val
                    config[param] = new_val
                    name += '_' + param + '_' + '{:.2e}'.format(config[param])
                for param in batch['random_int']:
                    new_val = random.randint(batch['random_int'][param]['min'], batch['random_int'][param]['max'])
                    run_variables[param] = new_val
                    config[param] = new_val
                    name += '_' + param + '_' + str(new_val)
                config['model_dir'] = base_name + '/' + name
                print('****** Starting a new run in this batch: ' + name + ' ******')
                run_one_env(config, count, run_variables, rename_if_duplicate=True, output_file=summary_file_name)
        else:
            run_one_env(config, rename_if_duplicate=False)


if __name__ == "__main__":
    main()
