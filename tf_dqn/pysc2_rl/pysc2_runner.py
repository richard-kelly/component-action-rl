import numpy as np
import json
import os
import sys
import datetime
import random
import tensorflow as tf

from absl import flags
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env.environment import StepType
from pysc2.env import sc2_env

from tf_dqn.pysc2_rl.dqn_agent import DQNAgent
from tf_dqn.common import utils

# Isn't used here, but allows pysc2 to use the maps
from tf_dqn.pysc2_rl.maps import MeleeMaps

# Needed to satisfy something in pysc2, though I'm not actually using the flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def get_action_function(obs, action, actions_in_use, screen_size, half_rect=20):
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
    return actions.FunctionCall(func_id, args)


def get_screen_coords(val, screen_size):
    y = val // screen_size
    x = val % screen_size
    return x, y


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


def process_config_env(config):
    config['env']['computed_action_list'] = compute_action_list(config['env']['action_functions'])

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

    config['env']['computed_action_components'] = all_components
    return config


def preprocess_state(obs, actions_in_use):
    avail_actions = np.in1d(actions_in_use, obs.observation['available_actions'])

    state = dict(
        screen_player_relative=obs.observation['feature_screen'].player_relative,
        screen_selected=obs.observation['feature_screen'].selected,
        screen_unit_hit_points=obs.observation['feature_screen'].unit_hit_points,
        available_actions=avail_actions
    )
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

    max_ep_score = None
    all_ep_scores = []
    num_eps = 20
    last_n_ep_score = []

    if output_file is not None and not os.path.isfile(output_file):
        with open(output_file, 'a+') as f:
            params = ''
            for name in run_variables:
                params += name + ' '
            f.write('Run_Name Run_num ' + params + 'Max_Score Avg_Score Last_' + str(num_eps) + '_Score\n')

    with sc2_env.SC2Env(
            map_name=config['env']['map_name'],
            players=[sc2_env.Agent(sc2_env.Race['random'], None)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(config['env']['screen_size'], config['env']['minimap_size'])
            ),
            visualize=config['env']['visualize'],
            step_mul=config['env']['step_mul']
    ) as env:
        tf.reset_default_graph()
        with tf.Session() as sess:
            rl_agent = DQNAgent(sess, config, restore)
            # observations from the env are tuples of 1 Timestep per player
            obs = env.reset()[0]
            step = 0
            episode = 1
            episode_reward = 0
            while (config['max_steps'] == 0 or step <= config['max_steps']) and (config['max_episodes'] == 0 or episode <= config['max_episodes']):
                step += 1
                state = preprocess_state(obs, config['env']['computed_action_list'])
                available_actions = obs.observation['available_actions']

                episode_reward += obs.reward
                if obs.step_type is StepType.LAST:
                    terminal = True
                    print("Episode", episode, "finished. Score:", episode_reward)
                    if len(last_n_ep_score) == num_eps:
                        last_n_ep_score.pop(0)
                    last_n_ep_score.append(episode_reward)
                    all_ep_scores.append(episode_reward)
                    if max_ep_score is None or episode_reward > max_ep_score:
                        max_ep_score = episode_reward
                    episode_reward = 0
                    episode += 1
                else:
                    terminal = False

                if step > 1:
                    rl_agent.observe(terminal=terminal, reward=obs.reward)

                action = rl_agent.act(state, available_actions)
                action_for_sc = get_action_function(
                    obs,
                    action,
                    config['env']['computed_action_list'],
                    config['env']['screen_size'],
                    half_rect=config['env']['select_rect_half_size']
                )
                # actions passed into env.step() are in a list with one action per player
                obs = env.step([action_for_sc])[0]

    if output_file is not None:
        with open(output_file, 'a+') as f:
            avg_last = sum(last_n_ep_score) / len(last_n_ep_score)
            avg = sum(all_ep_scores) / len(all_ep_scores)
            run_var_vals = ''
            for _, val in run_variables.items():
                run_var_vals += ' ' + '{:.2e}'.format(val)
            f.write(config['model_dir'] + ' ' + str(run_num) + run_var_vals + ' ' + str(max_ep_score) + ' ' + str(avg) + ' ' + str(avg_last) + '\n')


def main():
    # load configuration
    with open('pysc2_config.json', 'r') as fp:
        config = json.load(fp=fp)

    config = process_config_env(config)

    # load batch config file
    with open('batch.json', 'r') as fp:
        batch = json.load(fp=fp)

    if config['use_batch']:
        base_name = config['model_dir']
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        summary_file_name = base_name + '/_' + time + '_batch_summary.csv'
        count = 0
        while True:
            count += 1
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
            config['model_dir'] = base_name + '/' + name
            print('****** Starting a new run in this batch: ' + name + ' ******')
            run_one_env(config, count, run_variables, rename_if_duplicate=True, output_file=summary_file_name)
    else:
        run_one_env(config, rename_if_duplicate=False)


if __name__ == "__main__":
    main()
