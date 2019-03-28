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


from dqn_agent import DQNAgent
import utils

# Needed to satisfy something in pysc2, though I'm not actually using the flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)


def get_action_function(obs, action, actions_in_use, screen_size, half_rect=20):
    # id = action['function']
    # Masked actions instead
    id = actions_in_use[action['function']]

    if id not in obs.observation['available_actions']:
        # no_op - should not happen
        print("Action returned by RL agent is not available. Doing no_op.")
        return actions.FunctionCall(0, [])

    args = []
    pysc2_funcs = actions.FUNCTIONS
    for i in range(len(pysc2_funcs[id].args)):
        name = pysc2_funcs[id].args[i].name
        # special case of func id 3, select_rect, used only if 'screen2' isn't output by network
        # just select a rectangle around the point given by 'screen'
        if id == 3 and (name == 'screen' or name == 'screen2') and 'screen2' not in action:
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
            elif name not in action:
                # if network doesn't supply argument, uses first choice, which is usually default no modifier action
                args.append([0])
            else:
                args.append([action[name]])
    return actions.FunctionCall(id, args)


def get_screen_coords(val, screen_size):
    y = val // screen_size
    x = val % screen_size
    return x, y


def preprocess_state(obs):
    # avail_actions = np.zeros(len(FUNCTIONS))
    # avail_actions[obs.observation['available_actions']] = 1

    state = dict(
        screen_player_relative=obs.observation['feature_screen'].player_relative,
        screen_selected=obs.observation['feature_screen'].selected,
        # available_actions=avail_actions
    )
    return state


def run_one_env(config, rename_if_duplicate=False, output_file=None):
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
    last_10_ep_score = []

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
                state = preprocess_state(obs)
                available_actions = dict(
                    function=obs.observation['available_actions']
                )

                episode_reward += obs.reward
                if obs.step_type is StepType.LAST:
                    terminal = True
                    print("Episode", episode, "finished. Score:", episode_reward)
                    if len(last_10_ep_score) == 10:
                        last_10_ep_score.pop(0)
                    last_10_ep_score.append(episode_reward)
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
                    config['env']['action_list']['function'],
                    config['env']['screen_size'],
                    half_rect=config['env']['select_rect_half_size']
                )
                # actions passed into env.step() are in a list with one action per player
                obs = env.step([action_for_sc])[0]

    if output_file:
        with open(output_file, 'a+') as f:
            avg = sum(last_10_ep_score) / 10
            f.write(config['model_dir'] + ' ' + str(max_ep_score) + ' ' + str(avg) + '\n')


def main():
    # load configuration
    with open('pysc2_config.json', 'r') as fp:
        config = json.load(fp=fp)

    # load batch config file
    with open('batch.json', 'r') as fp:
        batch = json.load(fp=fp)

    if batch['use']:
        base_name = config['model_dir']
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
            run_one_env(config, rename_if_duplicate=True, output_file=base_name + '/batch_summary.txt')
    else:
        run_one_env(config, rename_if_duplicate=False)


if __name__ == "__main__":
    main()
