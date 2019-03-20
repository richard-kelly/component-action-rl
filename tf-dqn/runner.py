import numpy as np
import json
import os
import shutil
import sys
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


def get_action_function(obs, action, actions_in_use, screen_size):
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
            half_rect = 5
            x, y = get_screen_coords(action['screen'])
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


def run_one_env(config, rename_if_duplicate):
    # save a copy of the configuration file being used for a run in the run's folder (first time only)
    restore = True
    if not os.path.exists(config['model_dir']):
        restore = False
        os.makedirs(config['model_dir'])
        shutil.copy2('config.json', config['model_dir'])

    with sc2_env.SC2Env(
            map_name=config['env']['map_name'],
            players=[sc2_env.Agent(sc2_env.Race['random'], None)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(config['env']['screen_size'], config['env']['minimap_size'])
            ),
            visualize=config['env']['visualize'],
            step_mul=config['env']['step_mul']
    ) as env:
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
                    config['env']['screen_size']
                )
                # actions passed into env.step() are in a list with one action per player
                obs = env.step([action_for_sc])[0]

def main():
    # load configuration
    with open('config.json', 'r') as fp:
        config = json.load(fp=fp)


if __name__ == "__main__":
    main()