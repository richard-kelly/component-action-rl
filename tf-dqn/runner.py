import numpy as np
import json
import os
import shutil
import sys

from absl import flags

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env.environment import StepType
from pysc2.env import sc2_env
from pysc2.lib import features

from dqn_agent import DQNAgent
import utils

# Needed to satisfy something in pysc2, though I'm not actually using the flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

FUNCTIONS = actions.FUNCTIONS

# masking the actions functions so only these actions can be taken
relevant_actions = [0, 2, 3, 331]

# load configuration
with open('config.json', 'r') as fp:
    config = json.load(fp=fp)

# save a copy of the configuration files being used for a run in the run's folder (first time only)
restore = True
if not os.path.exists(config['model_dir']):
    restore = False
    os.makedirs(config['model_dir'])
    shutil.copy2('config.json', config['model_dir'])


def get_action_function(obs, action):
    # id = action['function']
    # Masked actions instead
    id = relevant_actions[action['function']]

    if id not in obs.observation['available_actions']:
        # no_op
        return actions.FunctionCall(0, [])

    args = []
    for i in range(len(FUNCTIONS[id].args)):
        name = FUNCTIONS[id].args[i].name
        # special case of func id 3, select_rect, used only if 'screen2' isn't output by network
        # just select a rectangle around the point given by 'screen'
        if id == 3 and (name == 'screen' or name == 'screen2') and 'screen2' not in action:
            half_rect = 5
            x, y = get_screen_coords(action['screen'])
            if name == 'screen':
                args.append([max(x - half_rect, 0), max(y - half_rect, 0)])
            elif name == 'screen2':
                s = config['environment_properties']['screen_size']
                args.append([min(x + half_rect, s - 1), min(y + half_rect, s - 1)])
        else:
            if name == 'screen':
                x, y = get_screen_coords(action['screen'])
                args.append([x, y])
            elif name == 'screen2':
                x, y = get_screen_coords(action['screen2'])
                args.append([x, y])
            elif name not in action:
                # if network doesn't supply argument, uses first choice, which is usually default no modifier action
                args.append([0])
            else:
                args.append([action[name]])
    return actions.FunctionCall(id, args)


def get_screen_coords(val):
    y = val // config['environment_properties']['screen_size']
    x = val % config['environment_properties']['screen_size']
    return x, y


def preprocess_state(obs):
    state = dict()

    state['screen'] = utils.one_hot_encode_int_arrays(
        (obs.observation['feature_screen'].player_relative, 4),
        (obs.observation['feature_screen'].selected, 1)
    )

    # avail_actions = np.zeros(len(FUNCTIONS))
    # avail_actions[obs.observation['available_actions']] = 1
    # state['available_actions'] = avail_actions

    return state


def main():
    agent = MineralsAgent()

    screen = config['environment_properties']['screen_size']
    minimap = config['environment_properties']['minimap_size']

    with sc2_env.SC2Env(
        map_name=config['environment_properties']['map_name'],
        players=[sc2_env.Agent(sc2_env.Race['random'], None)],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen, minimap)
        ),
        visualize=config['environment_properties']['visualize'],
        step_mul=config['environment_properties']['step_mul']
    ) as env:
        agent.reset()
        agent.setup(env.observation_spec(), env.action_spec())
        obs = env.reset()
        for _ in range(1000000):
            obs = env.step([agent.step(obs[0])])


class MineralsAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.rl_agent = DQNAgent(restore)

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

    def step(self, obs):
        super().step(obs)
        state = preprocess_state(obs)

        terminal = True if obs.step_type is StepType.LAST else False

        if self.steps > 1:
            self.rl_agent.observe(terminal=terminal, reward=obs.reward)

        action = self.rl_agent.act(state)

        action_for_sc = get_action_function(obs, action)

        return action_for_sc


if __name__ == "__main__":
    main()
