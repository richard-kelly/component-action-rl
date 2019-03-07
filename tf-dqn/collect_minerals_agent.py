import numpy as np
import json
import os
import shutil

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env.environment import StepType

from dqn_agent import DQNAgent
import utils

'''
to use:
python -m pysc2.bin.agent --map CollectMineralShards --agent collect_minerals_agent.MineralsAgent
'''

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


class MineralsAgent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.validLastAction = False
        self.rl_agent = DQNAgent(restore)

    def preprocess_state(self, obs):
        state = dict()

        state['screen'] = utils.one_hot_encode_int_arrays(
            (obs.observation['feature_screen'].player_relative, 4),
            (obs.observation['feature_screen'].selected, 1)
        )

        # avail_actions = np.zeros(len(FUNCTIONS))
        # avail_actions[obs.observation['available_actions']] = 1
        # state['available_actions'] = avail_actions

        return state

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

    def getActionFunction(self, obs, action):
        # id = action['function']
        # Masked actions instead
        id = relevant_actions[action['function']]

        m = 'Ep. ' + str(self.episodes) + '/Step ' + str(self.steps) + ' '

        if id not in obs.observation['available_actions']:
            # no_op
            # print(m + 'FAIL Invalid action: ' + FUNCTIONS[id].name)
            return actions.FunctionCall(0, []), False

        args = []
        for i in range(len(FUNCTIONS[id].args)):
            name = FUNCTIONS[id].args[i].name
            # special case of func id 3, select_rect, used only if 'screen2' isn't output by network
            # just select a rectangle around the point given by 'screen'
            if id == 3 and (name == 'screen' or name == 'screen2') and 'screen2' not in action:
                half_rect = 5
                x, y = self.getScreenCoords(action['screen'])
                if name == 'screen':
                    args.append([max(x - half_rect, 0), max(y - half_rect, 0)])
                elif name == 'screen2':
                    args.append([min(x + half_rect, config['screen_size'] - 1), min(y + half_rect, config['screen_size'] - 1)])
            else:
                if name == 'screen':
                    x, y = self.getScreenCoords(action['screen'])
                    args.append([x, y])
                elif name == 'screen2':
                    x, y = self.getScreenCoords(action['screen2'])
                    args.append([x, y])
                elif name not in action:
                    # if network doesn't supply argument, uses first choice, which is usually default no modifier action
                    args.append([0])
                else:
                    args.append([action[name]])
        # print(m + 'Valid action: ' + FUNCTIONS[id].name)
        return actions.FunctionCall(id, args), True

    def getScreenCoords(self, val):
        y = val // config['screen_size']
        x = val % config['screen_size']
        return x, y

    def step(self, obs):
        super().step(obs)
        state = self.preprocess_state(obs)

        terminal = True if obs.step_type is StepType.LAST else False

        if self.steps > 1:
            self.rl_agent.observe(terminal=terminal, reward=obs.reward)

        action = self.rl_agent.act(state)

        action_for_sc, self.validLastAction = self.getActionFunction(obs, action)

        return action_for_sc
