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
        self.rewardCount = 0
        self.episodeCount = 0
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

    def format_all_actions_spec(self, action_spec):
        # ALL ACTIONS
        # all_actions = {'function': dict(type='int', num_actions=len(action_spec.functions))}

        # version with limited action functions to speed up learning of simple minigame
        all_actions = {'function': dict(type='int', num_actions=len(relevant_actions))}
        for argument_type in action_spec.types:
            # the only arguments with a shape that isn't (1) are the screen and minimap ones,
            # so we're assuming the screen/minimap dimensions are square here
            spec = dict(type='int', shape=(len(argument_type.sizes),), num_actions=argument_type.sizes[0])
            all_actions[argument_type.name] = spec
        return all_actions

    def format_some_actions_spec(self, action_spec):
        return dict(
            function=dict(type='int', num_actions=len(relevant_actions)),
            screen=dict(type='int', shape=(1,), num_actions=84),
            screen2=dict(type='int', shape=(1,), num_actions=84),
            select_point_act=dict(type='int', shape=(1,), num_actions=4),
            select_add=dict(type='int', shape=(1,), num_actions=2),
            queued=dict(type='int', shape=(1,), num_actions=2)
        )

    def getStateSpec(self, obs_spec):
        states = dict(
            screen=dict(type='float', shape=(84, 84, 5)),
            # available_actions=dict(type='float', shape=(len(FUNCTIONS)))
        )
        return states

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
            if FUNCTIONS[id].args[i].name == 'screen':
                x, y = self.getScreenCoords(action['screen'])
                args.append([x, y])
            elif FUNCTIONS[id].args[i].name == 'screen2':
                x, y = self.getScreenCoords(action['screen2'])
                args.append([x, y])
            elif FUNCTIONS[id].args[i].name == 'queued':
                # no queueing
                args.append([0])
            else:
                args.append([action[FUNCTIONS[id].args[i].name]])
        # print(m + 'Valid action: ' + FUNCTIONS[id].name)
        return actions.FunctionCall(id, args), True

    def getScreenCoords(self, val):
        y = val // config['screen_size']
        x = val % config['screen_size']
        return x, y

    def step(self, obs):
        super().step(obs)
        state = self.preprocess_state(obs)

        self.rewardCount += obs.reward

        # writes out the average reward every 100 episodes
        if obs.step_type is StepType.LAST:
            self.episodeCount += 1
            if self.episodeCount == 100:
                out = open(config['model_dir'] + '/rewards.txt', 'a')
                out.write(str(self.steps) + ", " + str((self.rewardCount / 100)) + '\n')
                out.close()
                self.rewardCount = 0
                self.episodeCount = 0

        # print(obs.observation['game_loop'])
        # print('reward: ', obs.reward)
        terminal = True if obs.step_type is StepType.LAST else False

        if self.steps > 1:
            self.rl_agent.observe(terminal=terminal, reward=obs.reward)

        action = self.rl_agent.act(state)

        action_for_sc, self.validLastAction = self.getActionFunction(obs, action)

        return action_for_sc
