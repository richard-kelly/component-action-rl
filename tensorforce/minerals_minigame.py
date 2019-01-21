import sys
import numpy as np
import utils
import json

from tensorforce.agents import Agent

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env.environment import StepType

FUNCTIONS = actions.FUNCTIONS

# masking the actions functions so only these actions can be taken
relevant_actions = [0, 1, 2, 3, 331]

with open('config.json', 'r') as fp:
    config = json.load(fp=fp)

# to use:
# python -m pysc2.bin.agent --map CollectMineralShards --agent minerals_minigame.TestAgent

class TestAgent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.validLastAction = False

    def preprocess_state(self, obs):
        state = dict()

        # just using some features that should give minimum info needed for collect minerals minigame
        player_relative = utils.one_hot_encode_int_array(obs.observation['feature_screen'].player_relative, 5)
        # everything is 3 dimensional before concat
        selected = np.expand_dims(obs.observation['feature_screen'].selected, axis=0)
        state['screen'] = np.concatenate((player_relative, selected), axis=0)

        obs.observation['available_actions']
        avail_actions = []
        for i in range(len(FUNCTIONS)):
            if i in obs.observation['available_actions']:
                avail_actions.append(1)
            else:
                avail_actions.append(0)
        state['available_actions'] = np.array(avail_actions)

        return state

    def getActionSpec(self, action_spec):
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

    def getStateSpec(self, obs_spec):
        states = dict(
            screen=dict(type='float', shape=(6, 84, 84)),
            available_actions=dict(type='float', shape=(len(FUNCTIONS)))
        )
        return states

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

        with open(config['network_spec_file'], 'r') as fp:
            network_spec = json.load(fp=fp)

        with open(config['agent_spec_file'], 'r') as fp:
            agent_spec = json.load(fp=fp)

        self.tfAgent = Agent.from_spec(
            spec=agent_spec,
            kwargs=dict(
                states=self.getStateSpec(obs_spec),
                actions=self.getActionSpec(action_spec),
                network=network_spec,
                saver=dict(
                    directory=config['model_dir'],
                    seconds=6000
                ),
                summarizer=dict(
                    directory=config['model_dir'],
                    seconds=30,
                    labels=[
                        'configuration',
                        'losses',
                        'inputs',
                        'gradients_scalar'
                    ]
                )
            )
        )

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
            args.append(action[FUNCTIONS[id].args[i].name].tolist())
        # print(m + 'Valid action: ' + FUNCTIONS[id].name)
        return actions.FunctionCall(id, args), True

    def step(self, obs):
        super().step(obs)
        state = self.preprocess_state(obs)

        #print(obs.observation['game_loop'])
        #print('reward: ', obs.reward)
        terminal = True if obs.step_type is StepType.LAST else False

        if self.steps > 1:
            if self.validLastAction:
                self.tfAgent.observe(terminal=terminal, reward=obs.reward)
            else:
                # self.tfAgent.observe(terminal=False, reward=obs.reward - 10e-3)
                self.tfAgent.observe(terminal=terminal, reward=obs.reward)

        action = self.tfAgent.act(state)

        action_for_sc, self.validLastAction = self.getActionFunction(obs, action)

        return action_for_sc
