import sys
import numpy as np
import utils

from tensorforce.agents import DQNAgent

from pysc2.agents import base_agent
from pysc2.lib import actions

FUNCTIONS = actions.FUNCTIONS

# to use:
# python -m pysc2.bin.agent --map CollectMineralShards --agent minerals_minigame.TestAgent

'''
network_spec = [
    [
        dict(type='input', inputs=['screen']),
        dict(type='embedding', indices=2, size=64),
        dict(type='flatten'),
        dict(type='dense', size=64),
        dict(type='output', output='screen_net')
    ],
    [

        dict(type='input', inputs=['screen_net']),
        dict(type='dense', size=64),
        dict(type='output', output='output')
    ]
]
'''
'''[
        dict(type='input', inputs=['minimap']),
        dict(type='dense', size=64),
        dict(type='output', output='minimap_net')
    ],
    [
        dict(type='input', inputs=['player', 'game_loop', 'score_cumulative', 'available_actions', 'single_select', 'control_groups']),
        dict(type='dense', size=64),
        dict(type='output', output='non_spatial_net')
],'''
# dict(type='input', inputs=['screen_net', 'minimap_net', 'non_spatial_net']),
# modify obs_spec to fit right format

'''
actions (spec, or dict of specs): Actions specification, with the following attributes (required):
    - type: one of 'bool', 'int', 'float' (required).
    - shape: integer, or list/tuple of integers (default: []).
    - num_actions: integer (required if type == 'int').
    - min_value and max_value: float (optional if type == 'float', default: none).
'''


class TestAgent(base_agent.BaseAgent):

    def __init__(self):
        super().__init__()
        self.validLastAction = False

    def preprocess_state(self, obs):
        # just using some features that should give minimum info needed for collect minerals minigame
        player_relative = utils.one_hot_encode_int_array(obs.observation['feature_screen'].player_relative, 5)
        # everything is 3 dimensional before concat
        selected = np.expand_dims(obs.observation['feature_screen'].selected, axis=0)
        return np.concatenate((player_relative, selected), axis=0)

    def getActionSpec(self, action_spec):
        # all_actions = {'function': dict(type='int', num_actions=len(action_spec.functions))}
        # version with limited actions to speed up learning of simple minigame
        all_actions = {'function': dict(type='int', num_actions=21)}
        for argument_type in action_spec.types:
            # the only arguments with a shape that isn't (1) are the screen and minimap ones,
            # so we're assuming the screen/minimap dimensions are square here
            spec = dict(type='int', shape=(len(argument_type.sizes),), num_actions=argument_type.sizes[0])
            all_actions[argument_type.name] = spec
        return all_actions

    def getStateSpec(self, obs_spec):
        states = dict(type='float', shape=(6, 84, 84))
        return states

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)

        network_spec = [
            dict(
                size=32,
                type='conv2d',
                window=5,
                stride=1,
                padding='SAME',
                bias=True,
                activation='relu',
                l2_regularization=0.0,
                l1_regularization=0.0,
            ),
            dict(
                type='pool2d',
                pooling_type='max',
                window=2,
                stride=2,
                padding='SAME',
            ),
            dict(
                size=32,
                type='conv2d',
                window=3,
                stride=1,
                padding='SAME',
                bias=True,
                activation='relu',
                l2_regularization=0.0,
                l1_regularization=0.0,
            ),
            dict(
                type='pool2d',
                pooling_type='average',
                window=2,
                stride=2,
                padding='SAME',
            ),
            dict(type='flatten'),
            dict(type='dense', size=512),
            dict(type='dense', size=512)
        ]

        self.tfAgent = DQNAgent(
            states=self.getStateSpec(obs_spec),
            actions=self.getActionSpec(action_spec),
            network=network_spec,
            batching_capacity=10,
            saver=dict(
                directory='models/test3',
                seconds=6000
            ),
            summarizer=dict(
                directory='models/test3',
                seconds=30,
                labels=[
                    'configuration',
                    'losses',
                    'inputs',
                    'gradients_scalar'
                ]
            ),
            update_mode=dict(
                unit='timesteps',
                batch_size=8,
                frequency=8
            ),
            memory=dict(
                type='replay',
                include_next_states=True,
                capacity=100
            ),
            # memory=dict(
            #     type='prioritized_replay',
            #     include_next_states=True,
            #     buffer_size=50,
            #     capacity=100
            # ),
            optimizer=dict(
                type='adam',
                learning_rate=10e-3
            ),
            # actions_exploration=dict(
            #     type="epsilon_decay",
            #     initial_epsilon=1.0,
            #     final_epsilon=0.1,
            #     timesteps=100000
            # ),
            actions_exploration=dict(
                type="epsilon_anneal",
                initial_epsilon=1.0,
                final_epsilon=0.1,
                timesteps=1000000
            )
        )


    def getActionFunction(self, obs, action):
        id = action['function']

        m = 'Ep. ' + str(self.episodes) + '/Step ' + str(self.steps) + ' '

        if id not in obs.observation['available_actions']:
            # no_op
            # print(m + 'FAIL Invalid action: ' + FUNCTIONS[id].name)
            return actions.FunctionCall(0, []), False

        args = []
        for i in range(len(FUNCTIONS[id].args)):
            args.append(action[FUNCTIONS[id].args[i].name].tolist())
        print(m + 'Valid action: ' + FUNCTIONS[id].name)
        return actions.FunctionCall(id, args), True

    def step(self, obs):
        super().step(obs)
        state = self.preprocess_state(obs)

        if self.steps > 1:
            if self.validLastAction:
                self.tfAgent.observe(terminal=False, reward=obs.reward)
            else:
                # self.tfAgent.observe(terminal=False, reward=obs.reward - 10e-3)
                self.tfAgent.observe(terminal=False, reward=obs.reward)

        action = self.tfAgent.act(state)

        action_for_sc, self.validLastAction = self.getActionFunction(obs, action)

        return action_for_sc
