import tensorflow as tf
import numpy as np
import gym
import math
import random
import json
import matplotlib.pyplot as plt

from network import Network
from memory import Memory

with open('config.json', 'r') as fp:
    config = json.load(fp=fp)


class DQNAgent:
    def __init__(self):
        self._steps = 0
        self._episodes = 0
        self._last_state = None
        self._last_reward = None
        self._last_action = None

        # initialize epsilon
        self._epsilon = config['initial_epsilon']
        if config['decay_type'] is "exponential":
            self._decay = math.exp(math.log(config['final_epsilon'] / config['initial_epsilon'])/config['decay_steps'])
        elif config['decay_type'] is "linear":
            self._decay = (config['initial_epsilon'] - config['final_epsilon']) / config['decay_steps']

        self._memory = Memory(config['memory_size'])
        self._network = Network(config['batch_size'])
        self._sess = tf.Session()
        self._sess.run(self._network.var_init)

    def observe(self, terminal=False, reward=0):
        self._last_reward = reward
        # at end of episode store memory sample with None for next state
        # set last_state to None so that on next act() we know it is beginning of episode
        if terminal:
            self._memory.add_sample((self._last_state, self._last_action, reward, None))
            self._last_state = None
            self._episodes += 1

    def act(self, state):
        if self._last_state is not None:
            self._memory.add_sample((self._last_state, self._last_action, self._last_reward, state))

        action = self._choose_action(state)

        # do a batch of learning every "update_frequency" steps
        self._steps += 1
        if self._steps % config['update_frequency'] == 0:
            self._replay()

        self._update_epsilon()

        self._last_state = state
        self._last_action = action

        return action

    def _update_epsilon(self):
        if config['decay_type'] is "exponential":
            self._epsilon = self._epsilon * self._decay
        elif config['decay_type'] is "linear":
            self._epsilon = self._epsilon - self._decay

    def _choose_action(self, state):
        if random.random() < self._epsilon:
            return dict(
                function=random.randint(0, 3),
                screen=np.array(random.randint(0, 83), random.randint(0, 83)),
                screen2=np.array(random.randint(0, 83), random.randint(0, 83)),
                select_point_act=np.array(random.randint(0, 3)),
                select_add=np.array(random.randint(0, 1)),
                queued=np.array(random.randint(0, 1)),
            )
        else:
            # returns dict
            pred = self._network.predict_one(state, self._sess)
            return dict(
                function=np.argmax(pred['function']),
                screen=np.array(np.argmax(pred['screen_x']), np.argmax(pred['screen_y'])),
                screen2=np.array(np.argmax(pred['screen2_x']), np.argmax(pred['screen2_y'])),
                select_point_act=np.array(np.argmax(pred['select_point_act'])),
                select_add=np.array(np.argmax(pred['select_add'])),
                queued=np.array(np.argmax(pred['queued']))
            )

    def _replay(self):
        # states stored in memory as tuple (state, action, reward, next_state)
        # next_state=None if state is terminal
        batch = self._memory.sample(config['batch_size'])

        # turn states into dict with arrays of size (batch_size, ....) for each part of state
        states = np.array([val[0] for val in batch])


        next_states = np.array([(np.zeros(self._network._num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._network.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._network.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._network._num_states))
        y = np.zeros((len(batch), self._network._num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._network.train_batch(self._sess, x, y)

    def __del__(self):
        self._sess.close()
