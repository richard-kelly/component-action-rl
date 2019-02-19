import tensorflow as tf
import numpy as np
import math
import random
import json

from network import Network
from memory import Memory

with open('config.json', 'r') as fp:
    config = json.load(fp=fp)


class DQNAgent:
    def __init__(self, restore):
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
        self._network = Network(
            config['learning_rate'],
            config['model_checkpoint_max'],
            config['model_checkpoint_every_n_hours']
        )
        self._sess = tf.Session()
        self._writer = tf.summary.FileWriter(config['model_dir'], self._sess.graph)

        if restore:
            self._network.saver.restore(self._sess, config['model_dir'] + '/model.ckpt')
            print("Model restored.")
        else:
            self._sess.run(self._network.var_init)

    def observe(self, terminal=False, reward=0):
        self._last_reward = reward
        # at end of episode store memory sample with None for next state
        # set last_state to None so that on next act() we know it is beginning of episode
        if terminal:
            self._memory.add_sample(self._last_state, self._last_action, reward, None, True)
            self._last_state = None
            self._episodes += 1

    def act(self, state):
        if self._last_state is not None:
            self._memory.add_sample(self._last_state, self._last_action, self._last_reward, state, False)

        action = self._choose_action(state)

        self._steps += 1

        # do a batch of learning every "update_frequency" steps
        if self._steps % config['update_frequency'] == 0:
            self._replay()

        # save checkpoint if needed
        if self._steps % config['model_checkpoint_frequency'] == 0:
            save_path = self._network.saver.save(
                sess=self._sess,
                save_path=config['model_dir'] + '/model.ckpt',
                global_step=self._steps
            )
            print("Model saved in path: %s" % save_path)

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
                # function is an integer, but everything else is a 1D array of ints
                function=random.randint(0, 3),
                screen=np.array([random.randint(0, 83), random.randint(0, 83)]),
                screen2=np.array([random.randint(0, 83), random.randint(0, 83)]),
                select_point_act=np.array([random.randint(0, 3)]),
                select_add=np.array([random.randint(0, 1)]),
                queued=np.array([random.randint(0, 1)]),
            )
        else:
            # returns dict
            pred = self._network.predict_one(state, self._sess)
            return dict(
                function=np.argmax(pred['function']),
                screen=np.array([np.argmax(pred['screen_x']), np.argmax(pred['screen_y'])]),
                screen2=np.array([np.argmax(pred['screen2_x']), np.argmax(pred['screen2_y'])]),
                select_point_act=np.array([np.argmax(pred['select_point_act'])]),
                select_add=np.array([np.argmax(pred['select_add'])]),
                queued=np.array([np.argmax(pred['queued'])])
            )

    def _replay(self):
        # states stored in memory as tuple (state, action, reward, next_state)
        # next_state=None if state is terminal
        states, actions, rewards, next_states, is_terminal = self._memory.sample(config['batch_size'])

        # predict Q(s,a) given the batch of states
        q_s_a = self._network.predict_batch(states, self._sess)

        # TODO: this should be a different target network updated periodically
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_target = self._network.predict_batch(next_states, self._sess)

        for i in range(rewards.shape[0]):
            # update the q value for action
            # TODO: Store action the same way the network produces it, and convert to SC2 agent format later
            # TODO: This would remove lots/all?? of the SC2 stuff from the DQN implementation
            if is_terminal[i]:
                # terminal state
                q_s_a['function'][i, actions['function'][i]] = rewards[i]
                q_s_a['screen_x'][i, actions['screen'][i, 0]] = rewards[i]
                q_s_a['screen_y'][i, actions['screen'][i, 1]] = rewards[i]
                q_s_a['screen2_x'][i, actions['screen2'][i, 0]] = rewards[i]
                q_s_a['screen2_y'][i, actions['screen2'][i, 1]] = rewards[i]
                q_s_a['select_point_act'][i, actions['select_point_act'][i, 0]] = rewards[i]
                q_s_a['select_add'][i, actions['select_add'][i, 0]] = rewards[i]
                q_s_a['queued'][i, actions['queued'][i, 0]] = rewards[i]
                # for key in action.keys():
                #     q_s_a[key][i][action[key]] = reward
            else:
                q_s_a['function'][i, actions['function'][i, 0]] = rewards[i] + config['discount'] * np.amax(q_s_a_target['function'][i])
                q_s_a['screen_x'][i, actions['screen'][i, 0]] = rewards[i] + config['discount'] * np.amax(q_s_a_target['screen_x'][i])
                q_s_a['screen_y'][i, actions['screen'][i, 1]] = rewards[i] + config['discount'] * np.amax(q_s_a_target['screen_y'][i])
                q_s_a['screen2_x'][i, actions['screen2'][i, 0]] = rewards[i] + config['discount'] * np.amax(q_s_a_target['screen2_x'][i])
                q_s_a['screen2_y'][i, actions['screen2'][i, 1]] = rewards[i] + config['discount'] * np.amax(q_s_a_target['screen2_y'][i])
                q_s_a['select_point_act'][i, actions['select_point_act'][i, 0]] = rewards[i] + config['discount'] * np.amax(q_s_a_target['select_point_act'][i])
                q_s_a['select_add'][i, actions['select_add'][i, 0]] = rewards[i] + config['discount'] * np.amax(q_s_a_target['select_add'][i])
                q_s_a['queued'][i, actions['queued'][i, 0]] = rewards[i] + config['discount'] * np.amax(q_s_a_target['queued'][i])
                # for key in action.keys():
                #     q_s_a[key][i][action[key]] = reward + config['discount'] * np.amax(q_s_a_target[key][i])

        self._network.train_batch(self._sess, states, q_s_a)

    def __del__(self):
        self._sess.close()
