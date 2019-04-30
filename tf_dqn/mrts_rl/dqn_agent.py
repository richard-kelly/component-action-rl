import tensorflow as tf
import numpy as np
import math
import random
import time
import copy

from tf_dqn.mrts_rl.mrts_network import MRTSNetwork
from tf_dqn.common.latest_replay_mem import LatestReplayMemory
from tf_dqn.common.prioritized_replay_mem import PrioritizedReplayMemory


class DQNAgent:
    def __init__(self, sess, config, restore):
        self._steps = 0
        self._episodes = 0
        self._average_episode_score = 0
        self._episode_score = {}
        self._memory_start_size_reached = config['inference_only']
        self._last_state = {}
        self._last_reward = {}
        self._last_action = {}
        self._sample_action = None
        self._config = config

        self._times = dict(
            sample=0,
            train_batch=0
        )
        self._time_count = 0

        # initialize epsilon
        self._epsilon = self._config['initial_epsilon']
        if self._config['decay_type'] == "exponential":
            self._decay = math.exp(math.log(self._config['final_epsilon'] / self._config['initial_epsilon']) / self._config['decay_steps'])
        elif self._config['decay_type'] == "linear":
            self._decay = (self._config['initial_epsilon'] - self._config['final_epsilon']) / self._config['decay_steps']
        else:
            self._decay = 0.0

        if self._config['use_priority_experience_replay']:
            self._memory = PrioritizedReplayMemory(
                self._config['memory_size'],
                config['per_alpha'],
                config['per_starting_beta'],
                config['per_beta_anneal_steps']
            )
        else:
            self._memory = LatestReplayMemory(self._config['memory_size'])

        self._network = MRTSNetwork(
            self._config['double_DQN'],
            self._config['dueling_network'],
            self._config['learning_rate'],
            self._config['learning_rate_decay_method'],
            self._config['learning_rate_decay_steps'],
            self._config['learning_rate_decay_param'],
            self._config['discount'],
            self._config['model_checkpoint_max'],
            self._config['model_checkpoint_every_n_hours'],
            self._config['reg_type'],
            self._config['reg_scale'],
            self._config['env']
        )
        self._sess = sess
        self._writer = tf.summary.FileWriter(self._config['model_dir'], self._sess.graph)

        if restore:
            try:
                checkpoint = tf.train.get_checkpoint_state(self._config['model_dir'])
                self._network.saver.restore(self._sess, checkpoint.model_checkpoint_path)
                self._steps = int(checkpoint.model_checkpoint_path.split('-')[-1])
                # this makes sure tensorboard deletes any "future" events logged after the checkpoint
                self._writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=self._steps)

                # adjust epsilon for current step
                self._update_epsilon(min(self._config['decay_steps'], self._steps))
                print("Model restored at step: ", self._steps, ", epsilon: ", self._epsilon)

            except (ValueError, AttributeError):
                # if the directory exists but there's no checkpoints, just continue
                # usually because a test crashed immediately last time
                self._sess.run(self._network.var_init)
                self._network.update_target_q_net(self._sess)
        else:
            self._sess.run(self._network.var_init)
            self._network.update_target_q_net(self._sess)

    def reset(self, game_num):
        # new episode; just ensures that we don't store a transition across episodes when there is no terminal obs
        self._last_state[game_num] = None
        self._last_action[game_num] = None
        self._last_reward[game_num] = None
        self._episode_score[game_num] = 0

    def observe(self, game_num, terminal=False, reward=0):
        self._episode_score[game_num] += reward
        if terminal:
            self._average_episode_score = (self._average_episode_score * self._episodes + self._episode_score[game_num]) / (self._episodes + 1)
            epsilon = self._epsilon if self._memory_start_size_reached else 1.0
            summary = self._network.episode_summary(self._sess, self._episode_score[game_num], self._average_episode_score, epsilon)
            self._writer.add_summary(summary, self._steps)

        if not self._config['inference_only']:
            self._last_reward[game_num] = reward
            # at end of episode store memory sample with None for next state
            # set last_state to None so that on next act() we know it is beginning of episode
            if terminal:
                # Next state doesn't matter for a terminal experience, but when it's sampled later the validation
                # acts on the entire batch, and it's nice to have a valid state in there rather than all zeros.
                self._memory.add_sample(self._last_state[game_num], self._last_action[game_num], reward, self._last_state[game_num], True)
                self._episodes += 1

                self._episode_score.pop(game_num, None)
                self._last_reward.pop(game_num, None)
                self._last_action.pop(game_num, None)
                self._last_state.pop(game_num, None)

    def act(self, game_num, state, remember):
        action = self._choose_action(state)

        if remember:
            self._steps += 1

        if not self._config['inference_only'] and remember:
            if self._last_state[game_num] is not None:
                self._memory.add_sample(self._last_state[game_num], self._last_action[game_num], self._last_reward[game_num], state, False)

            # update target network parameters occasionally
            if self._steps % self._config['target_update_frequency'] == 0:
                print('updating target network')
                self._network.update_target_q_net(self._sess)

            # do a batch of learning every "update_frequency" steps
            if self._steps % self._config['update_frequency'] == 0 and self._memory_start_size_reached:
                # only start training once memory has reached minimum size
                self._replay()

            # save checkpoint if needed
            if self._steps % self._config['model_checkpoint_frequency'] == 0:
                save_path = self._network.saver.save(
                    sess=self._sess,
                    save_path=self._config['model_dir'] + '/model.ckpt',
                    global_step=self._steps
                )
                print("Model saved in path: %s" % save_path)

            if self._steps <= self._config['decay_steps'] + self._config['memory_burn_in'] and self._memory_start_size_reached:
                # only start changing epsilon once memory has reached minimum size
                self._update_epsilon()

            # make a copy of the state because we may alter it outside this scope, but before it is stored in replay mem
            self._last_state[game_num] = copy.deepcopy(state)
            self._last_action[game_num] = action

            self._memory_start_size_reached = self._memory.get_size() >= self._config['memory_burn_in']

        return action

    def _update_epsilon(self, steps=1):
        if self._config['decay_type'] == "exponential":
            self._epsilon = self._epsilon * (self._decay ** steps)
        elif self._config['decay_type'] == "linear":
            self._epsilon = self._epsilon - (self._decay * steps)

    def _choose_action(self, state):
        state_for_validation = {}
        for name in state:
            # adds a new dimension of length 1 at the beginning (the batch size)
            state_for_validation[name] = np.expand_dims(state[name], axis=0)

        epsilon = self._config['inference_only_epsilon'] if self._config['inference_only'] else self._epsilon

        if not self._memory_start_size_reached or random.random() < epsilon:
            if self._sample_action is None:
                # store one action to serve as action specification
                _, self._sample_action = self._network.predict_one(self._sess, state)
            random_q_vals = {}
            for name, q_values in self._sample_action.items():
                random_q_vals[name] = np.random.rand(1, q_values.shape[1])
            actions = self._network.choose_valid_action(state_for_validation, random_q_vals)
        else:
            summary, pred = self._network.predict_one(self._sess, state)
            self._writer.add_summary(summary, self._steps)
            actions = self._network.choose_valid_action(state_for_validation, pred)
        action = {}
        for name in actions:
            action[name] = actions[name][0]
        return action

    def _replay(self):
        last_time = time.time()

        # states stored in memory as tuple (state, action, reward, next_state)
        # next_state=None if state is terminal
        if self._config['use_priority_experience_replay']:
            states, actions, rewards, next_states, is_terminal, weights = self._memory.sample(self._config['batch_size'])
        else:
            states, actions, rewards, next_states, is_terminal = self._memory.sample(self._config['batch_size'])
            weights = None

        self._times['sample'] += time.time() - last_time
        last_time = time.time()

        summary, priorities = self._network.train_batch(self._sess, self._steps, states, actions, rewards, next_states, is_terminal, weights)
        self._writer.add_summary(summary, self._steps)

        if self._config['use_priority_experience_replay']:
            self._memory.update_priorities_of_last_sample(priorities)

        self._times['train_batch'] += time.time() - last_time
        self._time_count += 1
        if self._config['debug_logging'] and self._time_count == self._config['log_frequency']:
            print("average ms for", self._config['log_frequency'], "replays:")
            for k, v in self._times.items():
                print("%25s:%.2f" % (k, v / (self._config['log_frequency'] / 1000)))
                self._times[k] = 0
            self._time_count = 0
