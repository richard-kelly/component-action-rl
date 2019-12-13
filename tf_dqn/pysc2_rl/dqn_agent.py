import tensorflow as tf
import numpy as np
import math
import random
import time

from tf_dqn.pysc2_rl.pysc2_network import SC2Network
from tf_dqn.common.latest_replay_mem import LatestReplayMemory
from tf_dqn.common.prioritized_replay_mem import PrioritizedReplayMemory


class DQNAgent:
    def __init__(self, sess, config, restore):
        self._steps = 0
        self._episodes = 0
        self._episode_score = 0
        self._average_episode_score = 0
        self._average_episode_win = 0
        # if doing inference only, we don't need to populate the experience memory
        self._memory_start_size_reached = config['inference_only']
        self._last_state = []
        self._last_reward = []
        self._last_action = []
        self._sample_action = None
        self._config = config

        # used for timing steps of action selection
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

        self._network = SC2Network(
            self._config
        )
        self._sess = sess
        self._writer = tf.summary.FileWriter(self._config['model_dir'], self._sess.graph)

        if restore:
            try:
                if config['copy_model_from'] == "":
                    checkpoint = tf.train.get_checkpoint_state(self._config['model_dir'])
                else:
                    checkpoint = tf.train.get_checkpoint_state(config['copy_model_from'])
                self._network.saver.restore(self._sess, checkpoint.model_checkpoint_path)
                if config['copy_model_from'] == "":
                    self._steps = int(checkpoint.model_checkpoint_path.split('-')[-1])
                    # this makes sure tensorboard deletes any "future" events logged after the checkpoint
                    if not self._config['inference_only']:
                        self._writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=self._steps)

                    # adjust epsilon for current step
                    self._update_epsilon(min(self._config['decay_steps'], self._steps))
                    print("Model restored at step: ", self._steps, ", epsilon: ", self._epsilon)
                else:
                    print("Model copied from:", config['copy_model_from'])

            except (ValueError, AttributeError):
                # if the directory exists but there's no checkpoints, just continue
                # usually because a test crashed immediately last time
                self._sess.run(self._network.var_init)
                self._network.update_target_q_net(self._sess)
                print('Model restore failed')
        else:
            self._sess.run(self._network.var_init)
            self._network.update_target_q_net(self._sess)

    def observe(self, terminal=False, reward=0, win=0):
        self._episode_score += reward
        if terminal:
            self._average_episode_score = (self._average_episode_score * self._episodes + self._episode_score) / (self._episodes + 1)
            self._average_episode_win = (self._average_episode_win * self._episodes + win) / (self._episodes + 1)
            epsilon = self._epsilon if self._memory_start_size_reached else 1.0
            epsilon = epsilon if not self._config['inference_only'] else self._config['inference_only_epsilon']
            summary = self._network.episode_summary(
                self._sess,
                self._episode_score,
                self._average_episode_score,
                win,
                self._average_episode_win,
                epsilon
            )
            if not self._config['inference_only']:
                self._writer.add_summary(summary, self._steps)
            self._episode_score = 0

        # don't store things in memory if only doing inference
        if not self._config['inference_only']:
            for i in range(len(self._last_reward)):
                self._last_reward[i-1] = self._last_reward[i-1] + reward * self._config['discount'] ** (i + 1)
            self._last_reward.append(reward)
            # at end of episode store memory sample with None for next state
            # set last_state to None so that on next act() we know it is beginning of episode
            if terminal:
                for i in range(len(self._last_state)):
                    self._memory.add_sample(self._last_state.pop(0), self._last_action.pop(0), self._last_reward.pop(0), None, True)
                self._episodes += 1

    def act(self, state, available_actions):
        action = self._choose_action(state, available_actions)
        self._steps += 1

        # if only doing inference no need to store anything in memory, update network, etc.
        if not self._config['inference_only']:
            if len(self._last_state) >= self._config['bootstrapping_steps']:
                self._memory.add_sample(self._last_state.pop(0), self._last_action.pop(0), self._last_reward.pop(0), state, False)

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

            self._last_state.append(state)
            self._last_action.append(action)

            self._memory_start_size_reached = self._memory.get_size() >= self._config['memory_burn_in']

        return action

    def _update_epsilon(self, steps=1):
        if self._config['decay_type'] == "exponential":
            self._epsilon = self._epsilon * (self._decay ** steps)
        elif self._config['decay_type'] == "linear":
            self._epsilon = self._epsilon - (self._decay * steps)

    def _choose_action(self, state, available_actions):
        action = {}

        # use epsilon set in config if doing inference only, otherwise use calculated current epsilon
        epsilon = self._config['inference_only_epsilon'] if self._config['inference_only'] else self._epsilon
        if not self._memory_start_size_reached or random.random() < epsilon:
            # take a random action
            if self._sample_action is None:
                # store one action to serve as action specification
                _, self._sample_action = self._network.predict_one(self._sess, state)
            for name, q_values in self._sample_action.items():
                if name == 'function':
                    valid = np.in1d(self._config['env']['computed_action_list'], available_actions)
                    try:
                        options = np.nonzero(valid)[0]
                        action[name] = np.random.choice(options)
                    except Exception:
                        print("WARNING: There were no valid actions. SOMETHING WENT WRONG.")
                        action[name] = available_actions[0]
                else:
                    action[name] = random.randint(0, q_values.shape[1] - 1)
        else:
            # get action by inference from network
            summary, pred = self._network.predict_one(self._sess, state)
            if not self._config['inference_only']:
                self._writer.add_summary(summary, self._steps)
            for name, q_values in pred.items():
                if name == 'function':
                    q_values = q_values.flatten()
                    valid = np.in1d(self._config['env']['computed_action_list'], available_actions)
                    indices = np.nonzero(np.logical_not(valid))
                    q_values[indices] = np.nan
                action[name] = np.nanargmax(q_values)
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
        if not self._config['inference_only']:
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
