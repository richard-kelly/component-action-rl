import tensorflow as tf
import numpy as np
import math
import random
import json
import time

from network import Network
from memory import Memory

with open('config.json', 'r') as fp:
    config = json.load(fp=fp)


class DQNAgent:
    def __init__(self, restore):
        self._steps = 0
        self._episodes = 0
        self._episode_score = 0
        self._memory_start_size_reached = False
        self._last_state = None
        self._last_reward = None
        self._last_action = None
        self._sample_action = None

        self._times = dict(
            sample=0,
            train_batch=0
        )
        self._time_count = 0

        # initialize epsilon
        if not config['run_model_no_training']:
            self._epsilon = config['initial_epsilon']
        else:
            self._epsilon = 0.0
        if config['decay_type'] == "exponential":
            self._decay = math.exp(math.log(config['final_epsilon'] / config['initial_epsilon'])/config['decay_steps'])
        elif config['decay_type'] == "linear":
            self._decay = (config['initial_epsilon'] - config['final_epsilon']) / config['decay_steps']
        else:
            self._decay = 0.0

        self._memory = Memory(config['memory_size'])
        self._network = Network(
            config['learning_rate'],
            config['discount'],
            config['model_checkpoint_max'],
            config['model_checkpoint_every_n_hours'],
            config['screen_size']
        )
        self._sess = tf.Session()
        self._writer = tf.summary.FileWriter(config['model_dir'], self._sess.graph)

        if restore:
            try:
                checkpoint = tf.train.get_checkpoint_state(config['model_dir'])
                self._network.saver.restore(self._sess, checkpoint.model_checkpoint_path)
                self._steps = int(checkpoint.model_checkpoint_path.split('-')[-1])
                # this makes sure tensorboard deletes any "future" events logged after the checkpoint
                self._writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=self._steps)

                # adjust epsilon for current step
                self._update_epsilon(min(config['decay_steps'], self._steps))
                print("Model restored at step: ", self._steps, ", epsilon: ", self._epsilon)

            except (ValueError, AttributeError):
                # if the directory exists but there's no checkpoints, just continue
                # usually because a test crashed immediately last time
                self._sess.run(self._network.var_init)
                self._network.update_target_q_net(self._sess)
        else:
            self._sess.run(self._network.var_init)
            self._network.update_target_q_net(self._sess)

    def observe(self, terminal=False, reward=0):
        self._episode_score += reward
        if terminal:
            epsilon = self._epsilon if self._memory_start_size_reached else 1.0
            summary = self._network.episode_summary(self._sess, self._episode_score, epsilon)
            self._writer.add_summary(summary, self._steps)
            self._episode_score = 0

        if not config['run_model_no_training']:
            self._last_reward = reward
            # at end of episode store memory sample with None for next state
            # set last_state to None so that on next act() we know it is beginning of episode
            if terminal:
                self._memory.add_sample(self._last_state, self._last_action, reward, None, True)
                self._last_state = None
                self._episodes += 1

    def act(self, state):
        action = self._choose_action(state)
        self._steps += 1

        if not config['run_model_no_training']:
            if self._last_state is not None:
                self._memory.add_sample(self._last_state, self._last_action, self._last_reward, state, False)

            # update target network parameters occasionally
            if self._steps % config['target_update_frequency'] == 0:
                print('updating target network')
                self._network.update_target_q_net(self._sess)

            # do a batch of learning every "update_frequency" steps
            if self._steps % config['update_frequency'] == 0 and self._memory_start_size_reached:
                # only start training once memory has reached minimum size
                self._replay()

            # save checkpoint if needed
            if self._steps % config['model_checkpoint_frequency'] == 0:
                save_path = self._network.saver.save(
                    sess=self._sess,
                    save_path=config['model_dir'] + '/model.ckpt',
                    global_step=self._steps
                )
                print("Model saved in path: %s" % save_path)

            if self._steps <= config['decay_steps'] + config['memory_burn_in'] and self._memory_start_size_reached:
                # only start changing epsilon once memory has reached minimum size
                self._update_epsilon()

            self._last_state = state
            self._last_action = action

            self._memory_start_size_reached = self._memory.get_size() >= config['memory_burn_in']

        return action

    def _update_epsilon(self, steps=1):
        if config['decay_type'] == "exponential":
            self._epsilon = self._epsilon * (self._decay ** steps)
        elif config['decay_type'] == "linear":
            self._epsilon = self._epsilon - (self._decay * steps)

    def _choose_action(self, state):
        action = {}
        if not self._memory_start_size_reached or random.random() < self._epsilon:
            if self._sample_action is None:
                # store one action to serve as action specification
                _, self._sample_action = self._network.predict_one(self._sess, state)
            for name, logits in self._sample_action.items():
                action[name] = random.randint(0, logits.shape[1] - 1)
        else:
            summary, pred = self._network.predict_one(self._sess, state)
            self._writer.add_summary(summary, self._steps)
            for name, q_values in pred.items():
                action[name] = np.argmax(q_values)
        return action

    def _replay(self):
        last_time = time.time()

        # states stored in memory as tuple (state, action, reward, next_state)
        # next_state=None if state is terminal
        states, actions, rewards, next_states, is_terminal = self._memory.sample(config['batch_size'])

        self._times['sample'] += time.time() - last_time
        last_time = time.time()

        summary = self._network.train_batch(self._sess, states, actions, rewards, next_states, is_terminal * -1)
        self._writer.add_summary(summary, self._steps)

        self._times['train_batch'] += time.time() - last_time
        self._time_count += 1
        if config['debug_logging'] and self._time_count == config['log_frequency']:
            print("average ms for", config['log_frequency'], "replays:")
            for k, v in self._times.items():
                print("%25s:%.2f" % (k, v / (config['log_frequency'] / 1000)))
                self._times[k] = 0
            self._time_count = 0

    def __del__(self):
        self._sess.close()
