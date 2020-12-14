import tensorflow as tf
import time

from pysc2_rl.ac_network import ACNetwork
from common.ac_mem import ACMemory


class ACAgent:
    def __init__(self, sess, config, restore):
        self._steps = 0
        self._episodes = 0
        self._episode_score = 0
        self._average_episode_score = 0
        self._average_episode_win = 0
        # versions of the above for eval episodes only
        self._eval_episodes = 0
        self._eval_average_episode_score = 0
        self._eval_average_episode_win = 0
        self._last_state = None
        self._last_reward = None
        self._last_action = None
        self._config = config

        # used for timing steps of action selection
        self._times = dict(
            sample=0,
            train_batch=0
        )
        self._time_count = 0

        self._memory = ACMemory(self._config['batch_size'])

        self._network = ACNetwork(
            self._config
        )
        self._sess = sess
        if not self._config['inference_only']:
            self._writer = tf.summary.FileWriter(self._config['model_dir'], self._sess.graph)

            # start output files
            with open(self._config['model_dir'] + '/episode_summaries.dat', 'a+') as f:
                f.write('step episode reward avg_reward win avg_win\n')

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

                    print("Model restored at step: ", self._steps)
                else:
                    print("Model copied from:", config['copy_model_from'])

            except (ValueError, AttributeError):
                # if the directory exists but there's no checkpoints, just continue
                # usually because a test crashed immediately last time
                self._sess.run(self._network.var_init)
                print('Model restore failed')
        else:
            self._sess.run(self._network.var_init)

    def observe(self, terminal=False, reward=0, win=0, eval_episode=False):
        self._episode_score += reward
        self._last_reward = reward
        if terminal:
            if eval_episode:
                self._eval_average_episode_score = (self._eval_average_episode_score * self._eval_episodes + self._episode_score) / (self._eval_episodes + 1)
                self._eval_average_episode_win = (self._eval_average_episode_win * self._episodes + win) / (self._episodes + 1)
                summary = self._network.eval_episode_summary(
                    self._sess,
                    self._episode_score,
                    self._eval_average_episode_score,
                    win,
                    self._eval_average_episode_win
                )
                self._eval_episodes += 1
            else:
                self._average_episode_score = (self._average_episode_score * self._episodes + self._episode_score) / (self._episodes + 1)
                self._average_episode_win = (self._average_episode_win * self._episodes + win) / (self._episodes + 1)
                summary = self._network.episode_summary(
                    self._sess,
                    self._episode_score,
                    self._average_episode_score,
                    win,
                    self._average_episode_win
                )
                self._episodes += 1
            if not self._config['inference_only']:
                self._writer.add_summary(summary, self._steps)
                # write out some episode stats for easy plotting later
                with open(self._config['model_dir'] + '/episode_summaries.dat', 'a+') as f:
                    f.write(str(self._steps) + ' ' + str(self._episodes) + ' ' + str(self._episode_score) + ' ' + str(self._average_episode_score) + ' ' + str(win) + ' ' + str(self._average_episode_win) + '\n')

            self._episode_score = 0

        # don't store things in memory if only doing inference or not training on eval episodes
        if not self._config['inference_only'] and (not eval_episode or self._config['train_on_eval_episodes']):
            # at end of episode store memory sample with None for next state
            # set last_state to None so that on next act() we know it is beginning of episode
            if terminal:
                self._memory.add_sample(self._last_state, self._last_action, reward)
                self._last_state = None

    def act(self, state, available_actions, eval_episode=False):
        # get action by inference from network
        summary, action, state_value = self._network.predict_one(self._sess, state)

        # convert action and value to ints/floats
        state_value = state_value[0]
        for name in action:
            action[name] = action[name][0]

        if not self._config['inference_only']:
            self._writer.add_summary(summary, self._steps)

        # if we are doing an eval ep and not training on it then we don't advance the training steps
        # or do any upkeep related to training
        if not eval_episode or self._config['train_on_eval_episodes']:
            self._steps += 1

        # if only doing inference no need to store anything in memory, update network, etc.
        # also if doing an eval episode and not training on those no need to train or store in memory
        if not self._config['inference_only'] and (not eval_episode or self._config['train_on_eval_episodes']):
            # store a sample in experiences
            if self._last_state is not None:
                td_target = self._last_reward + state_value * self._config['discount']
                self._memory.add_sample(self._last_state, self._last_action, td_target)

            # do a batch of learning if batch size reached
            if self._memory.get_size() == self._config['batch_size']:
                self._train()

            # save checkpoint if needed
            if self._steps % self._config['model_checkpoint_frequency'] == 0 or self._steps % self._config['max_steps'] == 0:
                save_path = self._network.saver.save(
                    sess=self._sess,
                    save_path=self._config['model_dir'] + '/model.ckpt',
                    global_step=self._steps
                )
                print("Model saved in path: %s" % save_path)

            self._last_state = state
            self._last_action = action

        return action

    def _train(self):
        last_time = time.time()

        # states stored in memory as tuple (state, action, reward, next_state)
        # next_state=None if state is terminal
        states, actions, td_targets = self._memory.sample(self._config['batch_size'])

        self._times['sample'] += time.time() - last_time
        last_time = time.time()

        summaries = self._network.train_batch(self._sess, self._steps, states, actions, td_targets)
        if not self._config['inference_only']:
            for summary in summaries:
                self._writer.add_summary(summary, self._steps)

        self._times['train_batch'] += time.time() - last_time
        self._time_count += 1
        if self._config['debug_logging'] and self._time_count == self._config['log_frequency']:
            print("average ms for", self._config['log_frequency'], "replays:")
            for k, v in self._times.items():
                print("%25s:%.2f" % (k, v / (self._config['log_frequency'] / 1000)))
                self._times[k] = 0
            self._time_count = 0
