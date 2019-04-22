from smac.env import StarCraft2Env

import numpy as np
import tensorflow as tf
import json
import os
import shutil

from tf_dqn.smac_rl.dqn_multi_agent import DQNAgent


def main():
    # load configuration
    with open('config.json', 'r') as fp:
        config = json.load(fp=fp)

    model_dir = config['model_dir'] + config['map_name'] + "_" + config['run_id']

    # save a copy of the configuration files being used for a run in the run's folder (first time only)
    restore = True
    if not os.path.exists(model_dir):
        restore = False
        os.makedirs(model_dir)
        shutil.copy2('config.json', model_dir)

    env = StarCraft2Env(map_name=config['map_name'])
    env_info = env.get_env_info()
    # <class 'dict'>: {'state_shape': 168, 'obs_shape': 80, 'n_actions': 14, 'n_agents': 8, 'episode_limit': 120}

    with tf.Session() as sess:
        rl_agent = DQNAgent(restore, model_dir, sess, env_info)

        n_agents = env_info["n_agents"]

        n_episodes = 10000000

        for e in range(n_episodes):
            env.reset()
            terminated = False
            episode_reward = 0

            while not terminated:
                obs = env.get_obs()
                # centralized information that individual agents don't have access to
                # state = env.get_state()

                avail_actions = []
                obs_formatted = []
                for agent_id in range(n_agents):
                    avail_actions.append(env.get_avail_agent_actions(agent_id))
                    obs_formatted.append({'state': obs[agent_id]})

                actions = rl_agent.act(obs_formatted, avail_actions)
                formatted_actions = []
                for action in actions:
                    formatted_actions.append(action['action'])
                # print(actions)
                reward, terminated, _ = env.step(formatted_actions)
                rl_agent.observe(terminated, reward)
                episode_reward += reward

            print("Total reward in episode {} = {}".format(e, episode_reward))

        env.close()


if __name__ == "__main__":
    main()
