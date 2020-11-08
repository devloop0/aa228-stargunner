import gym
import time
from dqn import DQN
import torch
from utils.gym import get_env
import random
import sys
import math
from statistics import stdev
import numpy as np
from memory import ReplayBuffer

NUM_EPISODES = 1000
NUM_TIMESTEPS = 10000
EPSILON = 0.01
GAME = 'StarGunnerDeterministic-v4'
SEED = 6
CHANNELS = 4
FPS = math.inf

def print_policy_statistics(reward_per_episode, reward_per_timestep, timesteps_per_episode, stdev_reward_per_episode, episode_num=None):
    if episode_num:
        episode_str = 'Episode {}; '.format(episode_num)
    else:
        episode_str = ''
    print ('{}reward / episode = {}'.format(episode_str, reward_per_episode))
    print ('{}reward / timestep = {}'.format(episode_str, reward_per_timestep))
    print ('{}timesteps / episode = {}'.format(episode_str, timesteps_per_episode))
    print ('{}stdev(reward / episode) = {}'.format(episode_str, stdev_reward_per_episode))

def wrapped_stdev(l):
    if len(l) <= 1:
        return 0
    else:
        return stdev(l)

def simulate_policy(env, action_func, num_episodes=NUM_EPISODES, num_timesteps=NUM_TIMESTEPS):
    total_reward = 0
    total_timesteps = 0
    all_rewards_per_episode = []

    memory = ReplayBuffer(num_timesteps, CHANNELS)

    for i in range(num_episodes):
        state = env.reset()[..., np.newaxis]
        curr_reward = 0

        for t in range(num_timesteps):
            env.render()
            last_idx = memory.store_frame(state)
            action = action_func(memory, state)
            state, reward, done, _ = env.step(action)
            state = state[..., np.newaxis]
            memory.store_effect(last_idx, action, reward, done)

            curr_reward += reward
            total_timesteps += 1

            time.sleep(1 / FPS)
            if done:
                break
        total_reward += curr_reward

        curr_episode = i + 1
        reward_per_episode = total_reward / curr_episode
        reward_per_timestep = total_reward / total_timesteps
        timesteps_per_episode = total_timesteps / curr_episode
        all_rewards_per_episode.append(reward_per_episode)

        print_policy_statistics(reward_per_episode, reward_per_timestep, timesteps_per_episode, wrapped_stdev(all_rewards_per_episode),
                episode_num=curr_episode)

    return total_reward / num_episodes, total_reward / total_timesteps, total_timesteps / num_episodes, wrapped_stdev(all_rewards_per_episode)

if __name__ == '__main__':
    env = get_env(GAME, 6, monitor=False)

    if len(sys.argv) != 2:
        print ('Incorrect number of arguments: python3 runner.py <random|dqn>')
        exit(1)

    if sys.argv[1] == 'dqn':
        print ('Simulating DQN...')

        device = torch.device("cpu")
        settings = {"num_actions": env.action_space.n, "num_channels": CHANNELS}
        model = DQN(settings)
        model.load_state_dict(torch.load('./dqn.model', map_location=device)["model_state_dict"])
        model.eval()

        def action_func(memory, state):
            recent_observations = memory.encode_recent_observation()

            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                obs = torch.from_numpy(recent_observations).to(device).unsqueeze(0) / 255.0
                with torch.no_grad():
                    forward_res = model(obs)
                    action = forward_res.argmax(dim=1).item()
            return action

    elif sys.argv[1] == 'random':
        print ('Simulating random policy...')

        def action_func(memory, state):
            return env.action_space.sample()
    else:
        print('Unrecognized option: {}'.format(sys.argv[1]))
        exit(1)

    reward_per_episode, reward_per_timestep, timesteps_per_episode, stdev_reward_per_episode = simulate_policy(env, action_func)
    print ('Final statistics:')
    print_policy_statistics(reward_per_episode, reward_per_timestep, timesteps_per_episode, stdev_reward_per_episode)
