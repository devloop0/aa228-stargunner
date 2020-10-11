import gym

env = gym.make('VideoPinball-v0')
print(env.action_space)
print(env.observation_space)

for episode_num in range(20):
    observation = env.reset()
    episode_reward = 0
    for t in range(10000):
        env.render()

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            print('Episode finished after {} timesteps'.format(t + 1))
            break

env.close()
