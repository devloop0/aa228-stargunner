import logging
import random
import time

import gym
import torch
from tqdm import tqdm

from dqn import DQN
from utils import process_state


def load_model(model_filename, num_actions):
    settings = {"num_actions": num_actions}
    model = DQN(settings)
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    return model


def load_model_checkpoint(checkpoint_filename, num_actions):
    settings = {"num_actions": num_actions}
    model = DQN(settings)
    model.load_state_dict(torch.load(checkpoint_filename)["model_state_dict"])
    model.eval()
    return model


def play_using_model(env, model, device, max_steps=10000, epsilon=0.01):
    model.eval()
    reward_acc = 0.0
    with torch.no_grad():
        state = process_state(env.reset()).to(device)
        for step in tqdm(range(max_steps)):
            env.render()
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                forward_res = model.forward(state)
                action = torch.argmax(forward_res, dim=1).item()
            state, reward, done, info = env.step(action)
            state = process_state(state).to(device)
            reward_acc += reward

            if done:
                break

            time.sleep(0.05)

    logging.info("Total Reward:", reward_acc)
    logging.info("Average Reward per Timestep:", reward_acc / step)
    logging.info("Timesteps:", step)


if __name__ == "__main__":
    # Initialize environment
    env = gym.make("VideoPinball-v0")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = env.action_space.n
    model = load_model_checkpoint("out/checkpoints/dqn_25", num_actions).to(device)

    # play using model
    play_using_model(env, model, device)
