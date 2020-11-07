import logging
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from dqn import DQN
from memory import ReplayBuffer
from utils.gym import get_env


logging.basicConfig(level=logging.INFO)


def load_model(model_filename, num_actions, num_channels):
    settings = {"num_actions": num_actions, "num_channels": num_channels}
    model = DQN(settings)
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    return model


def load_model_checkpoint(checkpoint_filename, num_actions, num_channels):
    settings = {"num_actions": num_actions, "num_channels": num_channels}
    model = DQN(settings)
    model.load_state_dict(torch.load(checkpoint_filename)["model_state_dict"])
    model.eval()
    return model


def play_using_model(env, model, device, max_steps=10000, epsilon=0.05):
    model.eval()
    reward_acc = 0.0
    memory = ReplayBuffer(max_steps, 4)
    state = env.reset()[..., np.newaxis]
    for _step in tqdm(range(max_steps)):
        env.render()
        last_idx = memory.store_frame(state)
        recent_observations = memory.encode_recent_observation()
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            obs = torch.from_numpy(recent_observations).to(device).unsqueeze(0) / 255.0
            with torch.no_grad():
                forward_res = model(obs)
                action = forward_res.argmax(dim=1).item()

        state, reward, done, _ = env.step(action)
        state = state[..., np.newaxis]
        memory.store_effect(last_idx, action, reward, done)

        reward_acc += reward

        if done:
            break

        time.sleep(0.05)

    logging.info(f"Total Reward: {reward_acc}")
    logging.info(f"Average Reward per Timestep: {reward_acc / _step}")
    logging.info(f"Timesteps: {_step}")


if __name__ == "__main__":
    # Initialize environment
    env = get_env("StarGunnerDeterministic-v4", 6, monitor=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = env.action_space.n
    num_channels = 4
    model = load_model_checkpoint(
        "out/checkpoints/dqn_1250000", num_actions, num_channels
    ).to(device)

    # play using model
    play_using_model(env, model, device)
