import logging
import random

import gym
import torch
from torch import optim
from tqdm import tqdm

from dqn import DQN
from utils import play_using_model, process_state, settings_is_valid


def train_dqn(settings):
    required_settings = [
        "device",
        "epsilon",
        "gamma",
        "lr",
        "max_steps",
        "num_episodes",
    ]
    if not settings_is_valid(settings, required_settings):
        raise Exception(f"Settings object {settings} missing some required settings.")

    device = settings["device"]
    epsilon = settings["epsilon"]
    gamma = settings["gamma"]
    lr = settings["lr"]
    max_steps = settings["max_steps"]
    num_episodes = settings["num_episodes"]

    # Initialize environment
    env = gym.make("VideoPinball-v0")

    # Initialize model
    settings["num_actions"] = env.action_space.n
    model = DQN(settings).to(device)

    # Initialize other model ingredients
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loop over episodes
    for episode in tqdm(range(num_episodes)):
        state = process_state(env.reset()).to(device)
        reward_acc = 0.0
        loss_acc = 0.0

        # Loop over steps in episode
        for t in range(max_steps):
            optimizer.zero_grad()
            with torch.no_grad():
                Q = model.forward(state)

            # Get best predicted action and perform it
            if random.random() < epsilon:
                predicted_action = env.action_space.sample()
            else:
                predicted_action = torch.argmax(Q, dim=1).item()
            state, reward, done, info = env.step(predicted_action)
            state = process_state(state).to(device)

            # Get next Q and use it to optimize
            Q_next = model.forward(state)
            target = reward + gamma * torch.max(Q_next, dim=1)[0]
            loss = criterion(Q, target)
            loss.backward()
            optimizer.step()

            # Store stats
            loss_acc += loss.item()
            reward_acc += reward

            # Exit if in terminal state
            if done:
                logging.debug(
                    f"Episode {episode} finished after {t} timesteps with reward {reward_acc}."
                )
                break

        logging.debug(f"Loss: {loss_acc / max_steps}")

    play_using_model(env, model, device)

    env.close()
    return model
