import logging
import math
import random

import gym
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dqn import DQN
from memory import ReplayMemory, Transition
from utils import (
    process_state,
    settings_is_valid,
    save_model,
    save_model_checkpoint,
)


def train_dqn(settings):
    required_settings = [
        "batch_size",
        "checkpoint_frequency",
        "device",
        "eps_start",
        "eps_end",
        "eps_decay",
        "gamma",
        "logs_dir",
        "lr",
        "max_steps",
        "memory_size",
        "model_name",
        "num_episodes",
        "out_dir",
    ]
    if not settings_is_valid(settings, required_settings):
        raise Exception(f"Settings object {settings} missing some required settings.")

    batch_size = settings["batch_size"]
    checkpoint_frequency = settings["checkpoint_frequency"]
    device = settings["device"]
    eps_start = settings["eps_start"]
    eps_end = settings["eps_end"]
    eps_decay = settings["eps_decay"]
    gamma = settings["gamma"]
    logs_dir = settings["logs_dir"]
    lr = settings["lr"]
    max_steps = settings["max_steps"]
    memory_size = settings["memory_size"]
    model_name = settings["model_name"]
    num_episodes = settings["num_episodes"]
    out_dir = settings["out_dir"]

    # Initialize environment
    env = gym.make("VideoPinball-v0")

    # Initialize model
    num_actions = env.action_space.n
    settings["num_actions"] = num_actions
    model = DQN(settings).to(device)

    # Initialize memory
    memory = ReplayMemory(memory_size)

    # Initialize other model ingredients
    criterion = F.smooth_l1_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize tensorboard
    writer = SummaryWriter(logs_dir)

    # Loop over episodes
    model.train()
    steps_done = 0
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
            epsilon = eps_end + (eps_start - eps_end) * math.exp(
                -1 * steps_done / eps_decay
            )
            if random.random() < epsilon:
                predicted_action = torch.tensor([env.action_space.sample()]).to(device)
            else:
                predicted_action = torch.argmax(Q, dim=1)
            next_state, reward, done, info = env.step(predicted_action.item())
            # Note that next state could also be a difference
            next_state = process_state(next_state).to(device)
            reward = torch.tensor([reward]).to(device)

            # Save to memory
            memory.push(state, predicted_action, next_state, reward)

            # Move to next state
            state = next_state

            # Sample from memory
            if len(memory) < batch_size:
                continue
            batch = Transition(*zip(*memory.sample(batch_size)))

            # Mask terminal state (taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
            final_mask = torch.tensor(
                tuple(map(lambda s: s is None, batch.next_state)),
                device=device,
                dtype=torch.bool,
            )
            # print("FINAL_MASK", final_mask.shape)
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # print("STATE_BATCH SHAPE", state_batch.shape)
            # print("ACTION_BATCH SHAPE", action_batch.shape)
            # print("REWARD_BATCH SHAPE", reward_batch.shape)

            # Compute Q
            # Q_next = torch.zeros((batch_size, num_actions))
            # print("MODEL STATE BATCH SHAPE", model(state_batch).shape)
            Q_pred = model(state_batch)
            Q_actual = Q_pred.gather(1, action_batch.view(action_batch.shape[0], 1))
            Q_max = torch.max(Q_actual, dim=1)[0]
            # print("Q_MAX shape", Q_max.shape)
            target = reward_batch + gamma * Q_max * final_mask.to(Q_max.dtype)
            # print("TARGET SIZE", target.shape)

            # Calculate loss
            loss = criterion(torch.max(Q_pred, dim=1)[0], target)
            loss.backward()

            # Clamp gradient to avoid gradient explosion
            for param in model.parameters():
                param.grad.data.clamp_(-100, 100)
            optimizer.step()

            # Store stats
            loss_acc += loss.item()
            reward_acc += reward
            steps_done += 1

            # Exit if in terminal state
            if done:
                logging.debug(
                    f"Episode {episode} finished after {t} timesteps with reward {reward_acc}."
                )
                break

        logging.debug(f"Loss: {loss_acc / t}")

        # Save model checkpoint
        if (episode != 0) and (episode % checkpoint_frequency == 0):
            save_model_checkpoint(
                model,
                optimizer,
                episode,
                loss,
                f"{out_dir}/checkpoints/{model_name}_{episode}",
            )

        # Log to tensorboard
        writer.add_scalar("Total Reward", reward_acc, episode)
        writer.add_scalar("Average Reward", reward_acc / t, episode)
        writer.add_scalar("Timesteps", t, episode)
        writer.add_scalar("Average loss", loss_acc / t, episode)

    # Save model
    save_model(model, f"{out_dir}/{model_name}.model")

    env.close()
    return model
