import logging
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


def clamp_reward(reward):
    if reward > 0:
        return 1.0
    elif reward < 0:
        return -1.0
    else:
        return 0.0


def train_dqn(settings):
    required_settings = [
        "batch_size",
        "checkpoint_frequency",
        "device",
        "eps_start",
        "eps_end",
        "eps_cliff",
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
    eps_cliff = settings["eps_cliff"]
    # eps_decay = settings["eps_decay"]
    gamma = settings["gamma"]
    logs_dir = settings["logs_dir"]
    lr = settings["lr"]
    max_steps = settings["max_steps"]
    memory_size = settings["memory_size"]
    model_name = settings["model_name"]
    num_episodes = settings["num_episodes"]
    out_dir = settings["out_dir"]

    # Initialize environment
    env = gym.make("StarGunner-v0")

    # Initialize model
    num_actions = env.action_space.n
    settings["num_actions"] = num_actions
    model = DQN(settings).to(device)

    # Initialize memory
    logging.info("Initializing memory.")
    memory = ReplayMemory(memory_size)
    memory.init_with_random((1, 3, 84, 84), num_actions)
    logging.info("Finished initializing memory.")

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
                Q = model.forward(state.type(torch.float))

            # Get best predicted action and perform it
            if steps_done < eps_cliff:
                epsilon = -(eps_start - eps_end) / eps_cliff * steps_done + eps_start
            else:
                epsilon = eps_end

            if random.random() < epsilon:
                predicted_action = torch.tensor([env.action_space.sample()]).to(device)
            else:
                predicted_action = torch.argmax(Q, dim=1)
            next_state, raw_reward, done, info = env.step(predicted_action.item())
            # Note that next state could also be a difference
            next_state = process_state(next_state)
            reward = torch.tensor([clamp_reward(raw_reward)])

            # Save to memory
            memory.push(state.to("cpu"), predicted_action.to("cpu"), next_state, reward)

            # Move to next state
            state = next_state.to(device)

            # Sample from memory
            batch = Transition(*zip(*memory.sample(batch_size)))

            # Mask terminal state (adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
            final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=device,
                dtype=torch.bool,
            )
            # print("FINAL_MASK", final_mask.shape)
            state_batch = torch.cat(batch.state).type(torch.float).to(device)
            next_state_batch = torch.cat(batch.next_state).type(torch.float).to(device)
            action_batch = torch.cat(batch.action).to(device)
            reward_batch = torch.cat(batch.reward).to(device)

            # print("STATE_BATCH SHAPE", state_batch.shape)
            # print("STATE_BATCH", state_batch[4, :, 100])
            # print("ACTION_BATCH SHAPE", action_batch.shape)
            # print("ACTION_BATCH", action_batch)
            # print("REWARD_BATCH SHAPE", reward_batch.shape)

            # Compute Q
            # Q_next = torch.zeros((batch_size, num_actions))
            # print("MODEL STATE BATCH SHAPE", model(state_batch).shape)
            Q_next_pred = model(next_state_batch)
            Q_actual = model(state_batch).gather(
                1, action_batch.view(action_batch.shape[0], 1)
            )
            Q_max = torch.max(Q_next_pred, dim=1)[0]
            # print("Q_MAX shape", Q_max.shape)
            target = reward_batch + gamma * Q_max * final_mask.to(Q_max.dtype)
            # print("TARGET SIZE", target.shape)

            # Calculate loss
            loss = criterion(Q_actual.squeeze(), target)
            loss.backward()

            # Clamp gradient to avoid gradient explosion
            for param in model.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            # Store stats
            loss_acc += loss.item()
            reward_acc += raw_reward
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
        writer.add_scalar("Steps", reward_acc / t, steps_done)

    # Save model
    save_model(model, f"{out_dir}/{model_name}.model")

    # Report final stats
    logging.info(f"Steps Done: {steps_done}")

    env.close()
    return model
