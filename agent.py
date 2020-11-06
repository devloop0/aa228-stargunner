# Inspired by https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py
import random

# from itertools import count

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dqn import DQN
from memory import ReplayBuffer
from utils.misc import settings_is_valid, save_model, save_model_checkpoint
from utils.gym import get_env, get_wrapper_by_name


class DQNAgent:
    def __init__(self, settings):
        self.check_settings(settings)

        # Constants
        self.batch_size = settings["batch_size"]
        self.checkpoint_frequency = settings["checkpoint_frequency"]
        self.device = settings["device"]
        self.dtype = (
            torch.cuda.FloatTensor if self.device.type == "cuda" else torch.FloatTensor
        )
        self.env = get_env(settings["env"], 0)
        self.eps_cliff = settings["eps_cliff"]
        self.eps_start = settings["eps_start"]
        self.eps_end = settings["eps_end"]
        self.frame_history_len = settings["frame_history_len"]
        self.gamma = settings["gamma"]
        self.learning_freq = settings["learning_freq"]
        self.learning_start = settings["learning_start"]
        self.logs_dir = settings["logs_dir"]
        self.log_freq = settings["log_freq"]
        self.memory_size = settings["memory_size"]
        self.model_name = settings["model_name"]
        self.num_actions = self.env.action_space.n
        settings["num_actions"] = self.num_actions
        settings["num_channels"] = self.frame_history_len
        self.out_dir = settings["out_dir"]
        self.target_update_freq = settings["target_update_freq"]
        self.total_timesteps = settings["total_timesteps"]

        # Init models
        self.Q = DQN(settings).to(self.device)
        self.target_Q = DQN(settings).to(self.device)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.target_Q.eval()

        # Init model supporting objects
        self.memory = ReplayBuffer(self.memory_size, self.frame_history_len)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=settings["lr"])
        self.loss = F.smooth_l1_loss

        # Logging
        self.writer = SummaryWriter(self.logs_dir)

    def check_settings(self, settings):
        required_settings = [
            "batch_size",
            "checkpoint_frequency",
            "device",
            "env",
            "eps_start",
            "eps_end",
            "eps_cliff",
            "frame_history_len",
            "gamma",
            "learning_start",
            "log_freq",
            "logs_dir",
            "lr",
            "memory_size",
            "model_name",
            "out_dir",
            "target_update_freq",
            "total_timesteps",
        ]

        if not settings_is_valid(settings, required_settings):
            raise Exception(
                f"Settings object {settings} missing some required settings."
            )

    def _get_epsilon(self, steps_done):
        if steps_done < self.eps_cliff:
            epsilon = (
                -(self.eps_start - self.eps_end) / self.eps_cliff * steps_done
                + self.eps_start
            )
        else:
            epsilon = self.eps_end
        return epsilon

    def select_epsilon_greedy_action(self, state, steps_done):
        if random.random() < self._get_epsilon(steps_done):
            return torch.IntTensor([random.randrange(self.num_actions)])
        obs = torch.from_numpy(state).type(self.dtype).unsqueeze(0) / 255.0
        with torch.no_grad():
            return self.Q(obs).argmax(dim=1).cpu()  # returns action

    def should_stop(self):
        return (
            get_wrapper_by_name(self.env, "Monitor").get_total_steps() >= self.max_steps
        )

    def train(self):
        num_param_updates = 0
        loss_acc_since_last_log = 0.0
        param_updates_since_last_log = 0

        state = self.env.reset()[..., np.newaxis]
        for t in tqdm(range(self.total_timesteps)):
            last_idx = self.memory.store_frame(state)
            recent_observations = self.memory.encode_recent_observation()

            # Choose random action if learning hasn't started yet
            if t > self.learning_start:
                action = self.select_epsilon_greedy_action(
                    recent_observations, t
                ).item()
            else:
                action = random.randrange(self.num_actions)

            # Advance a step
            next_state, reward, done, _ = self.env.step(action)
            next_state = next_state[..., np.newaxis]

            # Store result in memory
            self.memory.store_effect(last_idx, action, reward, done)

            # Reset if done (life lost, due to atari wrapper)
            if done:
                next_state = self.env.reset()
                next_state = next_state[..., np.newaxis]
            state = next_state

            # Train network using experience replay when
            # memory is sufficiently large.
            if (
                t > self.learning_start
                and t % self.learning_freq == 0
                and self.memory.can_sample(self.batch_size)
            ):
                # Sample from replay buffer
                (
                    state_batch,
                    act_batch,
                    r_batch,
                    next_state_batch,
                    done_mask,
                ) = self.memory.sample(self.batch_size)
                state_batch = torch.from_numpy(state_batch).type(self.dtype) / 255.0
                act_batch = torch.from_numpy(act_batch).long().to(self.device)
                r_batch = torch.from_numpy(r_batch).to(self.device)
                next_state_batch = (
                    torch.from_numpy(next_state_batch).type(self.dtype) / 255.0
                )
                not_done_mask = 1 - torch.from_numpy(done_mask).type(self.dtype)

                # Calculate current Q value
                current_Q_vals = self.Q(state_batch).gather(1, act_batch.unsqueeze(1))

                # Calculate next Q value based on action that gives max Q vals
                next_max_Q = self.target_Q(next_state_batch).detach().max(dim=1)[0]
                next_Q_vals = not_done_mask * next_max_Q

                # Calculate target of current Q values
                target_Q_vals = r_batch + (self.gamma * next_Q_vals)

                # Calculate loss and backprop
                loss = F.smooth_l1_loss(current_Q_vals.squeeze(), target_Q_vals)
                self.optimizer.zero_grad()
                loss.backward()

                # Update weights
                self.optimizer.step()
                num_param_updates += 1

                # Store stats
                loss_acc_since_last_log += loss.item()
                param_updates_since_last_log += 1

                # Update target network periodically
                if num_param_updates % self.target_update_freq == 0:
                    self.target_Q.load_state_dict(self.Q.state_dict())

                # Save model checkpoint
                if num_param_updates % self.checkpoint_frequency == 0:
                    save_model_checkpoint(
                        self.Q,
                        self.optimizer,
                        t,
                        f"{self.out_dir}/checkpoints/{self.model_name}_{num_param_updates}",
                    )

            # Log progress, potentially at end of episode
            if done:
                wrapper = get_wrapper_by_name(self.env, "Monitor")
                episode_rewards = wrapper.get_episode_rewards()
                if (
                    param_updates_since_last_log > 0
                    and len(episode_rewards) % self.log_freq == 0
                ):
                    mean_reward = round(
                        np.mean(episode_rewards[-self.log_freq - 1 : -1]), 2
                    )
                    sum_reward = np.sum(episode_rewards[-self.log_freq - 1 : -1])
                    episode_lengths = wrapper.get_episode_lengths()
                    mean_duration = round(
                        np.mean(episode_lengths[-self.log_freq - 1 : -1]), 2
                    )
                    sum_duration = np.sum(episode_lengths[-self.log_freq - 1 : -1])

                    self.writer.add_scalar(
                        f"Mean Reward ({self.log_freq} episodes)",
                        mean_reward,
                        len(episode_rewards),
                    )
                    self.writer.add_scalar(
                        f"Mean Duration ({self.log_freq} episodes)",
                        mean_duration,
                        len(episode_rewards),
                    )
                    self.writer.add_scalar(
                        f"Mean Reward per Timestep ({self.log_freq} episodes)",
                        round(sum_reward / sum_duration, 2),
                        len(episode_rewards),
                    )
                    self.writer.add_scalar(
                        f"Mean Loss ({self.log_freq} episodes)",
                        loss_acc_since_last_log / param_updates_since_last_log,
                        len(episode_rewards),
                    )

                    loss_acc_since_last_log = 0.0
                    param_updates_since_last_log = 0

        # Save model
        save_model(self.Q, f"{self.out_dir}/{self.model_name}.model")

        self.env.close()
        return self.Q
