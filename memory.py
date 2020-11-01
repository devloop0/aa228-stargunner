# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from collections import namedtuple
import random
import torch


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def init_with_random(self, state_shape, num_actions):
        while len(self.memory) < self.capacity:
            # Store in CPU to avoid GPU memory issues
            state = torch.randn(*state_shape).to("cpu")
            next_state = torch.randn(*state_shape).to("cpu")
            action = torch.tensor([random.randint(0, num_actions - 1)]).to("cpu")
            reward = torch.randn(1).to("cpu")
            self.push(state, action, next_state, reward)

    def __len__(self):
        return len(self.memory)
