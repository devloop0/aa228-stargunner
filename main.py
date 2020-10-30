import logging
import torch

from train import train_dqn


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = {
        "device": device,
        "epsilon": 0.05,
        "gamma": 0.1,
        "lr": 0.0001,
        "max_steps": 10000,
        "num_episodes": 300,
    }
    train_dqn(settings)
