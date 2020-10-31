import logging
import torch

from train import train_dqn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("CUDA AVAILABLE:", torch.cuda.is_available())
    settings = {
        "batch_size": 64,
        "checkpoint_frequency": 2,  # number of episodes between each checkpoint save
        "device": device,
        "epsilon": 0.25,
        "gamma": 0.9,
        "logs_dir": "logs",
        "lr": 0.0001,
        "max_steps": 10000,
        "memory_size": 10000,
        "model_name": "dqn",
        "num_episodes": 300,
        "out_dir": "out",
    }
    train_dqn(settings)
