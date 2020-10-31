import logging
import torch

from train import train_dqn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        logging.info("!!! USING CUDA !!!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings = {
        "batch_size": 128,
        "checkpoint_frequency": 25,  # number of episodes between each checkpoint save
        "device": device,
        "eps_start": 0.95,
        "eps_end": 0.05,
        "eps_decay": 500,
        "gamma": 0.95,
        "logs_dir": "logs",
        "lr": 0.0002,
        "max_steps": 10000,
        "memory_size": 10000,
        "model_name": "dqn",
        "num_episodes": 400,
        "out_dir": "out",
    }
    train_dqn(settings)
