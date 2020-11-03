import logging
import torch

from train import train_dqn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        logging.info("!!! USING CUDA !!!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings = {
        "batch_size": 256,
        "checkpoint_frequency": 1,  # number of episodes between each checkpoint save
        "device": device,
        "eps_start": 1.0,
        "eps_end": 0.1,
        "eps_cliff": 1000000,
        "eps_decay": 500,
        "gamma": 0.99,
        "logs_dir": "logs",
        "lr": 0.00025,
        "max_steps": 10000,
        "memory_size": 200000,
        "model_name": "dqn",
        "num_episodes": 5,
        "out_dir": "out",
    }
    train_dqn(settings)
