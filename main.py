import logging
import torch

from train import train_dqn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        logging.info("!!! USING CUDA !!!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    settings = {
        "batch_size": 32,
        "checkpoint_frequency": 1,  # number of episodes between each checkpoint save
        "device": device,
        "eps_start": 1.0,
        "eps_end": 0.1,
        "eps_cliff": 1000000,
        "eps_decay": 500,
        "gamma": 0.99,
        "logs_dir": "logs",
        "log_freq": 5,
        "lr": 0.00025,
        "max_steps": 10000,
        "memory_size": 20000,
        "model_name": "dqn",
        "num_episodes": 20,
        "out_dir": "out",
        "target_net_update_freq": 2,
    }
    train_dqn(settings)
