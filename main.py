import logging
import torch

from agent import DQNAgent


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        logging.info("!!! USING CUDA !!!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # See explanation for some of the settings here:
    # https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py
    settings = {
        "batch_size": 32,
        "checkpoint_frequency": 100000,  # number of parameter updates between each checkpoint save
        "device": device,
        "env": "StarGunner-v0",
        "eps_start": 1.0,
        "eps_end": 0.1,
        "eps_cliff": 1000000,
        "frame_history_len": 4,
        "gamma": 0.99,
        "learning_freq": 1,
        "learning_start": 50000,
        "logs_dir": "logs",
        "log_freq": 10,
        "lr": 0.00025,
        "memory_size": 1000000,
        "model_name": "dqn",
        "out_dir": "out",
        "target_update_freq": 10000,
        "total_timesteps": 200000,
    }
    dqn = DQNAgent(settings)
    dqn.train()
