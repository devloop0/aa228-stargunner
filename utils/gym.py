"""
    This file is apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import gym
from gym import wrappers

from utils.seed import set_global_seeds
from utils.atari_wrapper import wrap_deepmind


def get_env(task, seed, monitor=True):
    env = gym.make(task)

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = "tmp/gym-results"
    if monitor:
        env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind(env)

    return env


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)
