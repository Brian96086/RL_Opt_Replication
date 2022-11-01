import numpy as np
from numpy import random
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import scipy.integrate
import gym

from tqdm.notebook import tqdm

import datetime
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQN
#TODO: swap with ApexTrainer

from Game import Game
from MyKerasQModel import MyKerasQModel
import ray



POPULATION = 10000
NUM_CITIES = 100

register_env("Game", lambda _: Game())
ray.init()

ModelCatalog.register_custom_model("MLPModel", MyKerasQModel)

#observation_space = gym.spaces.Box(low=0, high=1, shape=(8,))
observation_space = gym.spaces.Box(low=0, high=1000, shape=(8,))
act_space = gym.spaces.Discrete(2)

def gen_policy(i):
    config = {
        "model": {
            "custom_model": "MLPModel",
        },
        "gamma": 0.99,
        #"parameter_noise": False
    }
    return (None, observation_space, act_space, config)

policies = {"policy_0": gen_policy(0)}
policy_ids = list(policies.keys())

trainer = DQN(
#trainer = ApexTrainer(
# trainer = ray.rllib.agents.dqn.DQNTrainer(
    env="Game",
    config={
        "env_config": {},

        # General
#         "log_level": "ERROR",

        # Method specific
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (
                lambda agent_id: policy_ids[0]),
        },
        "lr": 0.0005
    },
)

for i in range(500):
    print(pretty_print(trainer.train()))




