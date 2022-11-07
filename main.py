import numpy as np
from numpy import random
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import scipy.integrate
import gym

from tqdm.notebook import tqdm

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQN
#TODO: swap with ApexTrainer

from models.Game import Game
from models.MyKerasQModel import MyKerasQModel
import ray



POPULATION = 10000
NUM_CITIES = 100

register_env("Game", lambda _: Game())
ray.init()

ModelCatalog.register_custom_model("MLPModel", MyKerasQModel)

#bound for (population, symp_city/pop_city, symp_all/pop_all, recovered_city/pop_city, 
    # dead_city/pop_city, ExpPopIn_city, local_inc_city/pop_city, local_inc_all/pop_all)
low_bound = np.array([0, 0, 0, 0, 0, 0, -1, -1])
up_bound = np.array([10000, 1, 1, 1, 1, 10000, 1, 1])
observation_space = gym.spaces.Box(low=low_bound, high=up_bound, shape=(8,))
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




