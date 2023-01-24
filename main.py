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
import matplotlib.pyplot as plt
import csv


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


max_dict = {}
mean_dict = {}
min_dict = {}



for i in range(1, 55):
#     print(pretty_print(trainer.train()))
    trainer_result = trainer.train()
    max_dict[i] = trainer_result["sampler_results"]["episode_reward_max"]          
    mean_dict[i] = trainer_result["sampler_results"]["episode_reward_mean"]                                       
    min_dict[i] = trainer_result["sampler_results"]["episode_reward_min"] 
    
    print("epoch number is: -----------------------------------------------------------")
    print(i*20)
    #Every 20 epochs output graph
    fig, ax = plt.subplots(1)

#         max_lists = max_dict.items() # return a list of tuples
#         x, y_max = zip(*max_lists) # unpack a list of pairs into two tuples

#         mean_lists = mean_dict.items() # return a list of tuples
#         x, y_mean = zip(*mean_lists) # unpack a list of pairs into two tuples

#         min_lists = min_dict.items() # return a list of tuples
#         x, y_min = zip(*min_lists) # unpack a list of pairs into two tuples

    pd.DataFrame(mean_dict.values(), index = np.arange(20,len(mean_dict)*20 + 1, 20)).plot(color = "b", label = "Mean_Reward", ax=ax)

    plt.fill_between(np.arange(20,len(mean_dict)*20 + 1, 20), list(min_dict.values()), list(max_dict.values()),color="b", alpha=0.2)

    ax.legend(["Mean Reward", "Min-Max Reward"])
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    filename = "reward_" + str(i*20)
    plt.title("Episode " + str(i*20))
    plt.savefig("results/reward/{0}.png".format(filename))
    plt.close()


    # define a dictionary with key value pairs

    # open file for writing, "w" is writing
    f = open("output_max.csv", "w")
    max_w = csv.writer(f)

    # loop over dictionary keys and values
    for key, val in max_dict.items():

        # write every key and value to file
        max_w.writerow([key, val])

    f.close()

    l = open("output_mean.csv", "w")
    mean_w = csv.writer(l)

    # loop over dictionary keys and values
    for key, val in mean_dict.items():

        # write every key and value to file
        mean_w.writerow([key, val])

    l.close()

    k = open("output_min.csv", "w")
    min_w = csv.writer(k)

    # loop over dictionary keys and values
    for key, val in min_dict.items():

        # write every key and value to file
        min_w.writerow([key, val])

    k.close()