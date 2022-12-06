import numpy as np
from numpy import random
import pandas as pd
import gym
import sys
import argparse
from config import cfg 
import os 
import time
import datetime
import matplotlib.pyplot as plt
import csv

from tqdm.notebook import tqdm
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQN

from models.Game import Game
from models.MyKerasQModel import MyKerasQModel
import ray


def save_checkpoint(trainer):
    return trainer.save("./ckpt")

def build_trainer(observation_space, act_space, gamma):
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "MLPModel",
            },
            "gamma": gamma,
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
    return trainer

def main(args):
    print(args)
    print(f'python version = {sys.version}')
    register_env("Game", lambda _: Game(cfg))
    ray.init()

    ModelCatalog.register_custom_model("MLPModel", MyKerasQModel)

    #bound for (population, symp_city/pop_city, symp_all/pop_all, recovered_city/pop_city, 
        # dead_city/pop_city, ExpPopIn_city, local_inc_city/pop_city, local_inc_all/pop_all)
    low_bound = np.array(cfg.SIMULATOR.obs_low_bound)
    up_bound = np.array(cfg.SIMULATOR.obs_up_bound)
    observation_space = gym.spaces.Box(low=low_bound, high=up_bound, shape=(cfg.SIMULATOR.num_obs,))
    act_space = gym.spaces.Discrete(cfg.SIMULATOR.num_actions)
    trainer = build_trainer(observation_space, act_space, cfg.TRAIN.gamma)
    # print(f'trainer = {trainer}')
    # print(f'trainer attributes = { dir(trainer)}')
    # for (key, value) in vars(trainer).items():
    #     print(f'trainer.{key} = {value}')
    #     print(str(type(value)))
    #     if(str(type(value)) not in ["<class 'NoneType'>", "<class 'float'>", "<class 'int'>", "<class 'str'>", "<class 'list'>", "<class 'bool'>"]):
    #         iter_var = None
    #         if(str(type(value)).__contains__("dict")):
    #             iter_var = value.items()
    #         else:
    #             iter_var = vars(value).items()

    #         for (key, value) in iter_var:
    #             print(f'{key} : {value}')
    #         print("-"*15)
    #         print("\n")
    # exit()
    max_dict = {}
    mean_dict = {}
    min_dict = {}

    for i in range(cfg.TRAIN.episode):
        trainer_result = trainer.train()
        with open("results/log.txt", 'a') as f:
            f.write(pretty_print(trainer_result))
            ckpt_path = save_checkpoint(trainer)
        
        max_dict[i] = trainer_result["sampler_results"]["episode_reward_max"]          
        mean_dict[i] = trainer_result["sampler_results"]["episode_reward_mean"]                                       
        min_dict[i] = trainer_result["sampler_results"]["episode_reward_min"] 
        
        #Every 5 epochs, update output data
        if i % 5 == 0:
            fig, ax = plt.subplots(1)
            pd.DataFrame(mean_dict.values(), index = np.arange(1, len(mean_dict) + 1)).plot(color = "b", label = "Mean_Reward", ax=ax) 
            plt.fill_between(list(max_dict.keys()), list(min_dict.values()), list(max_dict.values()),color="b", alpha=0.2)
            
            ax.legend(["Mean Reward", "Min-Max Reward"])
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Reward')
            filename = "reward_" + str(i)
            plt.title("Episode " + str(i))
            plt.savefig("results/reward/{0}.png".format(filename))
            plt.close()


            # define a dictionary with key value pairs

            f = open("results/policy/output_max.csv", "w")
            max_w = csv.writer(f)

            # loop over dictionary keys and values
            for key, val in max_dict.items():
                # write every key and value to file
                max_w.writerow([key, val])

            f.close()

            l = open("results/policy/output_mean.csv", "w")
            mean_w = csv.writer(l)

            # loop over dictionary keys and values
            for key, val in mean_dict.items():
                # write every key and value to file
                mean_w.writerow([key, val])

            l.close()

            k = open("results/policy/output_min.csv", "w")
            min_w = csv.writer(k)

            # loop over dictionary keys and values
            for key, val in min_dict.items():
                # write every key and value to file
                min_w.writerow([key, val])

            k.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepQ network with SEIR"
    )
    parser.add_argument(
        "--cfg",
        default="config/config.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    print(f'args cfg = {args.cfg}')
    cfg.merge_from_file(args.cfg)
    
    cfg.DIR.output_dir = os.path.join(cfg.DIR.snapshot, cfg.DIR.exp)
    if not os.path.exists(cfg.DIR.output_dir):
        os.mkdir(cfg.DIR.output_dir)
    reward_dir = os.path.join(cfg.DIR.output_dir, "reward")
    if(not os.path.exists(reward_dir)):
        os.mkdir(reward_dir)


    cfg.TRAIN.resume = os.path.join(cfg.DIR.output_dir, cfg.TRAIN.resume)

    with open(os.path.join(cfg.DIR.output_dir, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    main(cfg)





    




