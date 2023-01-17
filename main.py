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
#from ray.rllib.algorithms.dqn import DQN

from utils.trainer.Trainer import Trainer
from utils.agents.DQN import DQN
import copy

from models.Game import Game
from models.MyKerasQModel import MyKerasQModel


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

class Config(object):
    """Object to hold the config requirements for an agent/game"""
    def __init__(self):
        self.seed = None
        self.environment = None
        self.requirements_to_solve_game = None
        self.num_episodes_to_run = None
        self.file_to_save_data_results = None
        self.file_to_save_results_graph = None
        self.runs_per_agent = None
        self.visualise_overall_results = None
        self.visualise_individual_results = None
        self.hyperparameters = None
        self.use_GPU = None
        self.overwrite_existing_results_file = None
        self.save_model = False
        self.standard_deviation_results = 1.0
        self.randomise_random_seed = True
        self.show_solution_score = False
        self.debug_mode = False

def build_config_dict(cfg):

    config = Config()
    config.seed = 1
    config.environment = Game(cfg)
    config.num_episodes_to_run = cfg.TRAIN.episode
    config.file_to_save_data_results = "results/"
    config.file_to_save_results_graph = "results/"
    config.show_solution_score = False
    config.visualise_individual_results = False
    config.visualise_overall_agent_results = True
    config.standard_deviation_results = 1.0
    config.runs_per_agent = 1
    config.use_GPU = False
    config.overwrite_existing_results_file = False
    config.randomise_random_seed = True
    config.save_model = False
    config.action_size = cfg.SIMULATOR.num_actions
    config.state_size = cfg.SIMULATOR.num_obs

    config.hyperparameters = {
        "learning_rate": cfg.TRAIN.learning_rate,
        "batch_size": cfg.TRAIN.batch_size,
        "buffer_size": cfg.TRAIN.buffer_size,
        "epsilon": cfg.TRAIN.epsilon,
        "epsilon_decay_rate_denominator": cfg.TRAIN.epsilon_decay_rate_denominator,
        "discount_rate": cfg.TRAIN.discount_rate,
        "tau": cfg.TRAIN.tau,
        "alpha_prioritised_replay": cfg.TRAIN.alpha_prioritised_replay,
        "beta_prioritised_replay": cfg.TRAIN.beta_prioritised_replay,
        "incremental_td_error": cfg.TRAIN.incremental_td_error,
        "update_every_n_steps": cfg.TRAIN.update_every_n_steps,
        "linear_hidden_units": cfg.TRAIN.linear_hidden_units,
        "final_layer_activation": cfg.TRAIN.final_layer_activation,
        "batch_norm": cfg.TRAIN.batch_norm,
        "gradient_clipping_norm": cfg.TRAIN.gradient_clipping_norm,
        "learning_iterations": cfg.TRAIN.learning_iterations,
        "clip_rewards": cfg.TRAIN.clip_rewards,
    }
    return config

def build_agents(agent_config, cfg, agent_class):
    agent_list = []
    for i in range(cfg.SIMULATOR.node_count):
        agent = agent_class(copy.deepcopy(agent_config), cfg, agent_idx = i)
        agent_list.append(copy.deepcopy(agent))
    return agent_list


def main(args):
    print(args)
    print(f'python version = {sys.version}')
    # print('cfg ------')
    # print(cfg)
    config = build_config_dict(cfg)
    agents = build_agents(config, cfg, DQN)
    trainer = Trainer(config = config, cfg_yaml = cfg, agents = agents)
    trainer.run_games_for_agents()



    # register_env("Game", lambda _: Game(cfg))
    # ray.init()

    # ModelCatalog.register_custom_model("MLPModel", MyKerasQModel)

    #bound for (population, symp_city/pop_city, symp_all/pop_all, recovered_city/pop_city, 
        # dead_city/pop_city, ExpPopIn_city, local_inc_city/pop_city, local_inc_all/pop_all)
    # low_bound = np.array(cfg.SIMULATOR.obs_low_bound)
    # up_bound = np.array(cfg.SIMULATOR.obs_up_bound)
    # observation_space = gym.spaces.Box(low=low_bound, high=up_bound, shape=(cfg.SIMULATOR.num_obs,))
    # act_space = gym.spaces.Discrete(cfg.SIMULATOR.num_actions)
    #trainer = build_trainer(observation_space, act_space, cfg.TRAIN.gamma)
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
    # max_dict = {}
    # mean_dict = {}
    # min_dict = {}

    # for i in range(cfg.TRAIN.episode):
    #     trainer_result = trainer.train()
    #     with open("results/log.txt", 'a') as f:
    #         f.write(pretty_print(trainer_result))
    #         ckpt_path = save_checkpoint(trainer)
        
    #     max_dict[i] = trainer_result["sampler_results"]["episode_reward_max"]          
    #     mean_dict[i] = trainer_result["sampler_results"]["episode_reward_mean"]                                       
    #     min_dict[i] = trainer_result["sampler_results"]["episode_reward_min"] 
        
    #     #Every 5 epochs, update output data
    #     if i % 5 == 0:
    #         fig, ax = plt.subplots(1)
    #         pd.DataFrame(mean_dict.values(), index = np.arange(1, len(mean_dict) + 1)).plot(color = "b", label = "Mean_Reward", ax=ax) 
    #         plt.fill_between(list(max_dict.keys()), list(min_dict.values()), list(max_dict.values()),color="b", alpha=0.2)
            
    #         ax.legend(["Mean Reward", "Min-Max Reward"])
    #         ax.set_xlabel('Episodes')
    #         ax.set_ylabel('Reward')
    #         filename = "reward_" + str(i)
    #         plt.title("Episode " + str(i))
    #         plt.savefig("results/reward/{0}.png".format(filename))
    #         plt.close()


    #         # define a dictionary with key value pairs

    #         f = open("results/policy/output_max.csv", "w")
    #         max_w = csv.writer(f)

    #         # loop over dictionary keys and values
    #         for key, val in max_dict.items():
    #             # write every key and value to file
    #             max_w.writerow([key, val])

    #         f.close()

    #         l = open("results/policy/output_mean.csv", "w")
    #         mean_w = csv.writer(l)

    #         # loop over dictionary keys and values
    #         for key, val in mean_dict.items():
    #             # write every key and value to file
    #             mean_w.writerow([key, val])

    #         l.close()

    #         k = open("results/policy/output_min.csv", "w")
    #         min_w = csv.writer(k)

    #         # loop over dictionary keys and values
    #         for key, val in min_dict.items():
    #             # write every key and value to file
    #             min_w.writerow([key, val])

    #         k.close()

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





    




