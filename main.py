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

from tqdm.notebook import tqdm
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQN
#TODO: swap with ApexTrainer

from models.Game import Game
from models.MyKerasQModel import MyKerasQModel
from models.TorchQModel import CustomDQNModel
import ray




POPULATION = 10000
NUM_CITIES = 100

print(f'python version = {sys.version}')
register_env("Game", lambda _: Game())
ray.init()

ModelCatalog.register_custom_model("MLPModel", MyKerasQModel)
#ModelCatalog.register_custom_model("MLPModel", CustomDQNModel)

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

def save_checkpoint(trainer):
    return trainer.save("./ckpt")

def main(args):
    print("*"*10)
    print(args)
    print('*'*10)

    for i in range(500):

        # if(i%10==0):
        #     ckpt_path = save_checkpoint(trainer)
        #     print(f'path = {ckpt_path}')
        #     #print(f'checkpoint saved, trainer empty = {trainer==None}')
        #     trainer.restore(ckpt_path)

        with open("results/log.txt", 'a') as f:
            x = trainer.train()
            print(type(x))
            print(x.keys())
            f.write(pretty_print(trainer.train()))

       
# ------------------------------------------------------------------------
# Training code for bilinear similarity network (BMNet and BMNet+)
# --cfg: path for configuration file
# ------------------------------------------------------------------------
# import random
# from pathlib import Path

# import numpy as np
# import torch
# from torch.utils.data import DataLoader


# import util.misc as utils
# from FSC147_dataset import build_dataset, batch_collate_fn 
# from engine import evaluate, train_one_epoch, visualization 
# from models import build_model

# import matplotlib.pyplot as plt
# plt.switch_backend('agg')

# def main(args):
#     print(args)
#     device = torch.device(cfg.TRAIN.device)
#     # fix the seed for reproducibility
#     seed = cfg.TRAIN.seed
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)

        
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.lr_drop)
#     # define dataset
#     dataset_train = build_dataset(cfg, is_train=True)
#     output_dir = Path(cfg.DIR.output_dir)
    
#     loss_list = []
#     val_mae_list = []
    
#     if cfg.VAL.evaluate_only:
#         if os.path.isfile(cfg.VAL.resume):
#             checkpoint = torch.load(cfg.VAL.resume, map_location='cpu')
#             model.load_state_dict(checkpoint['model'])
#         else:
#             print('model state dict not found.')
#         if cfg.VAL.visualization:
#             mae = visualization(cfg, model, dataset_val, data_loader_val, device, cfg.DIR.output_dir)
#         else:
#             mae = evaluate(model, data_loader_val, device, cfg.DIR.output_dir)
#         return
    
#     if os.path.isfile(cfg.TRAIN.resume):
#         if cfg.TRAIN.resume.startswith('https'):
#             checkpoint = torch.hub.load_state_dict_from_url(
#                 cfg.TRAIN.resume, map_location='cpu', check_hash=True)
#         else:
#             checkpoint = torch.load(cfg.TRAIN.resume, map_location='cpu')
#         model.load_state_dict(checkpoint['model'])
#         if not cfg.VAL.evaluate_only and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'episode' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer'])
#             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
#             cfg.TRAIN.start_episode = checkpoint['episode'] + 1
#             loss_list = checkpoint['loss']
#             val_mae_list = checkpoint['val_mae']

#     best_mae = 10000 if len(val_mae_list) == 0 else min(val_mae_list)
#     best_mae = 10000
    
#     print("Start training")
#     start_time = time.time()
    
#     for episode in range(cfg.TRAIN.start_episode, cfg.TRAIN.episodes):
#         loss = train_one_epoch(
#             model, criterion, data_loader_train, optimizer, device, episode,
#             cfg.TRAIN.clip_max_norm)
        
#         mae, mse = evaluate(model, data_loader_val, device, cfg.DIR.output_dir)
#         loss_list.append(loss)
#         val_mae_list.append(mae)
#         lr_scheduler.step()
        
#         utils.plot_learning_curves(loss_list, val_mae_list, cfg.DIR.output_dir)
        
#         if cfg.DIR.output_dir:
#             with (output_dir / "log.txt").open("a") as f:
#                 f.write('episode %d: loss %.8f, MAE %.2f, MSE %.2f, Best MAE %.2f, Best MSE %.2f \n'%(episode +1, loss, mae, mse, best_mae, best_mse))


#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print('Training time {}'.format(total_time_str))

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

    cfg.TRAIN.resume = os.path.join(cfg.DIR.output_dir, cfg.TRAIN.resume)
    cfg.VAL.resume = os.path.join(cfg.DIR.output_dir, cfg.VAL.resume)

    with open(os.path.join(cfg.DIR.output_dir, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    main(cfg)





    




