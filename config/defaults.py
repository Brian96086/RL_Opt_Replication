from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.sim_type = "naive_seir"
_C.SIMULATOR.population = 10000
_C.SIMULATOR.node_count = 100
_C.SIMULATOR.init_infected = 50
_C.SIMULATOR.obs_low_bound = [0, 0, 0, 0, 0, 0, -1, -1]
_C.SIMULATOR.obs_up_bound = [10000, 1, 1, 1, 1, 10000, 1, 1]
_C.SIMULATOR.num_actions = 2
_C.SIMULATOR.num_obs = 8
_C.SIMULATOR.num_weeks = 52


# -----------------------------------------------------------------------------
# FILES
# -----------------------------------------------------------------------------
_C.DIR = CN()
_C.DIR.exp = ""
_C.DIR.snapshot = ""

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.backbone = "deepQ"
_C.MODEL.hidden_dim = 256
_C.MODEL.refiner_proj_dim = 256
_C.MODEL.matcher_proj_dim = 256
_C.MODEL.dynamic_proj_dim = 128
_C.MODEL.refiner_layers = 6
_C.MODEL.matcher_layers = 6
_C.MODEL.repeat_times = 1
# dim of counter
_C.MODEL.counter_dim = 256
# use pretrained model
_C.MODEL.pretrain = True


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# restore training from a checkpoint
_C.TRAIN.resume = "checkpoint_000000"
_C.TRAIN.episode = 100
#_C.TRAIN.gamma = 0.99
# optimizer and learning rate
_C.TRAIN.optimizer = "AdamW"
_C.TRAIN.lr = 0.01
# momentum
_C.TRAIN.momentum = 0.95
# weights regularizer
_C.TRAIN.device = 'cuda:0'

_C.TRAIN.learning_rate = 0.01
_C.TRAIN.batch_size = 256
_C.TRAIN.buffer_size = 40000
_C.TRAIN.epsilon = 1.0
_C.TRAIN.epsilon_decay_rate_denominator = 1
_C.TRAIN.discount_rate = 0.99
_C.TRAIN.tau = 0.01
_C.TRAIN.alpha_prioritised_replay = 0.6
_C.TRAIN.beta_prioritised_replay = 0.1
_C.TRAIN.incremental_td_error = 1e-8
_C.TRAIN.update_every_n_steps = 1
_C.TRAIN.linear_hidden_units = [30, 15]
_C.TRAIN.final_layer_activation = "None"
_C.TRAIN.batch_norm = False
_C.TRAIN.gradient_clipping_norm = 0.7
_C.TRAIN.learning_iterations = 1
_C.TRAIN.clip_rewards = False

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# frequency to display
_C.VAL.disp_iter = 10
# frequency to validate
_C.VAL.val_epoch = 10
# evaluate_only
_C.VAL.evaluate_only = False
