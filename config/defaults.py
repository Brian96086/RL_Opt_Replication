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




# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.backbone = "resnet50"
_C.MODEL.epf_extractor = "direct_pooling"
_C.MODEL.refiner = "self_similarity_module"
_C.MODEL.matcher = "dynamic_similairty_matcher"
_C.MODEL.counter = "local_count"
_C.MODEL.fix_bn = True
_C.MODEL.ep_scale_embedding = False
_C.MODEL.use_bias = False
_C.MODEL.ep_scale_number = 20
_C.MODEL.backbone_layer = "layer4"
_C.MODEL.hidden_dim = 256
_C.MODEL.refiner_proj_dim = 256
_C.MODEL.matcher_proj_dim = 256
_C.MODEL.dynamic_proj_dim = 128
_C.MODEL.dilation = False
_C.MODEL.refiner_layers = 6
_C.MODEL.matcher_layers = 6
_C.MODEL.repeat_times = 1
# dim of counter
_C.MODEL.counter_dim = 256
# use pretrained model
_C.MODEL.pretrain = True
# fix bn params, only under finetuning
_C.MODEL.fix_bn = False


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# restore training from a checkpoint
_C.TRAIN.resume = "model_ckpt.pth.tar"
# numbers of exemplar boxes
_C.TRAIN.exemplar_number = 3
# loss function
_C.TRAIN.counting_loss = "l1loss"
_C.TRAIN.contrast_loss = "info_nce"
# weight for contrast loss
_C.TRAIN.contrast_weight = 1e-5
# loss reduction
_C.TRAIN.loss_reduction = "mean"
# batch size
_C.TRAIN.batch_size = 1
# epochs to train for
_C.TRAIN.epochs = 20
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000
# optimizer and learning rate
_C.TRAIN.optimizer = "AdamW"
_C.TRAIN.lr_backbone = 0.01
_C.TRAIN.lr = 0.01
# milestone
_C.TRAIN.lr_drop = 200
# momentum
_C.TRAIN.momentum = 0.95
# weights regularizer
_C.TRAIN.weight_decay = 5e-4
# gradient clipping max norm
_C.TRAIN.clip_max_norm = 0.1
# number of data loading workers
_C.TRAIN.num_workers = 0
# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 2020
_C.TRAIN.start_epoch = 0
_C.TRAIN.device = 'cuda:0'

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# the checkpoint to evaluate on
_C.VAL.resume = "model_best.pth.tar"
# currently only supports 1
_C.VAL.batch_size = 1
# frequency to display
_C.VAL.disp_iter = 10
# frequency to validate
_C.VAL.val_epoch = 10
# evaluate_only
_C.VAL.evaluate_only = False
_C.VAL.visualization = False