from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yacs.config import CfgNode as CN

_C = CN()

# ----- BASIC SETTINGS -----
_C.NAME = "debug"
_C.OUTPUT_DIR = 'checkpoint/'
_C.MODE = 'train'
_C.SEED = 0
_C.GPU = '1'

# ----- DATASET SETTINGS ------
_C.DATA = CN()
_C.DATA.NAME = 'awa2'
_C.DATA.ROOT = '/home/dev/Data/xiao/data/'
_C.DATA.DOMAIN = 5
_C.DATA.CLASS = 50
_C.DATA.SOURCE = 'HSUV'
_C.DATA.TARGET = 'HSUVO'
_C.DATA.BATCH_SIZE = 48
_C.DATA.NUM_WORKERS = 8

# ----- Augmentations --------
_C.AUGMENTATION = CN()
_C.AUGMENTATION.FLAG = False

# ----- Models -------
_C.MODEL = CN()
_C.MODEL.NAME = 'sfsc_meta'
_C.MODEL.F = 'resnet18'
_C.MODEL.C = 'fc'
_C.MODEL.PRETRAIN_FLAG = False

# ----- Loss  --------
_C.LOSS = CN()
_C.LOSS.TYPE = 'BSCE'
_C.LOSS.PER_DOMAIN_FLAG = True

# ----- Train -----
_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCH = 100
_C.TRAIN.ITER_PER_EPOCH = 101
_C.TRAIN.PRINT_FREQ = 10
_C.TRAIN.PRINT_VAL = False
## OPTIMIZER
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'sgd'
_C.TRAIN.OPTIMIZER.BASELR = 0.1
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 1e-4

## SCHEDULER
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.TYPE = 'steplr'
_C.TRAIN.LR_SCHEDULER.LR_STEP = [10, 20]
_C.TRAIN.LR_SCHEDULER.LR_FACTOR = 0.5
_C.TRAIN.LR_SCHEDULER.WARM_EPOCH = 0

# -------- EMBED --------
_C.EMBED = CN()
_C.EMBED.FLAG = False
_C.EMBED.NAME = 'glove'
_C.EMBED.MARGIN = 0.1
_C.EMBED.SCALE = 30.

# -------- AUG --------
_C.AUG = CN()
_C.AUG.FLAG = False
_C.AUG.EPOCH = 40
_C.AUG.LAM = 10.

# -------- META --------
_C.META = CN()
_C.META.FLAG = False
_C.META.META_STEP_SIZE = 0.2
_C.META.STOP_GRADIENT = True

# -------- Weight --------
_C.W = CN()
_C.W.V2S = 0.1
_C.W.S2S = 0.1
_C.W.S2V = 0.1
_C.W.AUG = 0.1
_C.W.V2S_M = 0.1
_C.W.AUG_M = 0.1
_C.W.VAL = 0.3

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    print(cfg.dump())