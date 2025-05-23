import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from stkriging.archs import Okriging
from stkriging.runners import SimpleSpatiotemporalKrigingRunner
from stkriging.data import STKrigingDataset
from stkriging.losses import masked_mae
from stkriging.utils import load_adj


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "Okriging model configuration"
CFG.RUNNER = SimpleSpatiotemporalKrigingRunner
CFG.DATASET_CLS = STKrigingDataset
CFG.DATASET_TRAINRATIO = 7
CFG.DATASET_VALRATIO = 1
CFG.DATASET_LEN = 24
CFG.DATASET_NAME = "PEMS08_{0}{1}_{2}".format(str(CFG.DATASET_TRAINRATIO), str(CFG.DATASET_VALRATIO), CFG.DATASET_LEN)
CFG.DATASET_TYPE = "Traffic speed"

CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 42
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "Okriging"
CFG.MODEL.ARCH = Okriging
adj_mx, train_index, valid_index, test_index = load_adj("datasets/" + CFG.DATASET_NAME +"/adj_mx.pkl",
                                                        "datasets/" + CFG.DATASET_NAME +"/adj_index.pkl")

CFG.DATASET = EasyDict()
CFG.DATASET.MATRIX = adj_mx
CFG.DATASET.MATRIX_TRANSFORM = "original"
CFG.DATASET.TRANSFORM = "standard_transform"# OR standard_transform
CFG.DATASET.TRAININDEX = train_index
CFG.DATASET.VALIDINDEX = valid_index
CFG.DATASET.TESTINDEX = test_index
CFG.MODEL.PARAM = {


}

CFG.MODEL.FORWARD_FEATURES = [0, 1,2]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = {'masked_mae': masked_mae}
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0000002,
    "weight_decay": 1.0e-6,
    'eps':1.0e-8
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [30, 70, 120, 160],
    "gamma": 0.5
}

# ================= train ================= #
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}
CFG.TRAIN.NUM_EPOCHS = 1
CFG.TRAIN.BATCH_SELECT_NODE_RANDOM = True
CFG.TRAIN.BATCH_MATRIX_RANDOM = False
CFG.TRAIN.Mask_Matrix = False
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints",
    "_".join([CFG.MODEL.NAME, str(CFG.TRAIN.NUM_EPOCHS)])
)
# train data
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.NULL_VAL = 0.0
CFG.TRAIN.RATIO = 0.7
# read data
CFG.TRAIN.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TRAIN.DATA.BATCH_SIZE = 12
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 12
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 12
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False