import os
import sys

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from easydict import EasyDict
from stkriging.archs import STAGANN
from stkriging.runners import SimpleSpatiotemporalKrigingRunner
from stkriging.data import STKrigingDataset, read_location
from stkriging.losses import masked_mae, masked_binary
from stkriging.utils import load_adj


CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "STAGANN model configuration"
CFG.RUNNER = SimpleSpatiotemporalKrigingRunner
CFG.DATASET_CLS = STKrigingDataset
CFG.DATASET_TRAINRATIO = 7
CFG.DATASET_VALRATIO = 1
CFG.DATASET_LEN = 24
CFG.DATASET_NAME = "AQI_{0}{1}_{2}".format(str(CFG.DATASET_TRAINRATIO), str(CFG.DATASET_VALRATIO), CFG.DATASET_LEN)
CFG.DATASET_TYPE = "Traffic speed"

CFG.GPU_NUM = 1

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 42
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = "STAGANN"
CFG.MODEL.ARCH = STAGANN
adj_mx, train_index, valid_index, test_index = load_adj("datasets/" + CFG.DATASET_NAME +"/adj_mx.pkl",
                                                        "datasets/" + CFG.DATASET_NAME +"/adj_index.pkl")

CFG.DATASET = EasyDict()
CFG.DATASET.LOCATION = read_location("datasets/" + CFG.DATASET_NAME + '/location.pkl')
CFG.DATASET.MATRIX = adj_mx
CFG.DATASET.MATRIX_TRANSFORM = "original"
CFG.DATASET.TRANSFORM = "standard_transform"
CFG.DATASET.TRAININDEX = train_index
CFG.DATASET.VALIDINDEX = valid_index
CFG.DATASET.TESTINDEX = test_index
CFG.MODEL.PARAM = {
    "time_dimension" : 24,
    "hidden_dimnesion" : 100,
    "order" : 2,
    "meta_data": 1,
    "num_grids": 28,
"mask": 5,
    'h_of_d':True,
    'd_of_w':True,
    'm_of_y':False,
'dropout':0.3
}
CFG.MODEL.FORWARD_FEATURES = [0, 1,2]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = {'masked_mae': masked_mae,  "masked_binary": masked_binary}# dilate_loss"linear_mmd": linear_mmd   "mmd": mmd   "coral": coral   "rsd": rsd    "masked_binary": masked_binary "cos": cosine
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.0003,
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
CFG.TRAIN.NUM_EPOCHS = 50
CFG.TRAIN.BATCH_SELECT_NODE_RANDOM = True
CFG.TRAIN.BATCH_MATRIX_RANDOM = False
CFG.TRAIN.Mask_Matrix = False
#CFG.TRAIN.BATCH_RANDOM = True

'''CFG.TRAIN.BATCH_SELECT_NODE_RANDOM = True
CFG.TRAIN.BATCH_MATRIX_RANDOM = False
CFG.TRAIN.Mask_Matrix = False'''
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
CFG.TRAIN.DATA.BATCH_SIZE = 32
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 1
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
# validating data
CFG.VAL.DATA = EasyDict()
# read data
CFG.VAL.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 1
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
# test data
CFG.TEST.DATA = EasyDict()
# read data
CFG.TEST.DATA.DIR = "datasets/" + CFG.DATASET_NAME
# dataloader args, optional
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 1
CFG.TEST.DATA.PIN_MEMORY = False