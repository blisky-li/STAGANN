import os
import sys
from argparse import ArgumentParser
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import torch
from stkriging import launch_training

#torch.set_num_threads(1) # aviod high cpu avg usage


#
def parse_args():
    parser = ArgumentParser(description="Run time series forecasting model in BasicTS framework!")

    parser.add_argument("-c", "--cfg", default="examples/STAGANN/STAGANN_PEMS-BAY.py", help="training config")

    parser.add_argument("--gpus", default="0", help="visible gpus")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    launch_training(args.cfg, args.gpus)
