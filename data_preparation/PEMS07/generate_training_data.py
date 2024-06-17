import os
import sys
import shutil
import pickle
import argparse

import numpy as np

from generate_adj_mx import generate_adj_pems08
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
#from basicts.data.transform import standard_transform
from stkriging.data.transform import standard_transform


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.
    Default settings of METR-LA dataset:
        - Normalization method: standard norm.
        - Dataset time dimension division: 7:3;
                  node dimension division: 7:1:2;
                  you can select the rate.
        - Window size:  24 or 96 , you can select the dimension of time series
        - Channels (features): three channels [traffic flow, time of day, day of week]
        - Target: predict the traffic flow of the unknown nodes.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    seqlen = args.seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    train_series_ratio = args.train_series_ratio
    valid_test_split = args.valid_test_split
    valid_series_ratio = args.valid_series_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    seed = args.seed
    steps_per_day = args.steps_per_day
    origin_oir = args.origin_dir
    # read data
    data = np.load(data_file_path)["data"]
    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    l, n, f = data.shape
    num_samples = l - (seqlen) + 1
    train_num_short = round(num_samples * train_series_ratio)
    if valid_test_split:
        valid_num_short = round(num_samples * valid_series_ratio)
    else:
        valid_num_short = 0
    other_num_short = num_samples - train_num_short - valid_num_short
    print("number of training samples:{0}".format(train_num_short))
    print("number of validation samples:{0}".format(valid_num_short))
    print("number of test samples:{0}".format(other_num_short))
    index_list = []
    for t in range(0, num_samples):
        index = (t, t + seqlen)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    if valid_test_split:
        valid_index = index_list[train_num_short: train_num_short + valid_num_short]
        test_index = index_list[train_num_short +
                                valid_num_short: train_num_short + valid_num_short + other_num_short]
    else:
        test_index = index_list[train_num_short: train_num_short + other_num_short]
        valid_index = test_index

    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, seqlen)
    print("   hererere", train_index[-1], train_index[-1][1])
    feature_list = [data_norm]

    node_index = np.array([i for i in range(n)])
    np.random.seed(seed)
    np.random.shuffle(node_index)

    num_samples = n
    train_num_short = round(num_samples * train_ratio)
    valid_num_short = round(num_samples * valid_ratio)
    test_num_short = num_samples - train_num_short - valid_num_short

    print("number of training nodes samples:{0}".format(train_num_short))
    print("number of validation nodes samples:{0}".format(valid_num_short))
    print("number of test nodes samples:{0}".format(test_num_short))


    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day /
               steps_per_day for i in range(l)]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)


    if add_day_of_week:
        # numerical day_of_week
        dow = [(i // steps_per_day) % 7 for i in range(l)]
        dow = np.array(dow)
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
    processed_data = np.concatenate(feature_list, axis=-1)
    print(processed_data.dtype,'22222222222222222222222')
    train_node = node_index[:train_num_short]
    valid_node = node_index[train_num_short:train_num_short + valid_num_short]
    test_node = node_index[train_num_short + valid_num_short:train_num_short + valid_num_short + test_num_short]

    data = {}
    data["processed_data"] = processed_data

    with open(output_dir + "/data.pkl".format(str(int(train_series_ratio * 10))), "wb") as f:
        pickle.dump(data, f)
    adjindex = {}
    adjindex['train_nodes'] = train_node
    adjindex['valid_nodes'] = valid_node
    adjindex['test_nodes'] = test_node
    with open(output_dir + "/adj_index.pkl", "wb") as f:
        pickle.dump(adjindex, f)
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index.pkl", "wb") as f:
        pickle.dump(index, f)

    # dump data

    # copy adj
    if os.path.exists(args.graph_file_path):
        # copy
        shutil.copyfile(args.graph_file_path, output_dir + "/adj_mx.pkl")
    else:
        # generate and copy
        generate_adj_pems08(origin_oir)
        shutil.copyfile(graph_file_path, output_dir + "/adj_mx.pkl")


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    SEQ_LEN = 24

    TRAIN_SERIES_RATIO = 0.7
    VALID_TEST_SPLIT = False
    VALID_SERIES_RATIO = 0.1
    SEED = 42
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0]                   # target channel(s)
    STEPS_PER_DAY = 288
    ORIGIN_DIR = "D:/myfile/ST-kriging"
    DATASET_NAME = "PEMS07"
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    OUTPUT_DIR = ORIGIN_DIR + "/datasets/" + DATASET_NAME + "_{0}{1}_{2}".format(str(int(TRAIN_RATIO * 10)), str(int(VALID_RATIO * 10)), str(SEQ_LEN))
    DATA_FILE_PATH = ORIGIN_DIR + "/datasets/raw_data/{0}/{0}.npz".format(DATASET_NAME)
    GRAPH_FILE_PATH = ORIGIN_DIR + "/datasets/raw_data/{0}/adj_{0}.pkl".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                        default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--seq_len", type=int,
                        default=SEQ_LEN, help="Sequence Length.")

    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--train_series_ratio", type=float,
                        default=TRAIN_SERIES_RATIO, help="Train Series ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--seed", type=float,
                        default=SEED, help="Random Seed")

    parser.add_argument("--origin_dir", type=str,
                        default=OUTPUT_DIR, help="Original directory.")
    parser.add_argument("--valid_test_split", type=bool,
                        default=False, help="Random Seed")
    parser.add_argument("--valid_series_ratio", type=float,
                        default=VALID_SERIES_RATIO, help="Train Series ratio")
    args_metr = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args_metr).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if os.path.exists(args_metr.output_dir):
        reply = str(input(
            f"{args_metr.output_dir} exists. Do you want to overwrite it? (y/n)")).lower().strip()
        if reply[0] != "y":
            sys.exit(0)
    else:
        os.makedirs(args_metr.output_dir)
    generate_data(args_metr)
