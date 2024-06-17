import os
import sys
import shutil
import pickle
import argparse
from math import radians, cos, sin, asin, sqrt
import torch
import joblib
import numpy as np
import pandas as pd

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
#from basicts.data.transform import standard_transform
#from stkriging.data.transform import min_max_transform
from stkriging.data.transform import standard_transform
def process_location(url):
    lst = []
    a = np.loadtxt(url, delimiter=",")
    #print(a.keys())
    '''lat = np.expand_dims(np.array(a['latitude'].values, dtype=np.float64),axis=-1)
    lon = np.expand_dims(np.array(a['longitude'].values, dtype=np.float64),axis=-1)
    lst = np.concatenate([lon, lat], axis=-1)'''
    return a

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r / 1000

def generate_ushcn_data(ori):
    """

    :param ori:
    :return: time series and matrix

    Modified from IGNNK
    """
    pos = []
    num = 1218
    Utensor = np.zeros((num, 120, 12, 2)) # 1218
    Omissing = np.ones((num, 120, 12, 2))
    with open(ori+"/raw_data/USHCN/Ulocation.txt", "r") as f:
        loc = 0
        for line in f.readlines():
            poname = line[0:11]
            pos.append(line[13:30])
            with open(ori+"/raw_data/USHCN/ushcn.v2.5.5.20191231/" + poname + ".FLs.52j.prcp", "r") as fp:
                temp = 0
                for linep in fp.readlines():
                    if int(linep[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linep[17 + 9 * i:22 + 9 * i]
                            p_temp = int(str_temp)
                            if p_temp == -9999:
                                Omissing[loc, temp, i, 0] = 0
                            else:
                                Utensor[loc, temp, i, 0] = p_temp / 100
                        temp = temp + 1
            with open(ori+"/raw_data/USHCN/ushcn.v2.5.5.20191231/" + poname + ".FLs.52j.tavg", "r") as ft:
                temp = 0
                for linet in ft.readlines():
                    if int(linet[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linet[17 + 9 * i:22 + 9 * i]
                            t_temp = int(str_temp)
                            if t_temp == -9999:
                                Omissing[loc, temp, i, 1] = 0
                            else:
                                Utensor[loc, temp, i, 1] = t_temp
                        temp = temp + 1
            loc = loc + 1


    print(Utensor[:,:,:,0].max(), Utensor[:,:,:,0].min(), Utensor[:,:,:,0].mean())
    latlon = np.loadtxt(ori+"/raw_data/USHCN/latlon.csv", delimiter=",")
    sim = np.zeros((num, num))
    Utensor = np.around(Utensor, decimals=4).astype('float32')
    for i in range(num):
        for j in range(num):
            sim[i, j] = haversine(latlon[i, 1], latlon[i, 0], latlon[j, 1], latlon[j, 0])  # RBF
    print(sim.max(), sim.min(), Utensor.mean())
    sim = np.around(sim, decimals=4).astype('float32')
    sigma = sim.std()
    sim = np.exp(-np.square(sim /sigma)).astype('float32')
    sim[sim < 0.6] = 0.
    print(sim)
    return Utensor, sim



def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.
    Default settings of USHCN dataset:
        - Normalization method: standard norm.
        - Dataset time dimension division: 7:3;
                  node dimension division: 7:1:2;
                  you can select the rate.
        - Window size:  24 or 96 , you can select the dimension of time series
        - Channels (features): three channels [precipitation, time of day, day of week]
        - Target: predict the precipitation of the unknown nodes.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    add_month_of_year = args.moy
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    train_series_ratio = args.train_series_ratio
    valid_test_split = args.valid_test_split
    valid_series_ratio = args.valid_series_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    graph_file_path = args.graph_file_path
    location_file_path = args.location_file_path

    seed = args.seed
    seqlen = args.seq_len
    # read data
    df, adj = generate_ushcn_data(data_file_path)
    '''df = joblib.load(data_file_path + "/raw_data/USHCN/Utensor.joblib") / 100


    adj = joblib.load(data_file_path + '/raw_data/USHCN/sim.joblib')'''
    df = df.reshape(1218,  120 * 12, 2)[:,:,0].astype(float)
    print(df[200:210,100:140])

    df = np.expand_dims(df,  axis=-1)
    print(df.shape, adj.shape)
    #df = np.load(data_file_path)
    data = df[..., target_channel].transpose(1, 0, 2)
    #data = np.expand_dims(df, axis=-1).transpose(1, 0, 2)
    print(data.shape, data.dtype, 'xxxxxxxxxxx')

    print("raw time series shape: {0}".format(data.shape))
    l, n, f = data.shape
    print(l,n,f,'kkkkkk')
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
    #data_norm = data
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, seqlen)
    #data_norm = np.ones_like(data)
    print("   hererere", train_index[-1], train_index[-1][1])
    feature_list = [data_norm, data_norm]

    node_index = np.array([i for i in range(n)])
    np.random.seed(seed)
    np.random.shuffle(node_index)

    if add_time_of_day:
        # numerical time_of_day
        timestamps_per_day = 24 * 12
        time_of_day_index = np.arange(l) % timestamps_per_day
        tod_tiled = np.tile(time_of_day_index, [1, n, 1]).transpose((2, 1, 0))
        print(tod_tiled)
        feature_list.append(tod_tiled.astype('float64'))

    if add_day_of_week:
        # numerical day_of_week
        days_per_week = 7
        timestamps_per_day = 24 * 12
        day_of_week_index = (np.arange(l) // timestamps_per_day) % days_per_week
        print(np.max(day_of_week_index))
        dow_tiled = np.tile(day_of_week_index, [1, n, 1]).transpose((2, 1, 0))

        feature_list.append(dow_tiled.astype('float64'))

    if add_month_of_year:
        months_per_year = 12

        months_of_year_index = np.arange(l) % months_per_year
        print(months_of_year_index)
        moy_tiled = np.tile(months_of_year_index, [1, n, 1]).transpose((2, 1, 0))
        print(moy_tiled)
        print(moy_tiled.shape)
        feature_list.append(moy_tiled)


    for i in feature_list:
        print('111')
        print(i.shape)
        print(add_month_of_year)
    print(len(feature_list))
    print(data.dtype)

    processed_data = np.concatenate(feature_list, axis=-1)

    print(processed_data.dtype)
    print(processed_data.shape,'xxxxxxxxxxxxxxxxxxxxxxxx')
    lon_lat = process_location(location_file_path)
    print(lon_lat)
    print(processed_data[0].dtype)
    print(processed_data[1].dtype)
    num_samples = n
    print(num_samples, 'numsamples')
    train_num_short = round(num_samples * train_ratio)
    valid_num_short = round(num_samples * valid_ratio)
    test_num_short = num_samples - train_num_short - valid_num_short

    print("number of training nodes samples:{0}".format(train_num_short))
    print("number of validation nodes samples:{0}".format(valid_num_short))
    print("number of test nodes samples:{0}".format(test_num_short))

    train_node = node_index[:train_num_short]
    valid_node = node_index[train_num_short:train_num_short + valid_num_short]
    test_node = node_index[train_num_short + valid_num_short:train_num_short + valid_num_short + test_num_short]
    print(train_node)
    data = {}

    #xdata = np.ones_like(processed_data)
    data["processed_data"] = processed_data

    '''with open(output_dir + "/data.pkl".format(str(int(train_series_ratio * 10))), "wb") as f:
        pickle.dump(data, f)
    adjindex = {}
    adjindex['train_nodes'] = train_node
    adjindex['valid_nodes'] = valid_node
    adjindex['test_nodes'] = test_node
    print(train_node.shape, valid_node.shape, test_node.shape)
    print(np.max(train_node), np.max(test_node), np.max(valid_node))
    #print(test_node)
    #print(valid_node)
    with open(output_dir + "/adj_index.pkl", "wb") as f:
        pickle.dump(adjindex, f)
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index.pkl", "wb") as f:
        pickle.dump(index, f)
    #adj = np.load(graph_file_path)
    print(adj.dtype)
    dic_adj = {}
    dic_adj['adj_mx'] = adj
    with open(output_dir + "/adj_mx.pkl", 'wb') as f:
        pickle.dump(adj, f)'''
    #shutil.copyfile(graph_file_path, output_dir + "/adj_mx.pkl")
    #print(adj.shape)
    #print(adj.dtype, train_node.dtype, processed_data.dtype)
    '''location = {}
    location["lon_lat"] = lon_lat
    with open(output_dir + "/location.pkl", 'wb') as f:
        pickle.dump(location, f)'''
    #a = np.load(graph_file_path)
    print(np.sort(train_node))
    print(np.sort(valid_node))
    print(np.sort(test_node))



if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    SEQ_LEN = 24
    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TRAIN_SERIES_RATIO = 0.7
    VALID_TEST_SPLIT = False
    VALID_SERIES_RATIO = 0.1
    SEED = 48
    TARGET_CHANNEL = [0]          # target channel(s)]

    DATASET_NAME = "USHCN"
    DATASET_NAME2 = "USHCN"
    TOD = False                 # if add time_of_day feature
    DOW = False
    MOY = True # if add day_of_week feature

    ORIGIN_DIR = "D:/myfile/ST-kriging"
    OUTPUT_DIR = ORIGIN_DIR + "/datasets/" + DATASET_NAME + "_{0}{1}_{2}".format(str(int(TRAIN_RATIO * 10)), str(int(VALID_RATIO * 10)), str(SEQ_LEN))
    DATA_FILE_PATH = ORIGIN_DIR + "/datasets"
    GRAPH_FILE_PATH = ORIGIN_DIR + "/datasets/raw_data/{0}/{1}_A.npy".format(DATASET_NAME, DATASET_NAME2)
    LOCATION_FILE_PATH = "D:/myfile/ST-kriging/datasets/raw_data/{0}/latlon.csv".format(DATASET_NAME, DATASET_NAME2)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--graph_file_path", type=str,
                        default=GRAPH_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--location_file_path", type=str,
                        default=LOCATION_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--seq_len", type=int,
                        default=SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--moy", type=bool, default=MOY,
                        help="Add feature month_of_year.")
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
