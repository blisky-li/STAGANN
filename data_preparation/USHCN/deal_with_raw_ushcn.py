
import os
import zipfile
import numpy as np
#import scipy.sparse as sp
import pandas as pd
from math import radians, cos, sin, asin, sqrt
#from sklearn.externals import joblib
import joblib
import pickle
#import scipy.io
import torch
from torch import nn
import scipy.sparse as sp

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
    pos = []
    Utensor = np.zeros((1218, 120, 12, 2)).astype('float64')
    Omissing = np.ones((1218, 120, 12, 2)).astype('float64')
    with open(ori+"raw_data/USHCN/Ulocation.txt", "r") as f:
        loc = 0
        for line in f.readlines():
            poname = line[0:11]
            pos.append(line[13:30])
            with open(ori+"raw_data/USHCN/ushcn.v2.5.5.20191231/" + poname + ".FLs.52j.prcp", "r") as fp:
                temp = 0
                for linep in fp.readlines():
                    if int(linep[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linep[17 + 9 * i:22 + 9 * i]
                            p_temp = int(str_temp)
                            if p_temp == -9999:
                                Omissing[loc, temp, i, 0] = 0
                            else:
                                Utensor[loc, temp, i, 0] = p_temp
                        temp = temp + 1
            with open(ori+"raw_data/USHCN/ushcn.v2.5.5.20191231/" + poname + ".FLs.52j.tavg", "r") as ft:
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

    latlon = np.loadtxt(ori+"raw_data/USHCN/latlon.csv", delimiter=",")
    print(latlon.shape)
    sim = np.zeros((1218, 1218)).astype('float64')

    for i in range(1218):
        for j in range(1218):
            sim[i, j] = float(haversine(latlon[i, 1], latlon[i, 0], latlon[j, 1], latlon[j, 0]))  # RBF

    print(Utensor.dtype, sim.dtype)
    #sim = np.exp(-sim / 10000 / 10)
    sigma = sim.std()
    adj = np.exp(-np.square(sim / sigma))
    adj = np.around(adj, decimals=4)
    adj[adj < 0.3] = 0.
    #adj[adj == 1] = 0.
    #adj[adj >= 0.5] = 1.
    adj2 =adj.view()
    print(adj2.dtype)
    #adj2 = sp.coo_matrix((adj2), dtype=np.int8).toarray()
    with open(ori + "raw_data/USHCN/adj_mx.pkl", 'wb') as f:
        pickle.dump(adj, f)
    np.save(ori+'raw_data/USHCN/USHCN_A.npy', adj2)

    Utensor = Utensor.astype(np.float64)
    Utensor = Utensor.reshape(1218,120*12, 2)
    #np.save('USHCN_A.npy', adj)
    np.save(ori+'raw_data/USHCN/USHCN_X.npy', Utensor)



def load_udata(ori):

    generate_ushcn_data(ori)


    '''print(A)
    print(A.shape)'''
    return None

ORIGIN_DIR = "D:/myfile/ST-kriging/datasets/"
load_udata(ORIGIN_DIR)