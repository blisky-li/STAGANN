import os.path
import pickle

import torch
import numpy as np
import sys
import os
from .adjacent_matrix_norm import calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian, calculate_symmetric_message_passing_adj, calculate_transition_matrix, calculate_random_walk_matrix
from .pkl_actions import load_pkl, dump_pkl

def load_adj(file_path: str,index_path: str):
    _check_if_file_exists(file_path, index_path)
    try:
        # METR and PEMS_BAY
        _, _, adj_mx = load_pkl(file_path)
    except ValueError:
        # PEMS04
        adj_mx = load_pkl(file_path)
    train_index = load_pkl(index_path)["train_nodes"]
    valid_index = load_pkl(index_path)["valid_nodes"]
    test_index = load_pkl(index_path)["test_nodes"]

    return adj_mx, train_index, valid_index, test_index

def adj_transform(adj_mx, adj_type: str):
    """load adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """
    #print(os.path.abspath(file_path))
    #print(adj_type)
    #adj_mx = adj_node_index(adj_mx,adj_index)
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'random_walk':
        adj = [calculate_random_walk_matrix(adj_mx)]
    elif adj_type == "original":
        adj = [adj_mx]
    else:
        error = 0
        assert error, "adj type not defined"
    print(adj_type, np.sum(adj), np.sum(adj_mx),'kkkkkkk')
    return adj, adj_mx

def adj_node_index(adj,index):
    """ Reorder the elements within the adjacency matrix according to the given new indexes  """
    adp = adj[index,:]
    adp2 = adp[:,index]
    return adp2


def transfer_index(index):
    """ Transfer the index (n < N) to index (n)  """
    l = np.array([])
    for index,value in np.ndenumerate(index):
        l = np.append(l,index)
    return l

def restore_index(index):
    """  Reorder indexes according to their size  """
    #transfer_idx = transfer_index(index)
    dic = {}
    for idx,value in np.ndenumerate(index):
        dic[value] = int(idx[0])
    restorm_index = np.sort(index)
    l = np.array([])
    for index, value in np.ndenumerate(restorm_index):
        l = np.append(l, dic[value])
    return l.astype('int64')


def restore_matrix(adj,restore_idx):
    """ Reorder the adjacency matrix according to smallest-to-largest indexes  """
    rt_idx = restore_index(restore_idx)
    adj2 = adj_node_index(adj,rt_idx)
    #print(adj)
    return adj2


def _check_if_file_exists(data_file_path: str, index_file_path: str):
    """Check if data file and index file exist.

            Args:
                data_file_path (str): data file path
                index_file_path (str): index file path

            Raises:
                FileNotFoundError: no data file
                FileNotFoundError: no index file
            """
    #print(os.path.abspath(data_file_path))
    if not os.path.isfile(data_file_path):
        raise FileNotFoundError("STkriging can not find data file {0}".format(data_file_path))
    if not os.path.isfile(index_file_path):
        raise FileNotFoundError("STkriging can not find index file {0}".format(index_file_path))

def adj_mask_unknown_node(adj, unknown_idx):
    if len(adj.shape) == 2:
        unknown_idx_adj = torch.LongTensor(unknown_idx).repeat(adj.shape[0],1)
        adj = adj.scatter(1,unknown_idx_adj,0)
        return adj
    elif len(adj.shape) == 3:
        l = unknown_idx.shape[1]
        unknown_idx_adj = torch.LongTensor(unknown_idx).repeat(1,adj.shape[1]).reshape(adj.shape[0],adj.shape[1],l)
        adj = adj.scatter(2, unknown_idx_adj, 0)
        return adj
    else:
        return adj
