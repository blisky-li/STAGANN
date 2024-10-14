import os

import torch
from torch.utils.data import Dataset
import  numpy as np

from ..utils import load_pkl

def node_index_get(node_index_file_path: str, mode: str):
    all_nodes = load_pkl(node_index_file_path)
    assert mode in ["train", "valid", "test"], "error mode"
    if mode in "train_nodes":
        train_node = all_nodes['train_nodes']
        nodes = train_node
        return nodes
    elif mode in 'valid_nodes':
        train_node = all_nodes['train_nodes']
        valid_node = all_nodes['valid_nodes']
        nodes = np.hstack((train_node, valid_node))
        return nodes, train_node, valid_node
    elif mode in 'test_nodes':
        train_node = all_nodes['train_nodes']
        test_node = all_nodes['test_nodes']
        nodes = np.hstack((train_node, test_node))
        return nodes, train_node, test_node
    else:
        nodes = np.array([])
        return nodes


class STKrigingDataset(Dataset):
    """Spatio-Temporal Kriging Datasets """
    def __init__(self, data_file_path: str, index_file_path: str, node_index_file_path: str, mode: str) -> None:
        super().__init__()

        assert mode in["train",  "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path, node_index_file_path)
        self.nodes = load_pkl(node_index_file_path)
        self.nodes_index = self._node_index_get(mode)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        data = torch.from_numpy(processed_data).float()
        if self._check_NaN_INF(data):
            print("Origin DATA have NaN or INF")
            data = self._replace_NaN_INF(data)
        self.data = data[:,self.nodes_index]

        self.index = load_pkl(index_file_path)[mode]
        #print(self.index[0])



    def _node_index_get(self, mode:str):
        if mode in "train_nodes":
            train_node =  self.nodes['train_nodes']
            nodes = train_node
        elif mode in 'valid_nodes':
            train_node = self.nodes['train_nodes']
            valid_node = self.nodes['valid_nodes']
            nodes = np.hstack((train_node,valid_node))
            print(nodes)
        elif mode in 'test_nodes':
            train_node = self.nodes['train_nodes']
            test_node = self.nodes['test_nodes']
            nodes = np.hstack((train_node,test_node))

            print("nodes:" ,nodes.shape)
        else:
            nodes = np.array([])
        return nodes



    def _check_NaN_INF(self, data: torch.Tensor):
        """ check whether NaN or INF in time series """

        # nan
        nan = torch.any(torch.isnan(data))
        # inf
        inf = torch.any(torch.isinf(data))
        if (nan or inf):
            return True
        else:
            return False



    def _replace_NaN_INF(self,data):
        """ replace NaN and INF with zero  """
        data = torch.where(torch.isnan(data), torch.zeros_like(data), data)
        data = torch.where(torch.isinf(data), torch.zeros_like(data), data)
        return data


    def _check_if_file_exists(self, data_file_path: str, index_file_path: str, node_index_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        if not os.path.isfile(node_index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(node_index_file_path))

    def __getitem__(self, index) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])
        #print(idx)
        if len(idx) == 1:
            lst_idx = list(self.index).index(idx)
            return self.data[:,lst_idx,:]

        elif len(idx) == 2:
            return self.data[idx[0]:idx[1],:,:]

        else:
            assert len(idx) != 3, "Indexes that do not apply !!!"
            lst_idx = list(self.index).index(idx[2])
            return self.data[idx[0]:idx[1], lst_idx, :]

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """

        return len(self.index)



def read_location(path):
    print(path)
    print(load_pkl(path).keys())
    loc = load_pkl(path)["lon_lat"]
    node_index = np.array([i for i in range(loc.shape[0])])
    print(type(loc[:,1]))
    '''np.random.seed(42)
    np.random.shuffle(node_index)
    print(loc.shape)
    print(np.max(loc[node_index, 1]), np.min(loc[node_index, 0]))'''
    lat, lon = np.rint(((loc[:, 1]).astype(float)) + 180) * 1000, np.rint(((loc[:, 0]).astype(float)) + 90) * 1000
    print(np.max(lat), np.min(lon))
    binary_vector_lat = np.where(lat >= 0, 11, 12)
    binary_vector_lon = np.where(lon >= 0, 11, 12)
    lat_str = np.char.zfill(np.char.mod('%d', np.abs(lat)), 6)
    lon_str = np.char.zfill(np.char.mod('%d', np.abs(lon)), 6)
    print(lat_str.shape)
    print(lat_str)
    lat_str = np.array([list(num) for num in lat_str]).astype(int)
    lon_str = np.array([list(num) for num in lon_str]).astype(int)

    eos = np.full(lat_str.shape[0],13).reshape(-1, 1)
    binary_vector_lat = binary_vector_lat.reshape(-1, 1)
    binary_vector_lon = binary_vector_lon.reshape(-1, 1)
    #print(binary_vector_lon.shape, binary_vector_lat.shape, eos.shape, lon_str.shape, lat_str.shape)
    #vector = np.concatenate((binary_vector_lat, lat_str, eos, binary_vector_lon, lon_str), axis=1)
    vector = torch.tensor(np.concatenate((lat_str, lon_str), axis=1))
    lst = []
    num = 0
    dic = {}
    '''for i in range(lat_str.shape[0]):
        loc_id = lat_str[i] + lon_str[i]
        if loc_id not in dic:
            dic[loc_id] = num
            lst.append(num)
            num += 1
        else:
            n = dic[loc_id]
            lst.append(n)
    lst = torch.tensor(lst).unsqueeze(-1)
    print(torch.max(lst))
    print(lst.shape)'''
    #vector = torch.tensor(vector)
    #print(vector.shape)

    loc_id = torch.tensor(load_pkl(path)["labels"]).unsqueeze(-1)
    return [vector, loc_id]


'''if __name__ == '__main__':
    ORIGIN_DIR = "D:/myfile/ST-kriging"
    dataset = 'METR-LA'
    origin_path = ORIGIN_DIR + '/datasets/' + dataset + '_71_24/'
    data_file_path = origin_path + 'data.pkl'
    index_file_path = origin_path + 'index.pkl'
    node_index_path = origin_path + 'adj_index.pkl'
    mode1 = 'train' # "train",  "valid", "test"
    stkriging = STKrigingDataset(data_file_path,index_file_path,node_index_path,mode1)
    data1 = stkriging.data
    index1 = stkriging.index
    node_index1 = stkriging.nodes_index
    print(data1.shape)

    print(len(index1))
    print(node_index1.shape)
    print(node_index1)
    mode1 = 'valid'  # "train",  "valid", "test"
    stkriging = STKrigingDataset(data_file_path, index_file_path, node_index_path, mode1)
    data2 = stkriging.data
    index2 = stkriging.index
    node_index2 = stkriging.nodes_index
    print(data2.shape)
    print(len(index2))
    print(node_index2.shape)
    print(node_index2)
    mode1 = 'test'  # "train",  "valid", "test"
    stkriging = STKrigingDataset(data_file_path, index_file_path, node_index_path, mode1)
    data3 = stkriging.data
    index3 = stkriging.index
    node_index3 = stkriging.nodes_index
    print(data3.shape)
    print(len(index3))
    print(node_index3.shape)
    print(node_index3)'''




