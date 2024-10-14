import math
import functools
from typing import Tuple, Union, Optional
import pandas as pd
import torch
import numpy as np
from easytorch.utils.dist import master_only

from .base_runner import BaseRunner
from ..data import SCALER_REGISTRY
from ..utils import load_pkl, load_adj, adj_node_index, adj_transform, adj_mask_unknown_node
from ..metrics import masked_mae, masked_mape, masked_rmse, R2, m_mae, masked_binary
from ..tools import draw_graph, draw_plot, similarity

class BaseSpatiotemporalKrigingRunner(BaseRunner):
    """
    Runner for short term multivariate time series forecasting datasets.
    Typically, models predict the future 12 time steps based on historical time series.
    Features:
        - Evaluate at horizon 3, 6, 12, and overall.
        - Metrics: MAE, RMSE, MAPE. The best model is the one with the smallest mae at validation.
        - Loss: MAE (masked_mae). Allow customization.
        - Support curriculum learning.
        - Users only need to implement the `forward` function.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.dataset_name = cfg["DATASET_NAME"]
        # different datasets have different null_values, e.g., 0.0 or np.nan.
        self.null_val = cfg["TRAIN"].get("NULL_VAL", np.nan)    # consist with metric functions
        self.dataset_type = cfg["DATASET_TYPE"]
        self.evaluate_on_gpu = cfg["TEST"].get("USE_GPU", True)     # evaluate on gpu or cpu (gpu is faster but may cause OOM)
        self.batch_select_node_random = cfg["TRAIN"].get("BATCH_SELECT_NODE_RANDOM", False)
        self.batch_matrix_random = cfg["TRAIN"].get("BATCH_MATRIX_RANDOM", False)
        self.mask_matrix = cfg["TRAIN"].get("Mask_Matrix", False)

        self.double_loss = cfg["TRAIN"].get("DOUBLE_LOSS", True)
        self.location = cfg["DATASET"].get("LOCATION", [[None]])
        self.matrix_transform = cfg["DATASET"].get("MATRIX_TRANSFORM", "original")

        self.seqlen = cfg["DATASET_LEN"]
        #self.model = cfg["MODEL"]

        # read scaler for re-normalization
        if cfg["DATASET"]["TRANSFORM"] == "standard_transform":
            self.scaler = load_pkl(
                "{0}/scaler.pkl".format(cfg["TRAIN"]["DATA"]["DIR"]))
        elif cfg["DATASET"]["TRANSFORM"] == "min_max_transform":
            self.scaler = load_pkl(
                "{0}/scaler2.pkl".format(cfg["TRAIN"]["DATA"]["DIR"]))
        elif cfg["DATASET"]["TRANSFORM"] == "logarithm_standard_transform":
            self.scaler = load_pkl(
                "{0}/scaler3.pkl".format(cfg["TRAIN"]["DATA"]["DIR"]))
        else:
            self.scaler = load_pkl(
                "{0}/scaler4.pkl".format(cfg["TRAIN"]["DATA"]["DIR"]))
            # 一种预处理方法的选择

        self.data_length = cfg["DATASET_LEN"]

        self.trainsetratio = cfg["DATASET_TRAINRATIO"]
        self.valsetratio = cfg["DATASET_VALRATIO"]
        self.train_ratio = cfg["TRAIN"]["RATIO"]
        self.adj_mx = cfg["DATASET"]["MATRIX"]
        self.train_index = cfg["DATASET"]["TRAININDEX"]
        self.valid_index = cfg["DATASET"]["VALIDINDEX"]
        self.test_index = cfg["DATASET"]["TESTINDEX"]


        self.train_mx,_ = adj_transform(adj_node_index(self.adj_mx, self.train_index), self.matrix_transform)

        self.valid_all_index = torch.cat((torch.tensor(self.train_index), torch.tensor(self.valid_index)), 0)
        self.test_all_index = torch.cat((torch.tensor(self.train_index), torch.tensor(self.test_index)), 0)

        self.valid_mx, _ = adj_transform(adj_node_index(self.adj_mx, self.valid_all_index), self.matrix_transform)
        self.test_mx, _ = adj_transform(adj_node_index(self.adj_mx, self.test_all_index), self.matrix_transform)

        # define loss
        self.loss = cfg["TRAIN"]["LOSS"]
        print(self.loss)
        # define metric
        self.metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape, "R2":R2, 'ACC': masked_binary}
        # curriculum learning for output. Note that this is different from the CL in Seq2Seq archs.
        self.cl_param = cfg.TRAIN.get("CL", None)
        if self.cl_param is not None:
            self.warm_up_epochs = cfg.TRAIN.CL.get("WARM_EPOCHS", 0)
            self.cl_epochs = cfg.TRAIN.CL.get("CL_EPOCHS")
            self.prediction_length = cfg["DATASET_LEN"]
            self.cl_step_size = cfg.TRAIN.CL.get("STEP_SIZE", 1)
        # evaluation horizon
        self.evaluation_horizons = [_ - 1 for _ in cfg["TEST"].get("EVALUATION_HORIZONS", range(1, self.seqlen + 1))]
        assert min(self.evaluation_horizons) >= 0, "The horizon should start counting from 0."

    def init_training(self, cfg: dict):
        """Initialize training.

        Including loss, training meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_training(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("train_"+key, "train", "{:.4f}")

    def init_validation(self, cfg: dict):
        """Initialize validation.

        Including validation meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_validation(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("val_"+key, "val", "{:.4f}")

    def init_test(self, cfg: dict):
        """Initialize test.

        Including test meters, etc.

        Args:
            cfg (dict): config
        """

        super().init_test(cfg)
        for key, _ in self.metrics.items():
            self.register_epoch_meter("test_"+key, "test", "{:.4f}")

    def build_train_dataset(self, cfg: dict):
        """Build MNIST train dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """
        if cfg["DATASET"]["TRANSFORM"] == "standard_transform":
            data_file_path = "{0}/data.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        elif cfg["DATASET"]["TRANSFORM"] == "min_max_transform":
            data_file_path = "{0}/data2.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        elif cfg["DATASET"]["TRANSFORM"] == "logarithm_standard_transform":
            data_file_path = "{0}/data3.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        else:
            data_file_path = "{0}/data4.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        index_file_path = "{0}/index.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        node_file_path = "{0}/adj_index.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, mode (train, valid, or test) and corresponding node_index file path
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["node_index_file_path"] = node_file_path
        dataset_args["mode"] = "train"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("train len: {0}".format(len(dataset)))

        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            validation dataset (Dataset)
        """
        if cfg["DATASET"]["TRANSFORM"] == "standard_transform":
            data_file_path = "{0}/data.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        elif cfg["DATASET"]["TRANSFORM"] == "min_max_transform":
            data_file_path = "{0}/data2.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        elif cfg["DATASET"]["TRANSFORM"] == "logarithm_standard_transform":
            data_file_path = "{0}/data3.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        else:
            data_file_path = "{0}/data4.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        index_file_path = "{0}/index.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        node_file_path = "{0}/adj_index.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["node_index_file_path"] = node_file_path
        dataset_args["mode"] = "valid"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("val len: {0}".format(len(dataset)))

        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        """Build MNIST val dataset

        Args:
            cfg (dict): config

        Returns:
            train dataset (Dataset)
        """

        if cfg["DATASET"]["TRANSFORM"] == "standard_transform":
            data_file_path = "{0}/data.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        elif cfg["DATASET"]["TRANSFORM"] == "min_max_transform":
            data_file_path = "{0}/data2.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        elif cfg["DATASET"]["TRANSFORM"] == "logarithm_standard_transform":
            data_file_path = "{0}/data3.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        else:
            data_file_path = "{0}/data4.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])

        index_file_path = "{0}/index.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])
        node_file_path = "{0}/adj_index.pkl".format(cfg["TRAIN"]["DATA"]["DIR"])

        # build dataset args
        dataset_args = cfg.get("DATASET_ARGS", {})
        # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
        dataset_args["data_file_path"] = data_file_path
        dataset_args["index_file_path"] = index_file_path
        dataset_args["node_index_file_path"] = node_file_path
        dataset_args["mode"] = "test"

        dataset = cfg["DATASET_CLS"](**dataset_args)
        print("test len: {0}".format(len(dataset)))

        return dataset

    def curriculum_learning(self, epoch: int = None) -> int:
        """Calculate task level in curriculum learning.

        Args:
            epoch (int, optional): current epoch if in training process, else None. Defaults to None.

        Returns:
            int: task level
        """

        if epoch is None:
            return self.prediction_length
        epoch -= 1
        # generate curriculum length
        if epoch < self.warm_up_epochs:
            # still warm up
            cl_length = self.prediction_length
        else:
            _ = ((epoch - self.warm_up_epochs) // self.cl_epochs + 1) * self.cl_step_size
            cl_length = min(_, self.prediction_length)
        return cl_length

    def forward(self, data: torch.Tensor, adj, unknown_nodes, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value). [B, L, N, C] for each of them.
        """

        raise NotImplementedError()

    def metric_forward(self, metric_func, args):
        """Computing metrics.

        Args:
            metric_func (function, functools.partial): metric function.
            args (list): arguments for metrics computation.
        """

        if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
            # support partial(metric_func, null_val = something)
            metric_item = metric_func(*args)
        elif callable(metric_func):
            # is a function
            metric_item = metric_func(*args, null_val=self.null_val)
        else:
            raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
        return metric_item

    def train_iters(self, epoch: int, iter_index: int, data: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Training details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            epoch (int): current epoch.
            iter_index (int): current iter.

        Returns:
            loss (torch.Tensor)
        """
        #print(self.location)
        #print(data.device,'11111')
        self.optim.zero_grad()
        if torch.is_tensor(self.location[0]):
            loc_grids, loc_id = self.location

            locg = loc_grids[self.train_index, :]
            location = loc_id[self.train_index, :]

        else:
            locg = None
            location = None
        #print(location)
        iter_num = (epoch-1) * self.iter_per_epoch + iter_index
        #print("******",iter_num, data.shape)
        masked_node_num = int(self.train_index.size * (1 - self.train_ratio))
        #print(self.train_index.size, masked_node_num)
        if self.batch_select_node_random:
            if not self.batch_matrix_random:
                train_mx = self.train_mx[0]
                #print(train_mx.shape)
                #print('1-1-1')
                index = np.random.permutation(self.train_index.size)

                train_mx = adj_node_index(train_mx, index)

                data2 = data[:, :, index, :]
                if torch.is_tensor(self.location[0]):
                    locg = locg[index,:]
                    location = location[index,:]

                data_ones = torch.ones_like(data2)
                real_value = torch.ones((data_ones.shape[0], data_ones.shape[1], masked_node_num))
                #print(real_value.shape)
                unknown_nodes = torch.zeros(data_ones.shape[0], data_ones.shape[2])

                random_recoder = []
                matrix_lst = []
                locg_lst = []
                location_lst = []
                for i in range(data_ones.shape[0]):
                    random_select_node = np.random.choice(index, masked_node_num, replace=False)
                    if torch.is_tensor(self.location[0]):
                        locg_lst.append(locg)
                        location_lst.append(location)
                    data_ones[i, :, random_select_node, 0] = 0
                    real_value[i] = data2[i, :, random_select_node, 0]
                    unknown_nodes[i, random_select_node] = 1
                    random_recoder.append(random_select_node)
                    matrix_lst.append(torch.FloatTensor(train_mx))
                #print(11111111)
                matrix = torch.stack(matrix_lst, dim=0)

                if torch.is_tensor(self.location[0]):
                    loc_g = torch.stack(locg_lst, dim=0)
                    loc = torch.stack(location_lst, dim=0)
                input_value = data2 * data_ones
                batch_random_nodes = np.stack(random_recoder, axis=0)
                # print(batch_random_nodes.shape)
                

                if self.mask_matrix:
                    matrix_mask = adj_mask_unknown_node(matrix, batch_random_nodes)
                    if torch.is_tensor(self.location[0]):
                        predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask], unknown_nodes = batch_random_nodes, location=[loc_g, loc], epoch=epoch,
                                           iter_num=iter_num, train=True)
                    else:
                        predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask],
                                                   unknown_nodes=batch_random_nodes,  epoch=epoch,
                                                   iter_num=iter_num, train=True)
                else:
                    if torch.is_tensor(self.location[0]):
                        predict_out = self.forward(data=input_value, adj=matrix, unknown_nodes=batch_random_nodes, location=[loc_g, loc],
                                                   epoch=epoch, iter_num=iter_num, train=True)
                    else:

                        predict_out = self.forward(data=input_value, adj=matrix, unknown_nodes = batch_random_nodes, epoch=epoch, iter_num=iter_num, train=True)
                # print(predict[torch.arange(data.shape[0])[:,None],:,batch_random_nodes].transpose(1,2)[0,:,1])
                # print(predict[:,:,batch_random_nodes[0]][0,:,1])
                predict = predict_out[0]
                forward_return = [predict[torch.arange(data.shape[0])[:, None], :, batch_random_nodes].transpose(1, 2),
                                  real_value.to(predict.device)]
            else:
                #print('1-2-2')
                train_mx = self.train_mx[0]


                data_ones = torch.ones_like(data)

                random_recoder = []
                matrix_lst = []
                real_value_lst = []
                locg_lst = []
                location_lst = []
                unknown_nodes = torch.zeros(data_ones.shape[0], data_ones.shape[2])
                for i in range(data_ones.shape[0]):
                    index = np.random.permutation(self.train_index.size)
                    batch_train_mx = adj_node_index(train_mx, index)
                    if torch.is_tensor(self.location[0]):
                        location_2 = location[index, :]
                        location_lst.append(location_2)
                        locg_lst.append(locg[index, :])
                    data[i] = data[i, :, index, :]
                    random_select_node = np.random.choice(index, masked_node_num, replace=False)

                    data_ones[i, :, random_select_node, 0] = 0
                    matrix_lst.append(torch.tensor(batch_train_mx))
                    unknown_nodes[i, random_select_node] = 1
                    real_value_lst.append(data[i, :, random_select_node, 0])
                    random_recoder.append(random_select_node)
                matrix = torch.stack(matrix_lst, dim=0)
                loc_g = torch.stack(locg_lst, dim=0)
                loc = torch.stack(location_lst, dim=0)
                real_value = torch.stack(real_value_lst, dim=0)
                input_value = data * data_ones
                batch_random_nodes = np.stack(random_recoder, axis=0)
                if self.mask_matrix:
                    matrix_mask = adj_mask_unknown_node(matrix, batch_random_nodes)
                    if torch.is_tensor(self.location[0]):
                        predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask],
                                                   unknown_nodes=batch_random_nodes, location=[loc_g, loc], epoch=epoch,
                                                   iter_num=iter_num, train=True)
                    else:
                        predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask],unknown_nodes = batch_random_nodes, epoch=epoch,
                                       iter_num=iter_num, train=True)
                else:
                    if torch.is_tensor(self.location[0]):
                        predict_out = self.forward(data=input_value, adj=matrix, unknown_nodes=batch_random_nodes, location=[loc_g, loc],
                                                   epoch=epoch, iter_num=iter_num, train=True)
                    else:

                        predict_out = self.forward(data=input_value, adj= matrix,unknown_nodes = batch_random_nodes, epoch=epoch, iter_num=iter_num, train=True)
                predict = predict_out[0]
                forward_return = [predict[torch.arange(data.shape[0])[:, None], :, batch_random_nodes].transpose(1, 2),
                                  real_value.to(predict.device)]

        else:
            #print('1-3-3')
            train_mx = self.train_mx[0]
            index = np.random.permutation(self.train_index.size)
            #print(data.shape, self.train_index, index)
            if torch.is_tensor(self.location[0]):
                locg = locg[index, :]
                location = location[index, :]
                N, L = locg.shape
                loc_g = locg.unsqueeze(0).expand(data.shape[0], N, L)

                N,L = location.shape
                loc = location.unsqueeze(0).expand(data.shape[0], N, L)
            else:
                loc_g = None
                loc = None
            train_mx = adj_node_index(train_mx, index)



            data2 = data[:, :, index, :]
            unknown_nodes = torch.zeros(data.shape[0], data.shape[2])
            random_select_node = np.random.choice(index, masked_node_num, replace=False)
            unknown_nodes[:, random_select_node] = 1
            real_value = data2[:, :, random_select_node, :]

            input_value = data2
            input_value[:, :, random_select_node, 0] = 0


            if self.mask_matrix:
                matrix_mask = adj_mask_unknown_node(torch.tensor(train_mx), random_select_node)
                matrix = torch.tensor(train_mx)
                if torch.is_tensor(self.location[0]):
                    predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask],
                                               unknown_nodes=random_select_node,location= [loc_g, loc], epoch=epoch, iter_num=iter_num,
                                               train=True)
                else:
                    predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask],unknown_nodes = random_select_node, epoch=epoch, iter_num=iter_num, train=True)
            else:

                matrix = torch.tensor(train_mx)
                if torch.is_tensor(self.location[0]):
                    #print(loc)
                    predict_out = self.forward(data=input_value, adj=matrix, unknown_nodes=random_select_node, location=[loc_g, loc],
                                               epoch=epoch, iter_num=iter_num, train=True)
                else:
                    predict_out = self.forward(data=input_value, adj= matrix, unknown_nodes = random_select_node, epoch=epoch, iter_num=iter_num, train=True)
            predict = predict_out[0]

            forward_return = [predict[:, :, random_select_node], real_value[:, :, :, 0].to(predict.device)]

        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        '''pred = torch.mean(torch.mean(prediction_rescaled, dim=2), dim=0)
        real = torch.mean(torch.mean(real_value_rescaled, dim=2), dim=0)'''

        df = pd.DataFrame(prediction_rescaled[0].detach().cpu().numpy())
        df.to_csv('output0.csv', index=False)

        df2 = pd.DataFrame(real_value_rescaled[0].detach().cpu().numpy())
        df2.to_csv('output0r.csv', index=False)
        # loss
        if self.cl_param:
            cl_length = self.curriculum_learning(epoch=epoch)
            forward_return[0] = prediction_rescaled[:, :cl_length, :, :]
            forward_return[1] = real_value_rescaled[:, :cl_length, :, :]
        else:
            forward_return[0] = prediction_rescaled
            forward_return[1] = real_value_rescaled






        if self.double_loss:
            prediction_all_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(predict, **self.scaler["args"])
            #print(len(predict_out))
            loss = 0
            for loss_func_name, loss_func in self.loss.items():
                if loss_func_name == 'masked_mae' or loss_func_name == 'cos':
                #if loss_func_name == 'masked_mae':
                    loss += self.metric_forward(loss_func, [forward_return[0],forward_return[1]])
                elif loss_func_name == 'rsd':
                    loss += self.metric_forward(loss_func, [prediction_rescaled, real_value_rescaled])
                elif loss_func_name ==  "masked_binary" and len(predict_out) > 1 and epoch <= 5:
                    node_predict = predict_out[1]

                    loss += self.metric_forward(loss_func, [node_predict, unknown_nodes.to(node_predict.device)])
                    #print(loss.item())
                '''else:
                    if len(data.shape) == 4:
                        data = data[:, :, :, 0].to(predict.device)
                    real_value_all_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(data, **self.scaler["args"])
                    loss += self.metric_forward(loss_func, [forward_return[0], real_value_all_rescaled])'''
        else:
            loss = self.metric_forward(self.loss, forward_return)

        l1_regularization = torch.tensor(0.).to(loss.device)

        for param in self.model.parameters():

            l1_regularization += torch.norm(param, 1)
        #print(l1_regularization* 0.0001)
        #loss += l1_regularization * 0.0001
        # metrics
        for metric_name, metric_func in self.metrics.items():
            if metric_name != 'ACC':
                metric_item = self.metric_forward(metric_func, forward_return[:2])

                self.update_epoch_meter("train_" + metric_name, metric_item.item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation details.

        Args:
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader
            train_epoch (int): current epoch if in training process. Else None.
            iter_index (int): current iter.
        """
        # print(data.shape)

        if torch.is_tensor(self.location[0]):
            loc_grids, loc_id = self.location
            #print(location.shape, self.valid_all_index.shape)
            #print(self.valid_all_index)
            locg = loc_grids[self.valid_all_index.type(torch.LongTensor),:]
            location = loc_id[self.valid_all_index.type(torch.LongTensor),:]

        else:
            locg = None
            location = None
        if self.batch_matrix_random: # self.batch_select_node_random  self.batch_matrix_random
            valid_mx = self.valid_mx[0]
            #print('22-1-1')
            data_ones = torch.ones_like(data)

            random_recoder = []
            matrix_lst = []
            real_value_lst = []
            locg_lst = []
            location_lst = []
            for i in range(data_ones.shape[0]):
                index = np.random.permutation(self.valid_all_index.shape[0])
                if torch.is_tensor(self.location[0]):
                    locg_lst.append(locg[index,:])
                    loc2 = location[index,:]
                    location_lst.append(loc2)
                real_index = torch.tensor(np.where(index >= self.train_index.size))
                batch_valid_mx = adj_node_index(valid_mx, index)

                data[i] = data[i, :, index, :]


                data_ones[i, :, real_index, 0] = 0
                matrix_lst.append(torch.tensor(batch_valid_mx))

                real_value_lst.append(data[i, :, real_index, 0])
                random_recoder.append(real_index)
            matrix = torch.stack(matrix_lst, dim=0)
            loc_g = torch.stack(locg_lst, dim=0)
            loc = torch.stack(location_lst, dim=0)
            real_value = torch.stack(real_value_lst, dim=0)
            input_value = data * data_ones
            batch_random_nodes = np.stack(random_recoder, axis=0)
            if self.mask_matrix:
                matrix_mask = adj_mask_unknown_node(matrix, batch_random_nodes)
                if torch.is_tensor(self.location[0]):
                    predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask],
                                               unknown_nodes=batch_random_nodes, location=[loc_g, loc], epoch=1, iter_num=None, train=False)
                else:
                    predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask], unknown_nodes = batch_random_nodes, epoch=1, iter_num=None, train=False)
            else:
                if torch.is_tensor(self.location[0]):
                    predict_out = self.forward(data=input_value, adj=matrix, unknown_nodes=batch_random_nodes, location = [loc_g, loc],
                                               epoch=1, iter_num=None, train=False)
                else:
                    predict_out = self.forward(data=input_value, adj=matrix, unknown_nodes = batch_random_nodes, epoch=1, iter_num=None, train=False)
            predict = predict_out[0]
            batch_random_nodes = np.squeeze(batch_random_nodes,axis=-2)
            real_value = torch.squeeze(real_value,dim=-2)

            forward_return = [predict[torch.arange(data.shape[0])[:, None], :, batch_random_nodes].transpose(1, 2),
                              real_value.to(predict.device)]


        else:
            #print('2-2-2')
            valid_mx = self.valid_mx[0]

            index = np.random.permutation(self.valid_all_index.shape[0])
            if torch.is_tensor(self.location[0]):
                locg = locg[index, :]
                location = location[index, :]
                N, L = locg.shape
                loc_g = locg.unsqueeze(0).expand(data.shape[0], N, L)

                N, L = location.shape
                loc = location.unsqueeze(0).expand(data.shape[0], N, L)

            else:
                loc_g = None
                loc = None
            real_index = torch.tensor(np.where(index >= self.train_index.size))

            valid_mx = adj_node_index(valid_mx, index)

            input_data = data
            # input_data[:,:,-self.valid_index.size:,0] = 0
            input_data = input_data[:, :, index, :]

            real_value = input_data[:, :, real_index, 0]
            input_data[:, :, real_index, 0] = 0
            if self.mask_matrix:
                matrix_mask = adj_mask_unknown_node(torch.tensor(valid_mx), real_index)
                matrix = torch.tensor(valid_mx)
                if torch.is_tensor(self.location[0]):
                    predict_out = self.forward(data=input_data, adj=[matrix, matrix_mask], unknown_nodes=real_index, location = [loc_g, loc],
                                               epoch=1, iter_num=None, train=False)
                else:
                    predict_out = self.forward(data=input_data, adj=[matrix,matrix_mask], unknown_nodes = real_index, epoch=1, iter_num=None, train=False)
            else:
                matrix = torch.tensor(valid_mx)
                if torch.is_tensor(self.location[0]):
                    predict_out = self.forward(data=input_data, adj=matrix, unknown_nodes=real_index, location = [loc_g, loc], epoch=1,
                                               iter_num=None, train=False)
                else:
                    predict_out = self.forward(data=input_data, adj=matrix, unknown_nodes = real_index, epoch=1, iter_num=None, train=False)
            predict = predict_out[0]
            forward_return = [predict[:, :, real_index], real_value.to(predict.device)]
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
        # metrics
        for metric_name, metric_func in self.metrics.items():
            if metric_name != 'ACC':
                metric_item = self.metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
                self.update_epoch_meter("val_" + metric_name, metric_item.item())

    @torch.no_grad()
    @master_only
    def test(self):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """

        # test loop

        prediction = []
        p = []
        r = []
        real_value = []
        class_error = []
        if torch.is_tensor(self.location[0]):
            loc_grids, loc_id = self.location
            locg = loc_grids[self.test_all_index.type(torch.LongTensor),:]
            location = loc_id[self.test_all_index.type(torch.LongTensor),:]
        else:
            locg = None
            location = None
        for _, data in enumerate(self.test_data_loader):
            # print(data.shape,torch.tensor(self.test_index).shape)
            if self.batch_matrix_random: # self.batch_select_node_random  self.batch_matrix_random
                test_mx = self.test_mx[0]
                #print('3-1-1-1')
                data_ones = torch.ones_like(data)
                data_classes = torch.zeros([data.shape[0], data.shape[2]]).to(data.device)
                random_recoder = []
                matrix_lst = []
                real_value_lst = []
                locg_lst = []
                location_lst = []
                for i in range(data_ones.shape[0]):
                    index = np.random.permutation(self.test_all_index.shape[0])
                    real_index = torch.tensor(np.where(index >= self.train_index.size))
                    batch_test_mx = adj_node_index(test_mx, index)
                    if torch.is_tensor(self.location[0]):
                        locg_lst.append(locg[index, :])
                        loc2 = location[index, :]
                        location_lst.append(loc2)
                    data[i] = data[i, :, index, :]

                    data_ones[i,:, real_index, 0] = 0

                    data_classes[i, real_index] = 1
                    matrix_lst.append(torch.tensor(batch_test_mx))

                    real_value_lst.append(data[i, :, real_index, 0])
                    random_recoder.append(real_index)
                matrix = torch.stack(matrix_lst, dim=0)
                loc_g = torch.stack(locg_lst, dim=0)
                loc = torch.stack(location_lst, dim=0)
                real = torch.stack(real_value_lst, dim=0)
                input_value = data * data_ones
                batch_random_nodes = np.stack(random_recoder, axis=0)
                if self.mask_matrix:
                    matrix_mask = adj_mask_unknown_node(matrix, batch_random_nodes)
                    if torch.is_tensor(self.location[0]):
                        predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask],
                                                   unknown_nodes=batch_random_nodes,location = [loc_g, loc], epoch=1, iter_num=None,
                                                   train=False)
                    else:
                        predict_out = self.forward(data=input_value, adj=[matrix, matrix_mask], unknown_nodes = batch_random_nodes, epoch=1, iter_num=None,
                                           train=False)
                else:
                    if torch.is_tensor(self.location[0]):
                        predict_out = self.forward(data=input_value, adj=matrix, unknown_nodes=batch_random_nodes,location = [loc_g, loc],
                                                   epoch=1, iter_num=None, train=False)
                    else:
                        predict_out = self.forward(data=input_value, adj=matrix, unknown_nodes = batch_random_nodes, epoch=1, iter_num=None, train=False)
                predict = predict_out[0]
                batch_random_nodes = np.squeeze(batch_random_nodes, axis=-2)
                real = torch.squeeze(real, dim=-2)
                if len(predict_out) > 1:
                    predict_classed = predict_out[1]
                    metric_item = self.metric_forward(masked_binary, [predict_classed.detach().cpu(), data_classes.detach().cpu(), False])
                    class_error.append(metric_item.item())
                prediction.append(predict[torch.arange(data.shape[0])[:, None], :, batch_random_nodes].transpose(1, 2))  # preds = forward_return[0]
                real_value.append(real.to(predict.device))  # testy = forward_return[1]

            else:
                #print('3-2-2-2')
                test_mx = self.test_mx[0]
                index = np.random.permutation(self.test_all_index.shape[0])
                real_index = torch.tensor(np.where(index >= self.train_index.size))
                if torch.is_tensor(self.location[0]):
                    locg = locg[index, :]
                    location = location[index, :]
                    N, L = locg.shape
                    loc_g = locg.unsqueeze(0).expand(data.shape[0], N, L)

                    N, L = location.shape
                    loc = location.unsqueeze(0).expand(data.shape[0], N, L)

                else:
                    loc_g = None
                    loc = None
                test_mx = adj_node_index(test_mx, index)



                input_data = data
                data_classes = torch.zeros([data.shape[0],  data.shape[2]]).to(data.device)
                input_data = input_data[:, :, index, :]

                real = input_data[:, :, real_index, 0]
                data_classes[:,  real_index] = 1
                input_data[:, :, real_index, 0] = 0
                if self.mask_matrix:
                    matrix_mask = adj_mask_unknown_node(torch.tensor(test_mx), real_index)
                    matrix = torch.tensor(test_mx)
                    if torch.is_tensor(self.location[0]):
                        predict_out = self.forward(data=input_data, adj=[matrix, matrix_mask], unknown_nodes=real_index, location=[loc_g, loc],
                                                   epoch=1, iter_num=None,
                                                   train=False)
                    else:
                        predict_out = self.forward(data=input_data, adj=[matrix,matrix_mask], unknown_nodes = real_index, epoch=1, iter_num=None,
                                           train=False)
                else:
                    matrix = torch.tensor(test_mx)
                    if torch.is_tensor(self.location[0]):
                        predict_out = self.forward(data=input_data, adj=matrix, unknown_nodes=real_index, location=[loc_g, loc],epoch=1,
                                                   iter_num=None,
                                                   train=False)
                    else:
                        predict_out = self.forward(data=input_data, adj=matrix, unknown_nodes = real_index, epoch=1, iter_num=None,
                                       train=False)

                predict = predict_out[0]
                if len(predict_out) > 1:
                    predict_classed = predict_out[1]
                    metric_item = self.metric_forward(masked_binary, [predict_classed.detach().cpu(), data_classes.detach().cpu(), False])
                    class_error.append(metric_item.item())


                prediction.append(predict[:, :, real_index])  # preds = forward_return[0]
                real_value.append(real.to(predict.device))  # testy = forward_return[1]


                r.append(data[:,:,:,0].to(predict.device))

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)

        r = torch.cat(r, dim=0)

        # re-scale data
        prediction = SCALER_REGISTRY.get(self.scaler["func"])(
            prediction, **self.scaler["args"])
        real_value = SCALER_REGISTRY.get(self.scaler["func"])(
            real_value, **self.scaler["args"])


        for metric_name, metric_func in self.metrics.items():
            if class_error and metric_name == 'ACC':
                self.update_epoch_meter("test_" + metric_name, sum(class_error) / len(class_error))
            elif metric_name != 'ACC':
                if self.evaluate_on_gpu:
                    metric_item = self.metric_forward(metric_func, [prediction, real_value])
                else:
                    metric_item = self.metric_forward(metric_func, [prediction.detach().cpu(), real_value.detach().cpu()])
                self.update_epoch_meter("test_" + metric_name, metric_item.item())



    @master_only
    def on_validating_end(self, train_epoch: Optional[int]):
        """Callback at the end of validating.

        Args:
            train_epoch (Optional[int]): current epoch if in training process.
        """

        if train_epoch is not None:
            self.save_best_model(train_epoch, "val_MAE", greater_best=False)
