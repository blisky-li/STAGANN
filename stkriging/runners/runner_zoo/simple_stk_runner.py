import torch
import numpy as np
from ..base_stk_runner import BaseSpatiotemporalKrigingRunner


class SimpleSpatiotemporalKrigingRunner(BaseSpatiotemporalKrigingRunner):
    """Simple Runner: select forward features and target features. This runner can cover most cases."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target feature.

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: torch.Tensor, adj : torch.Tensor, unknown_nodes : torch.tensor, location=None, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess

        data = self.to_running_device(data.clone().detach())
        adj = torch.tensor(adj, dtype=torch.float32).clone().detach()
        adj = self.to_running_device(adj)# B, L, N, C
        #print(data.shape, adj.shape)
        '''print(data.type(), adj.type())
        print(torch.isfinite(data).logical_not().any())
        print(torch.isfinite(adj).logical_not().any())
        for i in range(adj.shape[0]):
            print(adj[i])'''
            # B, L, N, C
        batch_size, length, num_nodes, _ = data.shape
        #print(location)

        if location != None:
            location1, location2 = location[0].to(data.device), location[1].to(data.device)
            location = [location1, location2]
            #print(location.device)
        else:
            location = None
        history_data = self.select_input_features(data)

        if unknown_nodes.ndim == 1:
            unknown_nodes = np.expand_dims(unknown_nodes, axis = 0)
            unknown_nodes = np.tile(unknown_nodes, (history_data.shape[0], 1))
        #print('kkkkkk', history_data.device, adj.device)
        # curriculum learning
        if self.cl_param is None:
            if  location != None:
                prediction_data = self.model(X=history_data, adj=adj, unknown_nodes=unknown_nodes,location=location, batch_seen=iter_num,
                                             epoch=epoch, train=train)
            else:
                prediction_data = self.model(X=history_data, adj=adj, unknown_nodes=unknown_nodes,location=None, batch_seen=iter_num, epoch=epoch, train=train)
        else:
            task_level = self.curriculum_learning(epoch)
            if  location != None:
                prediction_data = self.model(X=history_data, adj=adj, unknown_nodes=unknown_nodes, location=location, batch_seen=iter_num,
                                             epoch=epoch, train=train, task_level=task_level)
            else:
                prediction_data = self.model(X=history_data, adj=adj, unknown_nodes=unknown_nodes,location=None, batch_seen=iter_num, epoch=epoch, train=train,task_level=task_level)
        # feed forward
        #assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        #prediction = self.select_target_features(prediction_data)

        return prediction_data
