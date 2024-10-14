import torch
import torch.nn as nn

from .ignnk_gnn import D_GCN, C_GCN, K_GCN


class IGNNK(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets
   x_s
    |GNN_3
   H_2 + H_1
    |GNN_2
   H_1
    |GNN_1
  x^y_m
    """

    def __init__(self, time_dimension, hidden_dimnesion , order):
        super(IGNNK, self).__init__()
        self.time_dimension = time_dimension
        self.hidden_dimnesion = hidden_dimnesion
        self.order = order

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimnesion, self.order)
        self.GNN2 = D_GCN(self.hidden_dimnesion, self.hidden_dimnesion, self.order)
        self.GNN3 = D_GCN(self.hidden_dimnesion, self.time_dimension, self.order, activation='linear')

    def forward(self, X : torch.Tensor, adj : torch.Tensor, unknown_nodes, batch_seen: int, epoch: int, train: bool, **kwargs):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X = X[:,:,:,0]
        #print(adj.shape)
        #adj = torch.tensor(adj).to(X.device)
        #adj = torch.tensor(adj, dtype=torch.float32).to(X.device)


        if len(adj) == 1:
            A_q = adj
            A_h = A_q.T
        elif len(adj) == 2:
            A_q = adj[0]
            A_h = adj[1]
        else:
            if len(adj.shape) == 2:
                A_q = adj
                A_h = A_q.T
            else:
                A_q = adj
                A_h = A_q.permute(0,2,1)
        #print(A_h.shape, A_h.sum())
        X_S = X.permute(0, 2, 1)  # to correct the input dims





        X_s1 = self.GNN1(X_S, A_q, A_h)



        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1 # num_nodes, rank
        X_s3 = self.GNN3(X_s2, A_q, A_h)

        X_res = X_s3.permute(0, 2, 1)

        return [X_res]