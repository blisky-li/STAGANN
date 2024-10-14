import torch
import torch.nn as nn

#from .gcn_gnn import D_GCN, C_GCN, K_GCN


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim, order, support_len, dropout=0.3):
        super(GCNConv, self).__init__()
        in_dim = order * support_len * in_dim
        #number=order * support_len
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.order = order
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x,  support):
        out = x
        res = []
        #print(torch.cuda.memory_allocated(),'122222222222')
        for A in support:
            out = x
            for _ in range(self.order):
                out = torch.matmul(A, out)
                res.append(out)

        out = torch.cat(res, dim=2)
        out = self.linear(out)
        out = self.dp(out)
        return out

class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim, order, support_len, dropout=0.05):
        super(GINConv, self).__init__()
        in_dim = order * support_len * in_dim
        # number=order * support_len
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        '''nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

        print(torch.sum(torch.mean(self.linear.weight.data)), torch.sum(self.linear.weight.data),
              torch.max(self.linear.weight.data), 'GINCONV1')
        print(torch.sum(torch.mean(self.linear2.weight.data)), torch.sum(self.linear2.weight.data),
              torch.max(self.linear2.weight.data), 'GINCONV2')'''
        self.bn = nn.BatchNorm1d(out_dim)
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.order = order
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x, support):
        out = x
        res = []
        for A in support:
            out = x
            for _ in range(self.order):
                out = (1 + self.eps) * out + torch.matmul(A, out)
                res.append(out)

        out = torch.cat(res, dim=2)
        out = self.linear(out)
        out = self.dp(out)
        B, N, C = out.shape
        out = self.bn(out.view(B * N, C)).view(B, N, C)
        # out = self.dp(out)
        out = self.linear2(F.relu(out))
        out = self.dp(out)
        return out

class GCN(nn.Module):
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
        super(GCN, self).__init__()
        self.time_dimension = time_dimension
        self.hidden_dimnesion = hidden_dimnesion
        self.order = order

        self.GNN1 = GCNConv(self.time_dimension, self.hidden_dimnesion, self.order, 2)
        self.GNN2 = GCNConv(self.hidden_dimnesion, self.hidden_dimnesion, self.order, 2)
        self.GNN3 = GCNConv(self.hidden_dimnesion, self.time_dimension, self.order, 2)

    def forward(self, X : torch.Tensor, adj : torch.Tensor, unknown_nodes, batch_seen: int, epoch: int, train: bool, **kwargs):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """
        X = X[:,:,:,0]




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
                A_h = A_q.transpose(-1, -2)
        X_S = X.permute(0, 2, 1)  # to correct the input dims
        #print(A_q.device, A_h.device)
        '''A_q = A_q.to(X.device)
        A_h = A_h.to(X.device)'''

        #print(torch.cuda.memory_allocated())
        #print(A_q.shape)
        X_s1 = self.GNN1(X_S, [A_q, A_h])
        X_s2 = self.GNN3(X_s1, [A_q, A_h])
        '''print(X_S.shape, X_s1.shape, X_mask.shape)
        X_s1 = torch.mul(X_s1, X_mask) + X_S

        X_s2 = torch.mul(self.GNN2(X_s1, A_q, A_h), X_mask) + X_S + X_s1  # num_nodes, rank
        X_s3 = torch.mul(self.GNN3(X_s2, A_q, A_h), X_mask) + X_S'''
        #print(X_s2.shape)
        X_res = X_s2.permute(0, 2, 1)

        return [X_res]