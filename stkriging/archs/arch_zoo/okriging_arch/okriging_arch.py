import torch
import torch.nn as nn

class Okriging(nn.Module):

    def __init__(self,order = 1):
        super(Okriging, self).__init__()
        self.order = order
        self.Theta1 = nn.Parameter(torch.ones(order))

    def forward(self, X : torch.Tensor, adj , batch_seen: int, epoch: int, train: bool, **kwargs):
        X = X[:, :, :, 0]

        if len(adj) == 1:
            adj = adj
        elif len(adj) == 2:
            adj = adj[0]
        else:
            if len(adj.shape) == 2:
                adj = adj
            else:
                adj = adj
        adj = torch.tensor(adj).to(X.device)
        #X = X.permute(0,2,1)
        sum_adj = torch.sum(adj,dim=-2)
        if len(adj.shape) == 2:
            x = torch.matmul(X,adj)
            l = []
            for i in range(X.shape[0]):
                x_l = torch.div(x[i,:,:],sum_adj+0.000000001)
                l.append(x_l)

            x_adj = torch.stack(l, dim=0).reshape(X.shape[0], X.shape[1], X.shape[2])
            '''l = []
            for i in range(X.shape[0]):

                for j in range(X.shape[1]):
                    x_j = torch.sum(X[i, :, :] * adj[ j, :].reshape(-1, 1), dim=0) / sum_adj[j]

                    l.append(x_j)

            x_adj = torch.stack(l, dim=0).reshape(X.shape[0], X.shape[1], X.shape[2])'''
        else:
            x = torch.matmul(X, adj)
            l = []
            for i in range(X.shape[0]):
                x_l = torch.div(x[i, :, :], sum_adj[i,:]+0.000000001)
                l.append(x_l)
            x_adj = torch.stack(l, dim=0).reshape(X.shape[0], X.shape[1], X.shape[2])
            '''l = []
            for i in range(X.shape[0]):

                for j in range(X.shape[1]):

                    x_j = torch.sum(X[i,:,:] * adj[i,j,:].reshape(-1, 1),dim=0)/sum_adj[i,j]

                    l.append(x_j)

            x_adj = torch.stack(l,dim=0).reshape(X.shape[0],X.shape[1],X.shape[2])'''
        #print(x_adj)
        #x_adj= x_adj.permute(0, 2, 1)

        x_adj = torch.where(torch.round(self.Theta1) > 0, x_adj * torch.round(self.Theta1), x_adj * (torch.round(self.Theta1) + 1))

        return [x_adj]