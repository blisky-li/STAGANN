import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .satcn_gnn import AGGREGATORS,AGGREGATORS_MASK,SCALERS



# A_to_A = A[list(unknow_set), :][:, list(know_set)]
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class tcn_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="linear", dropout=0.1):
        super(tcn_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1,padding=(1,0))
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1,padding=(1,0))
        self.dropout = dropout


    def forward(self, x):
        """
        :param x: Input data of shape (batch_size, num_variables, num_timesteps, num_nodes)
        :return: Output data of shape (batch_size, num_features, num_timesteps - kt, num_nodes)
        """
        #x_in = self.align(x)
        x_in = x

        #print("xin", x_in.shape, x.shape)
        if self.act == "GLU":
            x_conv = self.conv(x)

            h = (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            return F.dropout(h, self.dropout, training=self.training)
        if self.act == "sigmoid":

            h = torch.sigmoid(self.conv(x) + x_in)
            return F.dropout(h, self.dropout, training=self.training)
        #print(self.conv(x).shape,x_in.shape)
        h = self.conv(x)[:,:,1:,:] + x_in
        return F.dropout(h, self.dropout, training=self.training)


class tcn_layer2(nn.Module):
    def __init__(self, c_in, c_out, act="GLU", dropout=0.1):
        super(tcn_layer2, self).__init__()

        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out * 2,kernel_size= (1, 1),bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size= (1, 1),bias=True)
        self.dropout = dropout


    def forward(self, x):
        """
        :param x: Input data of shape (batch_size, num_variables, num_timesteps, num_nodes)
        :return: Output data of shape (batch_size, num_features, num_timesteps - kt, num_nodes)
        """
        #x_in = self.align(x)
        x_in = x

        #print("xin", x_in.shape, x.shape)

        if self.act == "GLU":
            x_conv = self.conv(x)

            h = (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            return F.dropout(h, self.dropout, training=self.training)
        if self.act == "sigmoid":

            h = torch.sigmoid(self.conv(x) + x_in)
            return F.dropout(h, self.dropout, training=self.training)
        #print(self.conv(x).shape,x_in.shape)
        h = torch.tanh(self.conv(x) + x_in)

        return F.dropout(h, self.dropout, training=self.training)




class STower(nn.Module):
    """
    Spatil aggragation layer applies principle aggragation on the spatial dimension
    """

    def __init__(self, in_features, out_features, aggregators, scalers, masking=False, dropout=0.1):
        super(STower, self).__init__()

        #self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.aggregators = aggregators
        self.scalers = scalers

        '''self.Theta_po = nn.Parameter(
            torch.FloatTensor(len(aggregators) * len(scalers) * self.in_features, self.out_features)).cuda()'''
        self.Theta_po = nn.Parameter(
            torch.FloatTensor(len(aggregators) * len(scalers) * self.in_features, self.out_features)).cuda()
        self.bias_po = nn.Parameter(torch.FloatTensor(self.out_features)).cuda()
        self.reset_parameters()

        self.masking = masking
        self.dropout = dropout

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta_po.shape[1])
        self.Theta_po.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.bias_po.shape[0])
        self.bias_po.data.uniform_(-stdv, stdv)

    def forward(self, X, adj):
        """
        :param X: Input data of shape (batch_size, in_features, num_timesteps, in_nodes)
        :adj: The adjacency (num_nodes, num_nodes) missing_nodes (The kriging target nodes )
        :return: Output data of shape (batch_size, num_nodes, num_timesteps, out_features)
        """
        I = X.permute([0, 2, 3, 1])
        if len(adj.shape) == 3:
            (_, N,_) = adj.shape
        else:
            (N, _) = adj.shape
        adj = adj
        if self.masking:
            m = torch.cat([AGGREGATORS_MASK[aggregate](I, adj, device=X.device) for aggregate in self.aggregators],
                          dim=3)

            m[torch.isnan(m)] = 6
            m[torch.isinf(m)] = 0
            m[m > 6] = 6
        else:

            m = torch.cat([AGGREGATORS[aggregate](I, adj, device=X.device) for aggregate in self.aggregators], dim=3)

        m = torch.cat([SCALERS[scale](m, adj) for scale in self.scalers], dim=3)

        #print(m.shape,self.Theta_po.shape)
        out = torch.einsum("btji,io->btjo", [m, self.Theta_po])
        out += self.bias_po
        out = F.dropout(out, self.dropout, training=self.training)
        return out.permute([0, 3, 1, 2])


class SATCN(nn.Module):

    def __init__(self,  in_variables=1, layers=1, channels=32, neighbor=2,
                 aggragators=['mean'],
                 scalers=['identity'], masking=True, dropout=0):
        super(SATCN, self).__init__()
        self.s_layer0 = STower(in_variables, channels, aggragators, ['identity'],masking, dropout)
        #self.t_layer0 = tcn_layer(t_kernel, channels, channels, dropout=dropout)
        self.t_layer0 = tcn_layer2(channels, channels, "linear", dropout=dropout)
        self.s_convs = nn.ModuleList()
        self.t_convs = nn.ModuleList()
        self.neighbor = neighbor
        self.layers = layers
        for i in range(layers):
            self.s_convs.append(STower(channels, channels, aggragators, scalers,  False, dropout))
            self.t_convs.append(tcn_layer2( channels, channels, "linear", dropout=dropout))
            #self.t_convs.append(tcn_layer(t_kernel, channels, channels, dropout=dropout))
        self.out_conv = nn.Conv2d(channels, in_variables, (1, 1), 1)

    def forward(self, X, adj, adj_mask, batch_seen: int, epoch: int, train: bool, **kwargs):
        X = X.unsqueeze(1)[:,:,:,:,0]
        adj = torch.tensor(adj, dtype=torch.float32).to(X.device)
        adj_mask = torch.tensor(adj_mask, dtype=torch.float32).to(X.device)


        #print(X.shape)
        if len(adj.shape) == 2:
            adj = adj.repeat(X.shape[0],1,1)
        if len(adj_mask.shape) == 2:
            adj_mask = adj_mask.repeat(X.shape[0],1,1)
        adj,adj_mask = adj.to(X.device),adj_mask.to(X.device)


        adj2, _ = torch.topk(adj, k=self.neighbor, dim=-1, largest=True, sorted=True)

        adj = torch.where(adj < adj2[:,:,-1].unsqueeze(-1), torch.zeros_like(adj), adj)
        adj3, _ = torch.topk(adj_mask, k=self.neighbor, dim=-1, largest=True, sorted=True)

        adj_mask = torch.where(adj_mask < adj3[:, :, -1].unsqueeze(-1), torch.zeros_like(adj), adj_mask)



        x = self.s_layer0(X, adj_mask)
        #x = torch.relu(x)
        x = self.t_layer0(x)
        #x = torch.relu(x)
        #print("111", x.shape)
        for i in range(self.layers):

            x = self.s_convs[i](x, adj)
            x = torch.relu(x)
            x = self.t_convs[i](x)
            x = torch.relu(x)

        y = self.out_conv(x)
        #print(y.shape)
        return [y.reshape(X.shape[0],X.shape[2],X.shape[3])]


