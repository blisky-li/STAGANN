import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, units, activations):
        super(FC, self).__init__()
        self.layers = nn.ModuleList()
        for unit, activation in zip(units, activations):

            self.layers.append(nn.Linear(unit[0], unit[1]))
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Model(nn.Module):
    def __init__(self, T, d):
        super(Model, self).__init__()
        self.T = T
        self.d = d

        self.fc1 = FC([[1,d],[d,d]], ['relu',None])
        self.fc2 = FC([[d, d], [d, d]], ['relu', None])
        self.fc3 = FC([[d, d], [d, d]], ['relu', 'tanh'])
        self.fc4 = FC([[d, d]], [None])
        self.fc5 = FC([[d, d], [d, d]], ['relu', None])

        self.fc6 = FC([[2*d, d], [d, d]], ['relu', None])
        self.fc7 = FC([[2*d, d], [d, d]], ['relu', None])
        self.fc8 = FC([[d, d], [d, d]], ['relu', 'relu'])

        self.fcte = FC([[1, d], [d, d]], ['relu', None])

        self.fclast = FC([[d, d], [d, d], [d, 1]], ['relu', 'relu', None])
        self.linear = nn.Linear(d, d)

        self.cell = nn.GRUCell(d, d)

    def forward(self, x_gp_fw, x_gp_bw, gp_fw, gp_bw, TE):
        N_target = x_gp_fw.shape[1]
        h = x_gp_fw.shape[2]
        K = gp_fw.shape[-1]
        #print(N_target, h, K)
        # input
        #print(x_gp_fw.device, x_gp_bw.device)
        '''x_gp_fw = (x_gp_fw - self.mean) / self.std
        x_gp_bw = (x_gp_bw - self.mean) / self.std'''
        #x_gp_fw = x_gp_fw.transpose(-1, -2)
        #x_gp_bw = x_gp_bw.transpose(-1, -2)
        x_gp_fw = self.fc1(x_gp_fw)
        x_gp_bw = self.fc1(x_gp_bw)
        #print(x_gp_fw.shape,'fc1')
        # spatial aggregation
        gp_fw = gp_fw.repeat(1, 1, h, 1, 1)

        gp_bw = gp_bw.repeat(1, 1, h, 1, 1)
        #print(gp_fw.shape, x_gp_fw.shape, 'gp_fw')
        #y_gp_fw = gp_fw * x_gp_fw
        #y_gp_bw = gp_bw * x_gp_bw
        #print(y_gp_fw.shape)
        y_gp_fw = torch.matmul(gp_fw, x_gp_fw)
        y_gp_bw = torch.matmul(gp_bw, x_gp_bw)
        #print(y_gp_bw.shape, 'matual')
        y_gp_fw = self.fc2(y_gp_fw)
        y_gp_bw = self.fc2(y_gp_bw)

        x_gp_fw = self.fc2(x_gp_fw)
        x_gp_bw = self.fc2(x_gp_bw)
        x_gp_fw = torch.abs(y_gp_fw - x_gp_fw)
        x_gp_bw = torch.abs(y_gp_bw - x_gp_bw)
        #print(x_gp_fw.shape, y_gp_fw.shape, 'fc2')
        #x_gp_fw = gp_fw * x_gp_fw
        #x_gp_bw = gp_bw * x_gp_bw
        x_gp_fw = torch.matmul(gp_fw, x_gp_fw)
        x_gp_bw = torch.matmul(gp_bw, x_gp_bw)
        #print(x_gp_bw.shape, 'matual22222')
        x_gp_fw = self.fc3(x_gp_fw)
        x_gp_bw = self.fc3(x_gp_bw)
        y_gp_fw = self.fc4(y_gp_fw)
        y_gp_bw = self.fc4(y_gp_bw)
        y_gp_fw = x_gp_fw + y_gp_fw
        y_gp_bw = x_gp_bw + y_gp_bw
        y_gp_fw = self.fc5(y_gp_fw)
        y_gp_bw = self.fc5(y_gp_bw)
        #print(y_gp_bw.shape, 'fc3fc4fc5')
        # temporal modeling
        TE = self.fcte(TE)
        '''TE = F.one_hot(TE, num_classes = self.T)
        TE = self.fc(TE)
        TE = TE.repeat(N_target, 1, 1)'''
        y = torch.cat((y_gp_fw, y_gp_bw), dim = -1)
        y = y.squeeze(dim = 3)
        #print(y.shape, 'y   y')
        y = self.fc6(y)
        x = torch.cat((x_gp_fw, x_gp_bw), dim = -1)
        x = x.squeeze(dim = 3)
        #print(x.shape, 'x   x')
        x = self.fc7(x)
        g1 = self.fc8(x)
        g1 = 1 / torch.exp(g1)
        #print(g1.shape, 'g   g')
        y = g1 * y
        #print(y.shape, TE.shape)
        #y = torch.cat((y, TE), dim = -1)
        y = y + TE
        pred = []
        state = torch.zeros(x.shape[0], N_target, self.d).to(x.device)
        B = x.shape[0]
        #print(state.shape)
        for i in range(h):
            if i == 0:
                g2 = self.linear(state)

                g2 = F.relu(g2)
                g2 = 1 / torch.exp(g2)
                #print(g2.shape,'g2g2g2g2')
                state = g2 * state
                state = state.view(B*N_target, -1)
                #print(state.shape, 'state')
                #print(y[:, :, i, :].shape, '1111')
                state = self.cell(y[:, :, i, :].view(B*N_target, -1), state).view(B, N_target, -1)
                #print(state.shape,  'state11111111')
                pred.append(state.unsqueeze(-2))
            else:
                g2 = self.linear(x[:,:,i-1,:])
                g2 = F.relu(g2)
                g2 = 1 / torch.exp(g2)
                #print(g2.shape, 'g2g2g2g2')
                state = g2 * state
                state = state.view(B * N_target, -1)
                state = self.cell(y[:, :, i,:].view(B*N_target, -1), state).view(B, N_target, -1)
                pred.append(state.unsqueeze(-2))
        pred = torch.cat(pred, dim = -2)
        # output

        pred = self.fclast(pred).squeeze()
        return pred



class INCREASE(nn.Module):
    def __init__(self, K = 5, t_of_d = True, d_of_w = False, m_of_y = False):
        super(INCREASE, self).__init__()

        self.K = K
        #self.linear = nn.Linear(20,20)
        self.tod = t_of_d
        self.dow = d_of_w
        self.moy = m_of_y

        if self.tod:
            self.T_i_D_emb = nn.Parameter(torch.randn(288, 24))
            self.model = Model(T=288, d=8)
        elif self.dow:
            self.T_i_D_emb = nn.Parameter(torch.randn(7, 24))
            self.model = Model(T=7, d=8)
        elif self.moy:
            self.T_i_D_emb = nn.Parameter(torch.randn(12, 24))
            self.model = Model(T=12, d=8)
        else:
            self.T_i_D_emb = nn.Parameter(torch.randn(1, 24))
            self.model = Model(T=1, d=8)

    '''def forward(self, X: torch.Tensor, adj: torch.Tensor, unknown_nodes, batch_seen: int, epoch: int, train: bool,
                **kwargs):'''
    def forward(self, X: torch.Tensor, adj: torch.Tensor, unknown_nodes, batch_seen: int, epoch: int, train: bool, **kwargs):

        adj = adj.to(X.device)
        if len(adj.shape) == 2:
            adj = adj.repeat(X.shape[0],1, 1)
        #print(X.shape)
        if self.tod:
            if torch.max(X[:,:,:,1]) > 1:
                T_D = self.T_i_D_emb[X[:, :, :, 1].type(torch.LongTensor)][:, -1, :, :]
            else:
                T_D = self.T_i_D_emb[(X[:, :, :, 1] * 288).type(torch.LongTensor)][:, -1, :, :]
        elif self.dow:
            if torch.max(X[:, :, :, 2]) > 1:
                T_D = self.T_i_D_emb[X[:, :, :, 2].type(torch.LongTensor)][:, -1, :, :]
            else:
                T_D = self.T_i_D_emb[(X[:, :, :, 2] * 7).type(torch.LongTensor)][:, -1, :, :]
        elif self.moy:
            if torch.max(X[:, :, :, -1]) > 1:
                T_D = self.T_i_D_emb[(X[:, :, :, -1]).type(torch.LongTensor)][:, -1, :, :]
            else:
                T_D = self.T_i_D_emb[(X[:, :, :, -1] * 12).type(torch.LongTensor)][:, -1, :, :]
        else:
            T_D = self.T_i_D_emb

        X = X[...,0]
        X = X.transpose(-1,-2)
        B, N, E = X.shape
        if unknown_nodes.shape[0] == 1:
            unknown_nodes = unknown_nodes.repeat(X.shape[0], 1)
        _, Nindex = unknown_nodes.shape
        #Nindex = N

        T_D = T_D[torch.arange(X.shape[0])[:, None],  unknown_nodes, :]
        # Initialize the tensors to store the known nodes and their relationships
        known_nodes = torch.zeros((B, Nindex, E, self.K)).to(X.device)
        known_relations = torch.zeros((B, Nindex, self.K)).to(X.device)
        known_nodes2 = torch.zeros((B, Nindex, E, self.K)).to(X.device)
        known_relations2 = torch.zeros((B, Nindex, self.K)).to(X.device)
        diag = torch.diagonal(adj, dim1=-2, dim2=-1)
        a_diag = torch.diag_embed(diag)
        # print(a_diag.shape, A_q)
        adj = adj - a_diag
        adj2 = adj.transpose(-1,-2)

        # Iterate over each batch
        for b in range(B):
            # Iterate over each missing node
            for i in range(Nindex):
                # Get the index of the missing node
                missing_node_index = unknown_nodes[b, i]

                # Get the relationships of the missing node with all other nodes
                relationships = adj[b, missing_node_index, :]
                relationships2 = adj2[b, missing_node_index, :]
                # Get the indices of the nodes with the top-K relationships
                _, topk_indices = torch.topk(relationships, self.K)
                _, topk_indices2 = torch.topk(relationships2, self.K)
                # Get the known nodes and their relationships
                known_nodes[b, i, :, :] = X[b, topk_indices, :].transpose(0, 1)
                known_relations[b, i, :] = relationships[topk_indices]

                known_nodes2[b, i, :, :] = X[b, topk_indices2, :].transpose(0, 1)
                known_relations2[b, i, :] = relationships2[topk_indices2]
        #print(known_nodes, known_relations)
        known_nodes = known_nodes.unsqueeze(-1)
        known_nodes2 = known_nodes2.unsqueeze(-1)
        known_relations = known_relations.unsqueeze(-2).unsqueeze(-2)
        known_relations2 = known_relations2.unsqueeze(-2).unsqueeze(-2)
        T_D = T_D.unsqueeze(-1)

        x = self.model(known_nodes, known_nodes2, known_relations, known_relations2,T_D)
        for b in range(B):
            # 将计算出的节点值放回到它们原来的位置
            X[b, unknown_nodes[b], :] = x[b]
            #print(X[b, unknown_nodes[b], :])
        X = X.permute(0, 2, 1)
        #print(X.shape)
        return [X]
        #return [time_series]

