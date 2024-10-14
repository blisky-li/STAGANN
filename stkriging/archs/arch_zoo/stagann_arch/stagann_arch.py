import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .revin import RevIN
import numpy as np

import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean


def min_max(tensor):
    min_values = torch.min(tensor, dim=-1).values.unsqueeze(-1)
    max_values = torch.max(tensor, dim=-1).values.unsqueeze(-1)

    return (tensor - min_values) / (max_values - min_values + 1)

class TemporalConvNet(nn.Module):
    def __init__(self, residual_channels=4,
                    dilation_channels=4, skip_channels=8, end_channels=8,  layers=4, kernel_size=3):
        super(TemporalConvNet, self).__init__()
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.layers = layers
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        additional_scope = kernel_size - 1
        new_dilation = 1
        receptive_field = 1
        for i in range(layers):
            # dilated convolutions
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                               out_channels=dilation_channels,
                                               kernel_size=(1, kernel_size), dilation=new_dilation))

            self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                             out_channels=dilation_channels,
                                             kernel_size=(1, kernel_size), dilation=new_dilation))

            # 1x1 convolution for residual connection
            self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

            # 1x1 convolution for skip connection
            self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                             out_channels=skip_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            new_dilation *= 2
            receptive_field += additional_scope
            additional_scope *= 2
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    )

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=24,
                                    kernel_size=(1, 1),
                                    )
        self.receptive_field = receptive_field
    def forward(self, x):
        x = x.transpose(1, 2)
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(
                    x, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = x

        x = self.start_conv(x)
        skip = 0
        for i in range(self.layers):
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = torch.mean(x, dim=-1).transpose(1, 2)
        return x





class GINConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, order, support_len, dropout=0.05):
        super(GINConv, self).__init__()
        in_dim = order * support_len * in_dim
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.layersnorm = torch.nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-08,
                                             elementwise_affine=False)
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.order = order
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x, support):
        res = []
        for A in support:
            out = x
            for _ in range(self.order):
                if len(out.shape) == 3:
                    out = (1 + self.eps) * out + torch.matmul(A, out)
                elif len(out.shape) == 4:
                    out = (1 + self.eps) * out + torch.einsum('BNLE, BNK->BKLE', out, A)
                else:
                    out = out
                res.append(out)

        out = torch.cat(res, dim=-1)
        out = self.linear(out)
        if len(out.shape) == 3:
            B, N, C = out.shape
            out = self.bn(out.view(B * N, C)).view(B, N, C)
        elif len(out.shape) == 4:
            B, N, L, C = out.shape
            out = self.bn(out.view(B * N * L, C)).view(B, N, L, C)
        else:
            out = out
        #out = self.layersnorm(out)
        out = self.dp(out)
        out = self.linear2(F.relu(out))
        out = self.dp(out)
        return out



def adj_mask_unknown_node(adj, unknown_idx):
    if len(adj.shape) == 2:
        unknown_idx_adj = torch.LongTensor(unknown_idx).to(adj.device).repeat(adj.shape[0],1)
        adj = adj.scatter(1,unknown_idx_adj,0)
        return adj
    elif len(adj.shape) == 3:
        if unknown_idx.shape[0] == 1:
            unknown_idx = unknown_idx.tile(adj.shape[0], 1)
        l = unknown_idx.shape[1]

        unknown_idx_adj = torch.LongTensor(unknown_idx).to(adj.device).repeat(1,adj.shape[1]).reshape(adj.shape[0], adj.shape[1], l)
        adj = adj.scatter(2, unknown_idx_adj, 0)
        return adj
    else:
        return adj

class GINbackbone(nn.Module):
    def __init__(self, in_dim,hidden_dim, out_dim, order, support_len, dropout=0.05):
        super(GINbackbone, self).__init__()
        in_dim = order * support_len * in_dim
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.layersnorm = torch.nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-08,
                                             elementwise_affine=False)
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.order = order
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x, support):
        res = []
        for A in support:
            out = x
            for _ in range(self.order):
                if len(out.shape) == 3:
                    out = torch.matmul(A, out)
                elif len(out.shape) == 4:
                    out = torch.einsum('BNLE, BNK->BKLE', out, A)
                else:
                    out = out
                res.append(out)
        out = torch.cat(res, dim=-1)
        out = self.linear(out)

        if len(out.shape) == 3:
            B, N, C = out.shape
            out = self.bn(out.view(B * N, C)).view(B, N, C)
        elif len(out.shape) == 4:
            B, N, L, C = out.shape
            out = self.bn(out.view(B * N * L, C)).view(B, N, L, C)
        else:
            out = out
        out = self.dp(out)
        out = self.linear2(F.relu(out))
        out = self.dp(out)

        return out

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-8):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        return ln_out



class ScaledDotProductAttension(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale):
        super().__init__() #声明父类的Init方法
        self.scale = scale
        self.softmax = nn.Softmax(dim = 2) #沿哪一维实施softmax
    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) #matmul: matrix multiply
        u = u / self.scale #缩放

        if mask is not None:
            u = u.masked_fill(mask, -np.inf)
        attn = self.softmax(u)
        output = torch.bmm(attn, v)
        return attn, output

class MultiHeadAttention2(nn.Module):
    """ Multi-Head Attention """
    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o, dp=0.3):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(p=dp)
        # 用于投影变换mlp
        self.fc_q = nn.Linear(d_k_, n_head * d_k, bias=False)
        self.fc_k = nn.Linear(d_k_, n_head * d_k, bias=False)
        self.fc_v = nn.Linear(d_v_, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttension(scale=np.power(d_k, 0.5))
        self.fc_concatOutput = nn.Linear(n_head * d_v, d_o, bias=False) # concat -> mlp -> output
    def forward(self, q, k, v, mask = None):
        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v
        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        #投影变化，单头变多头
        q = self.fc_q(q)
        q = self.dropout(q)
        k = self.fc_k(k)
        k = self.dropout(k)
        v = self.fc_v(v)
        v = self.dropout(v)

        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            # repeat(n_head, 1, 1): 将mask沿第0维复制 n_head次，其他维度不变
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1)  # Concat
        output = self.fc_concatOutput(output)  # 投影变换得到最终输出
        return attn, output


class SelfAttention(nn.Module):
    """ Self-Attention """
    def __init__(self, n_head, d_k, d_v, d_x, d_y, d_z, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_y, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_z, d_v))
        self.dropout = 0.3
        self.dp = nn.Dropout(p=self.dropout)
        self.mha = MultiHeadAttention2(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, y, z, mask = None):
        q = torch.matmul(x, self.wq)
        k = torch.matmul(y, self.wk)
        v = torch.matmul(z, self.wv)
        q = self.dp(q)
        k = self.dp(k)
        v = self.dp(v)
        attn, output = self.mha(q, k, v, mask=mask)
        return attn, output

def graphmasekd(adj, K):
    values, indices = adj.topk(K, dim=2)

    # 创建一个形状和原Tensor相同的零Tensor
    zero_tensor = torch.zeros_like(adj).to(adj.device)
    #print(indices[0])
    # 使用索引将前K大的数值放入零Tensor中
    zero_tensor.scatter_(2, indices, values)
    # 将所有0的值替换为负无穷

    zero_tensor = zero_tensor / (zero_tensor.sum(dim=-1, keepdims=True) + 1e-5)
    return zero_tensor


class DFDgraph(nn.Module):
    def __init__(self, time_dimension, hidden, emb=12, dropout=0.01, t_n = 2, mask = 5):
        super(DFDgraph,self).__init__()
        self.hidden_emb = hidden
        self.mask = mask
        self.Wd0 = nn.Parameter(torch.randn(time_dimension // 2 + 1, self.hidden_emb), requires_grad=True)
        self.We0 = nn.Parameter(torch.randn(self.hidden_emb + emb * t_n, self.hidden_emb), requires_grad=True)
        self.Wxabs0 = nn.Parameter(torch.randn(self.hidden_emb, self.hidden_emb), requires_grad=True)
        self.W = nn.Parameter(torch.randn(self.hidden_emb, 1), requires_grad=True)
        self.drop = nn.Dropout(p=dropout)
        self.layersnorm = torch.nn.LayerNorm(normalized_shape=self.hidden_emb, eps=1e-08,
                                             elementwise_affine=False)

    def forward(self, x, t_emb):
        xn10 = torch.fft.rfft(x, dim=-1, norm="ortho")
        xn10 = torch.abs(xn10)

        xn10 = min_max(xn10)
        xn10 = torch.nn.functional.normalize(xn10, p=2.0, dim=2, eps=1e-12, out=None)
        t_emb = min_max(t_emb)
        t_emb = torch.nn.functional.normalize(t_emb, p=2.0, dim=2, eps=1e-12, out=None)

        xn10 = torch.matmul(xn10, self.Wd0)

        xn10 = torch.cat([xn10, t_emb], dim=2)

        xn10 = self.drop(self.layersnorm(torch.relu(torch.matmul(xn10, self.We0))))
        loc_emb = xn10.unsqueeze(2)
        loc_emb2 = xn10.unsqueeze(1)
        adj = loc_emb * loc_emb2
        adj = torch.relu(torch.matmul(adj, self.W).squeeze())
        adj = graphmasekd(adj, self.mask)

        return adj


class MetaGraph2(nn.Module):
    def __init__(self, time_dimension, hidden_emb,  emb=12, dropout=0.01, t_n=2, mask=5):
        super(MetaGraph2,self).__init__()
        self.hidden_emb = hidden_emb
        self.emb = emb
        self.dp = nn.Dropout(p = dropout)
        self.mask = mask
        self.Wdfft = nn.Parameter(torch.randn(time_dimension // 2 + 1, self.emb), requires_grad=True)
        self.linear_number = nn.Linear(12 * self.hidden_emb,  self.emb)
        self.linear_number2 = nn.Linear(12 * self.hidden_emb,  self.emb)
        self.Wd0 = nn.Parameter(torch.randn(self.emb, self.hidden_emb), requires_grad=True)
        self.We0 = nn.Parameter(torch.randn(self.hidden_emb + self.emb * t_n, self.hidden_emb), requires_grad=True)
        self.linear_number2_time = nn.Linear(self.emb * 2 + self.emb * t_n, self.emb)

        self.linear_num = nn.Parameter(torch.empty(12 * self.hidden_emb), requires_grad=True)

        self.W = nn.Parameter(torch.randn(self.hidden_emb, 1), requires_grad=True)
        self.layersnorm = torch.nn.LayerNorm(normalized_shape=self.hidden_emb, eps=1e-08,
                                             elementwise_affine=False)
        self.layersnorm2 = torch.nn.LayerNorm(normalized_shape=self.hidden_emb, eps=1e-08,
                                              elementwise_affine=False)
        self.atten = SelfAttention(n_head=8, d_k=emb, d_v=emb, d_x=time_dimension // 2 + 1, d_y= self.hidden_emb * 12 + emb * t_n , d_z= time_dimension // 2 + 1,d_o=self.hidden_emb)
    def forward(self, x, grid1, loc, td, unknown):
        xn10 = torch.fft.rfft(x, dim=-1, norm='ortho')
        xn10 = torch.abs(xn10)

        xn10 = min_max(xn10)
        xn10 = torch.nn.functional.normalize(xn10, p=2.0, dim=2, eps=1e-12, out=None)
        loc = min_max(loc)
        loc = torch.nn.functional.normalize(loc, p=2.0, dim=2, eps=1e-12, out=None)
        td = min_max(td)
        td = torch.nn.functional.normalize(td, p=2.0, dim=2, eps=1e-12, out=None)

        td_loc = torch.concat([td, loc], dim=-1)
        _, loc_vecn1 = self.atten(td_loc, xn10,xn10)
        loc_emb = self.dp(self.layersnorm(torch.relu(loc_vecn1)))
        loc_l = loc_emb.unsqueeze(2)
        loc_r = loc_emb.unsqueeze(1)
        loc_adj = loc_l * loc_r
        loc_adj = torch.relu(torch.matmul(loc_adj, self.W).squeeze())
        loc_adj = graphmasekd(loc_adj, self.mask)
        return loc_adj

class DimensionGraph(nn.Module):
    def __init__(self, hidden, hidden2):
        super(DimensionGraph, self).__init__()
        self.hidden = hidden
        self.hidden2 = hidden2
        self.linear1 = nn.Linear(hidden, hidden2)
        self.linear2 = nn.Linear(hidden, hidden2)
        self.thresholds = 0.5
    def forward(self, emb):
        dimension_emb1 = self.linear1(emb)
        dimension_emb2 = self.linear2(emb).transpose(-1, -2)
        adp = F.softmax(
            F.relu(torch.bmm(dimension_emb1, dimension_emb2) - self.thresholds), dim=2)
        return adp




class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)


    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.dim_per_head)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.dim_per_head)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.dim_per_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)


        attention_weights = F.softmax(
            Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float)), dim=-1)
        output = attention_weights @ V
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.input_dim)

        # 使用自注意力的输出作为门控的权重
        gate_weights = torch.sigmoid(self.gate(output))
        output = gate_weights * output

        return output, attention_weights


def deal_with_phase(x):
    x = (x + 3.142) / (3.143 * 2)
    x = torch.floor(x * 36).long()
    #x = x[:, :]).type(torch.LongTensor
    return x

class GINConvFFT(nn.Module):
    def __init__(self, in_dim, hidden_dim,  out_dim, order, support_len, dropout=0.05):
        super(GINConvFFT, self).__init__()


        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.scale = 1 / (order * support_len * hidden_dim)
        self.wphase = nn.Linear(in_dim, in_dim)
        self.weight = nn.Parameter(self.scale * torch.randn(in_dim , order * support_len * hidden_dim, out_dim))

        self.eps = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        self.order = order
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x, support):

        B, N, L, H = x.shape
        x = x.contiguous().view(B, N, L * H)
        res = []
        for A in support:
            out = x
            for _ in range(self.order):
                out = (1 + self.eps) * out + torch.matmul(A, out)
                res.append(out)

        out = torch.stack(res, dim=-1)

        out = out.reshape(B, N, L, H * self.order * len(support))
        out = torch.einsum("bnlm, lmo -> bno", [out, self.weight])
        out = self.dp(out)
        B, N, C = out.shape
        out = self.bn(out.view(B * N, C)).view(B, N, C)
        out = self.linear2(F.relu(out))
        return out

class MultiHeadAttentionqkv(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttentionqkv, self).__init__()
        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dim_per_head = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.gate = nn.Linear(input_dim, input_dim)

    def forward(self, q, k, v):
        batch_size, seq_length, _ = q.size()

        Q = self.query(q).view(batch_size, seq_length, self.num_heads, self.dim_per_head)
        K = self.key(k).view(batch_size, seq_length, self.num_heads, self.dim_per_head)
        V = self.value(v).view(batch_size, seq_length, self.num_heads, self.dim_per_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        attention_weights = F.softmax(
            Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(self.dim_per_head, dtype=torch.float)), dim=-1)
        output = attention_weights @ V
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.input_dim)

        # 使用自注意力的输出作为门控的权重
        gate_weights = torch.sigmoid(self.gate(output))
        output = gate_weights * output

        return output, attention_weights

class Discriminator(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, hidden_dim2=256, dropout=0.3):
        super(Discriminator, self).__init__()
        self.linear1 = nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim))
        self.linear2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim2))
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.dp = nn.Dropout(p = dropout)
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x



class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GIN(nn.Module):
    def __init__(self, time_dimension, hidden_dimnesion, order, meta_data, num_grids=50, mask=0, t_of_d=False, h_of_d=False, d_of_w=False, m_of_y = False, dropout=0.3):
        super(GIN, self).__init__()
        self.order = order
        self.support_len = 2
        self.meta_data = meta_data
        self.ginbackbone = GINbackbone(time_dimension, hidden_dimnesion, time_dimension, self.order, self.support_len)
        self.ginbackbone2 = GINConv(time_dimension, hidden_dimnesion, time_dimension, self.order, self.support_len )
        self.gindeconv = GINConv(time_dimension, hidden_dimnesion, time_dimension, self.order, 1)
        self.time_count = 0
        self.mask = mask

        self.t_of_d = t_of_d
        self.h_of_d = h_of_d
        self.d_of_w = d_of_w
        self.m_of_y = m_of_y
        self.layers = 1
        self.hidden_emb = 20
        self.emb = 12
        self.dropout = dropout
        self.dp = nn.Dropout(p=self.dropout)
        self.ginbacklst = nn.ModuleList()
        self.ginconvlst = nn.ModuleList()

        if t_of_d:
            self.time_count += 1
        self.T_i_D_emb = nn.Parameter(
                torch.empty(288, self.emb), requires_grad=False)
        if h_of_d:
            self.time_count += 1
        self.H_i_D_emb = nn.Parameter(
            torch.empty(24, self.emb), requires_grad=False)
        if d_of_w:
            self.time_count += 1
        self.D_i_W_emb = nn.Parameter(
                torch.empty(7, self.emb), requires_grad=False)
        if m_of_y:
            self.time_count += 1
        self.M_i_Y_emb = nn.Parameter(
                torch.empty(12, self.emb), requires_grad=False)

        self.ginFFT = GINConvFFT(time_dimension//2+1, self.emb, time_dimension//2+1, self.order, self.support_len - 1)
        self.ginFFT2 = GINConvFFT(time_dimension//2+1, self.emb, time_dimension//2+1, self.order, self.support_len - 1)

        self.layernormlst = nn.ModuleList()
        self.layernormlst2 = nn.ModuleList()

        self.revin1 = RevIN(time_dimension, eps=1e-5, affine=False, subtract_last=False)
        self.revin2 = RevIN(time_dimension, eps=1e-5, affine=False, subtract_last=False)
        self.revinfft = RevIN(time_dimension, eps=1e-5, affine=False, subtract_last=False)
        self.revinfft2 = RevIN(time_dimension, eps=1e-5, affine=False, subtract_last=False)

        self.revinlist1 = nn.ModuleList()
        self.revinlist2 = nn.ModuleList()
        self.attenlst1 = nn.ModuleList()
        self.attenlst2 = nn.ModuleList()

        for i in range(self.layers):
            self.revinlist1.append(RevIN(time_dimension, eps=1e-5,affine=False, subtract_last=False))
            self.revinlist2.append(RevIN(time_dimension, eps=1e-5,affine=False, subtract_last=False))
            self.ginbacklst.append(GINbackbone(time_dimension, hidden_dimnesion, hidden_dimnesion, self.order + i, self.support_len + 1, dropout=self.dropout))
            self.ginbacklst.append(GINbackbone(hidden_dimnesion, hidden_dimnesion, time_dimension, self.order + i, self.support_len + 1, dropout=self.dropout))
            self.layernormlst.append(LayerNormalization(time_dimension))

            self.ginconvlst.append(GINConv(time_dimension, hidden_dimnesion, hidden_dimnesion, self.order + i, self.support_len + 1, dropout=self.dropout))
            self.ginconvlst.append(
                GINConv(hidden_dimnesion, hidden_dimnesion, time_dimension,  self.order + i, self.support_len + 1, dropout=self.dropout))
            self.layernormlst2.append(LayerNormalization(time_dimension))



        self.dg0 = DFDgraph(time_dimension, self.hidden_emb, self.emb, t_n = self.time_count, mask = self.mask)
        self.dg1 = DFDgraph(time_dimension, self.hidden_emb, self.emb, t_n = self.time_count, mask = self.mask)
        self.d_res = DFDgraph(time_dimension, self.hidden_emb, self.emb, t_n = self.time_count, mask = self.mask)
        self.d_trend = DFDgraph(time_dimension, self.hidden_emb, self.emb, t_n = self.time_count, mask = self.mask)


        self.layersnorm = torch.nn.LayerNorm(normalized_shape= self.hidden_emb, eps=1e-08,
                                             elementwise_affine=False)
        kernel_size = [3, 7, 11]

        self.decompsition = series_decomp(kernel_size)

        self.phase_emb = nn.Parameter(torch.empty(38, self.emb), requires_grad=False)
        self.number_emb = nn.Parameter(
            torch.empty(10, self.hidden_emb), requires_grad=False)
        self.grids_emb = nn.Parameter(
            torch.empty(num_grids + 2, self.emb), requires_grad=False)
        self.grids_emb2 = nn.Parameter(
            torch.empty(num_grids + 3, self.emb), requires_grad=False)

        self.metagraph = MetaGraph2(time_dimension, self.hidden_emb, self.emb, 0.3, t_n = self.time_count, mask = self.mask)
        self.metagraph2 = MetaGraph2(time_dimension, self.hidden_emb, self.emb, 0.3, t_n = self.time_count, mask = self.mask)
        self.metagraphd = MetaGraph2(time_dimension, self.hidden_emb, self.emb, 0.3, t_n=self.time_count, mask = self.mask)
        self.metagraphd2 = MetaGraph2(time_dimension, self.hidden_emb, self.emb, 0.3, t_n=self.time_count, mask = self.mask)

        self.end_conv = nn.ModuleList([nn.Conv2d(in_channels=(self.layers + 1) * 2,
                                    out_channels=1,
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    bias=False) for _ in range(3)])

        self.classifier = Discriminator(self.emb, 32, 12, dropout=self.dropout)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)


    def forward(self, X : torch.Tensor, adj : torch.Tensor, unknown_nodes, location, batch_seen: int, epoch: int, train: bool, **kwargs):
        data = X
        X = X[:, :, :, 0]
        adj = torch.tensor(adj, dtype=torch.float32).to(X.device)

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

        X = X.permute(0, 2, 1)  # to correct the input dims
        diag = torch.diagonal(A_q, dim1=-2, dim2=-1)
        a_diag = torch.diag_embed(diag)
        A_q = A_q - a_diag
        A_h = A_h - a_diag
        if len(adj.shape) == 2:
            A_q = A_q.repeat(X.shape[0],1, 1)
            A_h = A_h.repeat(X.shape[0], 1, 1)
        if self.mask != 0:
            A_q = graphmasekd(A_q, self.mask)
            A_h = graphmasekd(A_h, self.mask)

        td = []
        if self.t_of_d:
            T_D = self.T_i_D_emb[(data[:, :, :, 1] * 288).type(torch.LongTensor)][:, -1, :, :]
            td.append(T_D)
        if self.h_of_d:
            H_D = self.H_i_D_emb[(data[:, :, :, 1] * 24).type(torch.LongTensor)][:, -1, :, :]
            td.append(H_D)
        # [B, L, N, d]
        if self.d_of_w:
            D_W = self.D_i_W_emb[(data[:, :, :, 1 + 1]).type(torch.LongTensor)][:, -1, :, :]
            td.append(D_W)
        if self.m_of_y:
            M_Y = self.M_i_Y_emb[(data[:, :, :, -1]).type(torch.LongTensor)][:, -1, :, :]
            td.append(M_Y)
        support0 = [A_q, A_h]
        td = torch.cat(td, dim=-1)

        X_mask = torch.zeros_like(X)
        X_mask[torch.arange(X.shape[0])[:, None], unknown_nodes, : ] = 1

        X_mask0 = torch.ones_like(X)
        X_mask0[torch.arange(X.shape[0])[:, None], unknown_nodes, :] = 0

        X_b = self.ginbackbone(X, support0)
        X_b = X_b * X_mask + X

        res_x = [X_b]

        x_res, x_trend = self.decompsition(X_b.permute(0, 2, 1))
        x_res, x_trend = x_res.permute(0, 2, 1), x_trend.permute(0, 2, 1)
        if not self.meta_data:
            adj_res = self.d_res(x_res, td)
            adj_trend = self.d_trend(x_trend, td)
            adj = self.dg0(x_res, td)
            diag = torch.diagonal(adj, dim1=-2, dim2=-1)
            a_diag = torch.diag_embed(diag)
            adj = adj - a_diag
            sup = [A_q, A_h]
            sup = sup + [adj]

            adj2 = self.dg1(x_res, td)
            sup2 = [A_q, A_h]
            sup2 = sup2 + [adj2]
        else:
            loc_num = location[0]
            loc = location[1]

            loc_num = loc_num.type(torch.LongTensor)

            loc_vecn = self.number_emb[loc_num.type(torch.LongTensor)].flatten(-2)


            grids_vec = self.grids_emb[loc.type(torch.LongTensor)].squeeze()

            metagraph = self.metagraph(x_res, grids_vec, loc_vecn, td, unknown_nodes)
            metagraph2 = self.metagraph2(x_trend, grids_vec, loc_vecn, td, unknown_nodes)

            adj_res = metagraph
            adj_trend = metagraph2

            adj_d0 = self.metagraphd(X_b, grids_vec, loc_vecn, td, unknown_nodes)
            adj_d1 = self.metagraphd2(X_b, grids_vec, loc_vecn, td, unknown_nodes)
            diag = torch.diagonal(adj_d0, dim1=-2, dim2=-1)
            a_diag = torch.diag_embed(diag)
            adj_d0 = adj_d0 - a_diag
            sup = [A_q, A_h]
            sup = sup + [adj_d0]


            sup2 = [A_q, A_h]
            sup2 = sup2 + [adj_d1]

        FFT_res = torch.fft.rfft(x_res, dim=-1, norm='ortho')
        amplitude_res = torch.abs(FFT_res)  # 形状为 (B, N, L)
        _, indics_res = torch.topk(amplitude_res, 5, dim=-1)


        phase_res = torch.angle(FFT_res)
        phase_res_ori = phase_res
        maskresones = torch.ones_like(phase_res_ori).to(phase_res.device)
        maskreszeros = torch.zeros_like(phase_res_ori).to(phase_res.device)
        maskreszeros[torch.arange(indics_res.shape[0])[:,None, None], :, indics_res] = 1
        maskresones[torch.arange(indics_res.shape[0])[:,None, None], :, indics_res] = 0
        phase_res = deal_with_phase(phase_res)

        phase_res = self.phase_emb[phase_res]

        phase_res = self.ginFFT(phase_res, [adj_res])
        phase_res = ((phase_res - torch.min(phase_res)) / (torch.max(phase_res) - torch.min(phase_res))) * 3.14

        phase_res = phase_res * maskreszeros + phase_res_ori.detach() * maskresones

        X_reconstructed_res = amplitude_res * torch.exp(1j * phase_res)
        x_reconstructed_res = torch.fft.irfft(X_reconstructed_res, dim=-1, norm='ortho')


        FFT_trend = torch.fft.rfft(x_trend, dim=-1, norm='ortho')
        amplitude_trend = torch.abs(FFT_trend)  # 形状为 (B, N, L)
        _, indics_trend = torch.topk(amplitude_trend, 5, dim=-1)

        phase_trend = torch.angle(FFT_trend)
        phase_trend_ori = phase_trend
        masktrendones = torch.ones_like(phase_trend_ori).to(phase_trend.device)
        masktrendzeros = torch.zeros_like(phase_trend_ori).to(phase_trend.device)
        masktrendzeros[torch.arange(indics_trend.shape[0])[:, None, None], :, indics_trend] = 1
        masktrendones[torch.arange(indics_trend.shape[0])[:, None, None], :, indics_trend] = 0


        phase_trend = deal_with_phase(phase_trend)
        phase_trend = self.phase_emb[phase_trend]
        phase_trend = self.ginFFT2(phase_trend, [adj_trend])
        phase_trend = ((phase_trend - torch.min(phase_trend)) / (torch.max(phase_trend) - torch.min(phase_trend))) * 3.14

        phase_trend = phase_trend * masktrendzeros + phase_trend_ori.detach() * masktrendones
        X_reconstructed_trend = amplitude_trend * torch.exp(1j * phase_trend)

        x_reconstructed_trend = torch.fft.irfft(X_reconstructed_trend, dim=-1, norm='ortho')

        X = x_reconstructed_res  * X_mask  + x_res * X_mask0+ x_reconstructed_trend * X_mask + x_trend * X_mask0
        x2 = X
        res_x.append(X)
        x_backbone = x2

        X_b = X
        X_r = X

        X = self.revin1(X, 'norm')

        for i in range(self.layers):
            X_b = self.revinlist1[i](X_b, 'norm')
            X_b = self.ginbacklst[2 * i](X_b, sup)
            X_b = self.ginbacklst[2 * i + 1](X_b, sup) * X_mask + X * X_mask0
            X_b = self.revinlist1[i](X_b, 'denorm')
            res_x.append(X_b)

        for i in range(self.layers):
            X_r = self.revinlist2[i](X_r, 'norm')
            X_r = self.ginconvlst[2 * i](X_r, sup2)
            X_r = self.ginconvlst[2 * i + 1](X_r, sup2) * X_mask + X * X_mask0
            X_r = self.revinlist2[i](X_r, 'denorm')
            res_x.append(X_r)

        result = torch.stack(res_x, dim=-1)

        output = []
        for i in range(len(self.end_conv)):
            self.end_conv[i].weight.data = torch.abs(self.end_conv[i].weight.data) / torch.sum(torch.abs(self.end_conv[i].weight.data))
            output.append(self.end_conv[i](result.permute(0, 3, 2, 1)))
        output = torch.concat(output, dim=1).permute(0, 3, 2, 1)

        output = torch.mean(output, dim=-1)


        if epoch <= 50 or not train:
            x_backbone = x_backbone.unfold(-1, self.emb, 1)
            x_backbone = GRL.apply(x_backbone, 1)
            node_class = self.classifier(x_backbone)
            return [output.permute(0, 2, 1), node_class]
        else:
            return [output.permute(0, 2, 1)]