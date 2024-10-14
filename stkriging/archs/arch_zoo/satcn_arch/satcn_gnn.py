import math
import torch
import numpy as np
EPS = 1e-5

# masked spatial aggregation, adj is T x N x N
# each aggregator is a function taking as input X (B x T x N x Din), adj (B x T x N x N), device
# returning the aggregated value of X (B x T x N x Din) for each dimension

def aggregate_mean(X, adj, device = 'cpu'):
    adj_ = torch.sign(adj)
    D = torch.sum(adj_, -1, keepdim=True) + EPS

    X_sum =  torch.einsum("btji,boj->btoi", [X, adj_])
    #print(X_sum.shape,D.shape)
    X_mean = torch.einsum("btoi,boi->btoi",X_sum,1/D)


    return X_mean

def aggregate_normalised_mean(X, adj, device='cpu'):
    # D^{-1} A * X    i.e. the mean of the neighbours
    D = torch.sum(adj, -1, keepdim=True)  + EPS

    X_sum =  torch.einsum("btji,boj->btoi", [X, adj])
    X_mean = torch.einsum("btoi,boi->btoi",X_sum,1/D)
    return X_mean

def aggregate_d(X, adj, device='cpu'):
    # D^{-1} A * X    i.e. the mean of the neighbours
    (B, ST, N, D) = X.shape
    P = torch.ones([B, ST, N, 1]).cuda()
    adj_ = torch.sign(adj)
    D = torch.sum(adj_, -1, keepdim=True) + EPS
#    rD = torch.mul(torch.pow(torch.sum(adj, -1, keepdim=True), -0.5), torch.eye(N, device=device))  # D^{-1/2]
#    adj = torch.matmul(torch.matmul(rD, adj), rD)
    X_sum =  torch.einsum("btji,boj->btoi", [P, adj])
    X_mean = torch.einsum("btoi,boi->btoi",X_sum,1/D)
    return X_mean

def aggregate_d_var(X, adj, device='cpu'):
    # relu(D^{-1} A X^2 - (D^{-1} A X)^2)     i.e.  the variance of the features of the neighbours

    (B, ST, N, D) = X.shape
    P = torch.ones([B, ST, N, 1]).cuda()
    D = torch.sum(adj, -1, keepdim=True) + EPS

    X_sum = torch.einsum("btji,boj->btoi", [P * P, adj])
    X_sum = torch.einsum("btoi,boi->btoi",X_sum,1/D)
    X_mean = aggregate_mean(X, adj)  # D^{-1} A X
    var = torch.relu(X_sum - X_mean * X_mean)  # relu(mean_squares_X - mean_X^2)
    return var

def aggregate_d_std(X, adj, device='cpu'):
    # sqrt(relu(D^{-1} A X^2 - (D^{-1} A X)^2) + EPS)     i.e.  the standard deviation of the features of the neighbours
    # the EPS is added for the stability of the derivative of the square root
    std = torch.sqrt(aggregate_d_var(X, adj, device) + EPS)  # sqrt(mean_squares_X - mean_X^2)
    return std

def aggregate_max(X, adj, min_value=-math.inf, device='cpu'): #softmax is better

    (B, ST, N, _) = X.shape
    adj = adj.unsqueeze(-1)
    X = X.unsqueeze(-2).repeat(1, 1, 1, N, 1).permute([0, 1, 3, 2, 4])
    M = torch.where(adj > 0.0, X, torch.tensor(min_value, device=device))
    max = torch.max(M, -2)[0]
    return max

def aggregate_min(X, adj, max_value=math.inf, device='cpu'): #softmin is better

    (B, ST, N, _) = X.shape
    adj = adj.unsqueeze(-1)
    X = X.unsqueeze(-2).repeat(1, 1, 1, N, 1).permute([0, 1, 3, 2, 4])
    M = torch.where(adj > 0.0, X, torch.tensor(max_value, device=device))
    min = torch.min(M, -2)[0]
    return min

def aggregate_var(X, adj, device='cpu'):
    # relu(D^{-1} A X^2 - (D^{-1} A X)^2)     i.e.  the variance of the features of the neighbours

    D = torch.sum(adj, -1, keepdim=True) + EPS

    X_sum = torch.einsum("btji,boj->btoi", [X * X, adj])

    X_sum = torch.einsum("btoi,boi->btoi",X_sum,1/D)

    X_mean = aggregate_mean(X, adj)  # D^{-1} A X
    var = torch.relu(X_sum - X_mean * X_mean)  # relu(mean_squares_X - mean_X^2)
    return var

def aggregate_std(X, adj, device='cpu'):
    # sqrt(relu(D^{-1} A X^2 - (D^{-1} A X)^2) + EPS)     i.e.  the standard deviation of the features of the neighbours
    # the EPS is added for the stability of the derivative of the square root
    std = torch.sqrt(aggregate_var(X, adj, device) + EPS)  # sqrt(mean_squares_X - mean_X^2)
    return std

def aggregate_sum(X, adj, device='cpu'):
    # A * X    i.e. the mean of the neighbours

    X_sum =  torch.einsum("btji,boj->btoi", [X, adj])
    return X_sum

def aggregate_softmax(X, adj, device='cpu'):
    # for each node sum_i(x_i*exp(x_i)/sum_j(exp(x_j)) where x_i and x_j vary over the neighbourhood of the node
    X_sum =  torch.einsum("btji,boj->btoi", [X, adj])
    softmax = torch.nn.functional.softmax(X_sum, dim = 2)
    return softmax

def aggregate_softmin(X, adj, device='cpu'):
    # for each node sum_i(x_i*exp(-x_i)/sum_j(exp(-x_j)) where x_i and x_j vary over the neighbourhood of the node
    return -aggregate_softmax(-X, adj, device=device)

AGGREGATORS_MASK = {'mean': aggregate_mean, 'sum': aggregate_sum,
               'std': aggregate_std, 'var': aggregate_var,'max': aggregate_max, 'min': aggregate_min,
               'normalised_mean': aggregate_normalised_mean, 'softmax': aggregate_softmax, 'softmin': aggregate_softmin, 'distance': aggregate_d, 'd_std': aggregate_d_std}

AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max, 'min': aggregate_min,
               'std': aggregate_std, 'var': aggregate_var,
               'normalised_mean': aggregate_normalised_mean, 'softmax': aggregate_softmax, 'softmin': aggregate_softmin, 'distance': aggregate_d, 'd_std': aggregate_d_std}


def scale_identity(X, adj, avg_d=None):
    return X


def scale_amplification(X, adj, avg_d=None):
    # log(D + 1) / d * X     where d is the average of the ``log(D + 1)`` in the training set

    D = torch.sum(adj, -1) + EPS
    avg_d = torch.tensor(torch.mean(torch.log(torch.sum(torch.abs(adj), dim = 0) + 1)))
    scale = (torch.log(D + 1) / avg_d).unsqueeze(-1)
    #print(scale.shape, X.shape)
    #X_scaled = torch.mul(scale, X)
    X_scaled = torch.einsum("bni,btnk->btnk", scale, X)
    return X_scaled


def scale_attenuation(X, adj, avg_d=None):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    D = torch.sum(adj, -1) + EPS
    scale = (avg_d["log"] / torch.log(D + 1)).unsqueeze(-1)
    X_scaled = torch.mul(scale, X)
    return X_scaled


def scale_linear(X, adj, avg_d=None):
    # d^{-1} D X     where d is the average degree in the training set
    D = torch.sum(adj, -1, keepdim=True) + EPS
    X_scaled = D * X / avg_d["lin"]
    return X_scaled


def scale_inverse_linear(X, adj, avg_d=None):
    # d D^{-1} X     where d is the average degree in the training set
    D = torch.sum(adj, -1, keepdim=True) + EPS
    X_scaled = avg_d["lin"] * X / D
    return X_scaled


SCALERS = {'identity': scale_identity, 'amplification': scale_amplification, 'attenuation': scale_attenuation,
           'linear': scale_linear, 'inverse_linear': scale_inverse_linear}