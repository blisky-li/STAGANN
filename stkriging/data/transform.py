import pickle

import torch
import numpy as np

from .registry import SCALER_REGISTRY


@SCALER_REGISTRY.register()
def standard_transform(data: np.array, output_dir: str, train_index: list, seq_len: int, norm_each_channel: int = False) -> np.array:
    """Standard normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.
        seq_len (int): sequence length.
        norm_each_channel (bool): whether to normalization each channel.

    Returns:
        np.array: normalized raw time series data.
    """

    # data: L, N, C, C=1
    data_train = data[:train_index[-1][1], ...]
    print("data_train_here！", train_index[-1][1])
    if norm_each_channel:
        mean, std = data_train.mean(axis=0, keepdims=True), data_train.std(axis=0, keepdims=True)
    else:
        mean, std = data_train[..., 0].mean(), data_train[..., 0].std()

    print("mean (training data):", mean)
    print("std (training data):", std)
    scaler = {}
    scaler["func"] = re_standard_transform.__name__
    scaler["args"] = {"mean": mean, "std": std}
    # label to identify the scaler for different settings.
    with open(output_dir + "/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        return (x - mean) / std

    data_norm = normalize(data)
    return data_norm


@SCALER_REGISTRY.register()
def re_standard_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:
    """Standard re-transformation.

    Args:
        data (torch.Tensor): input data.

    Returns:
        torch.Tensor: re-scaled data.
    """

    mean, std = kwargs["mean"], kwargs["std"]
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).type_as(data).to(data.device).unsqueeze(0)
        std = torch.from_numpy(std).type_as(data).to(data.device).unsqueeze(0)
    data = data * std
    data = data + mean
    return data


@SCALER_REGISTRY.register()
def min_max_transform(data: np.array, output_dir: str, train_index: list, seq_len: int) -> np.array:
    """Min-max normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.
        seq_len (int): sequence length.

    Returns:
        np.array: normalized raw time series data.
    """

    # L, N, C, C=1
    data_train = data[:train_index[-1][1], ...]
    print("data_train_here！", train_index[-1][1])
    min_value = data_train.min(axis=(0, 1), keepdims=False)[0]
    max_value = data_train.max(axis=(0, 1), keepdims=False)[0]

    print("min: (training data)", min_value)
    print("max: (training data)", max_value)
    scaler = {}
    scaler["func"] = re_min_max_transform.__name__
    scaler["args"] = {"min_value": min_value, "max_value": max_value}
    # label to identify the scaler for different settings.
    # To be fair, only one transformation can be implemented per dataset.
    # TODO: Therefore we (for now) do not distinguish between the data produced by the different transformation methods.
    with open(output_dir + "/scaler2.pkl", "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        # ref:
        # https://github.com/guoshnBJTU/ASTGNN/blob/f0f8c2f42f76cc3a03ea26f233de5961c79c9037/lib/utils.py#L17
        x = 1. * (x - min_value) / (max_value - min_value)
        x = 2. * x - 1.
        return x

    data_norm = normalize(data)
    return data_norm


@SCALER_REGISTRY.register()
def re_min_max_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:
    """Standard re-min-max transform.

    Args:
        data (torch.Tensor): input data.

    Returns:
        torch.Tensor: re-scaled data.
    """

    min_value, max_value = kwargs["min_value"], kwargs["max_value"]
    # ref:
    # https://github.com/guoshnBJTU/ASTGNN/blob/f0f8c2f42f76cc3a03ea26f233de5961c79c9037/lib/utils.py#L23
    data = (data + 1.) / 2.
    data = 1. * data * (max_value - min_value) + min_value
    return data

@SCALER_REGISTRY.register()
def logarithm_standard_transform(data: np.array, output_dir: str, train_index: list, seq_len: int, norm_each_channel: int = False) -> np.array:
    """Standard normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.
        seq_len (int): sequence length.
        norm_each_channel (bool): whether to normalization each channel.

    Returns:
        np.array: normalized raw time series data.
    """

    # data: L, N, C, C=1
    base = np.exp(1)
    data = np.log(data + 1)
    data_train = data[:train_index[-1][1], ...]
    print("data_train_here！", train_index[-1][1])

    print("Base (training data):", 10)
    if norm_each_channel:
        mean, std = data_train.mean(axis=0, keepdims=True), data_train.std(axis=0, keepdims=True)
    else:
        mean, std = data_train[..., 0].mean(), data_train[..., 0].std()

    print("mean (training data):", mean)
    print("std (training data):", std)

    scaler = {}
    scaler["func"] = re_logarithm_standard_transform.__name__
    scaler["args"] = {"base": base, "mean": mean, "std": std}
    # label to identify the scaler for different settings.
    with open(output_dir + "/scaler3.pkl", "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        return (x - mean) / std

    data_norm = normalize(data)

    return data_norm

@SCALER_REGISTRY.register()
def re_logarithm_standard_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:
    base = kwargs["base"]
    mean, std = kwargs["mean"], kwargs["std"]
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).type_as(data).to(data.device).unsqueeze(0)
        std = torch.from_numpy(std).type_as(data).to(data.device).unsqueeze(0)
    data = data * std
    data = data + mean
    data = torch.pow(torch.tensor(base).type_as(data).to(data.device), data) - 1
    return data

@SCALER_REGISTRY.register()
def logarithm_min_max_transform(data: np.array, output_dir: str, train_index: list, seq_len: int) -> np.array:


    # data: L, N, C, C=1
    base = np.exp(1)
    data = np.log(data + 1)
    data_train = data[:train_index[-1][1], ...]
    print("data_train_here！", train_index[-1][1])


    print("Base (training data):", 10)
    min_value = data_train.min(axis=(0, 1), keepdims=False)[0]
    max_value = data_train.max(axis=(0, 1), keepdims=False)[0]

    print("min: (training data)", min_value)
    print("max: (training data)", max_value)

    scaler = {}
    scaler["func"] = re_logarithm_min_max_transform.__name__
    scaler["args"] = {"base": base, "min_value": min_value, "max_value": max_value}
    # label to identify the scaler for different settings.
    with open(output_dir + "/scaler4.pkl", "wb") as f:
        pickle.dump(scaler, f)

    def normalize(x):
        # ref:
        # https://github.com/guoshnBJTU/ASTGNN/blob/f0f8c2f42f76cc3a03ea26f233de5961c79c9037/lib/utils.py#L17
        x = 1. * (x - min_value) / (max_value - min_value)
        x = 2. * x - 1.
        return x

    data_norm = normalize(data)

    return data_norm

@SCALER_REGISTRY.register()
def re_logarithm_min_max_transform(data: torch.Tensor, **kwargs) -> torch.Tensor:
    base = kwargs["base"]
    min_value, max_value = kwargs["min_value"], kwargs["max_value"]
    # ref:
    # https://github.com/guoshnBJTU/ASTGNN/blob/f0f8c2f42f76cc3a03ea26f233de5961c79c9037/lib/utils.py#L23
    data = (data + 1.) / 2.
    data = 1. * data * (max_value - min_value) + min_value

    data = torch.pow(torch.tensor(base).type_as(data).to(data.device), data) - 1
    return data