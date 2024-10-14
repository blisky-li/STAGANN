import torch
import numpy as np


def R2(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked mean absolute percentage error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.
                                    Zeros in labels will lead to inf in mape. Therefore, null_val is set to 0.0 by default.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values

    labels = torch.where(torch.abs(labels) < 1e-4, torch.zeros_like(labels), labels)
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))

    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)




    #SSR = (preds - torch.mean(labels)) * (preds - torch.mean(labels))
    SSE = (labels - preds) * (labels - preds)
    #SSR = SSR * mask
    SSE = SSE * mask
    #SSR = torch.sum(SSR)
    SSE = torch.sum(SSE)
    #SST = SSR + SSE
    SST = (labels - torch.mean(labels)) ** 2
    SST = torch.sum(SST)
    return 1 - (SSE/SST)
