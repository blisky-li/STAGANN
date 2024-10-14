import torch
import numpy as np


def masked_binary(preds: torch.Tensor, labels: torch.Tensor, fusion=True, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """
    preds = torch.squeeze(preds)
    labels = labels.unsqueeze(-1).repeat(1, 1, preds.shape[2])
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    #print(preds)
    #preds = preds * mask
    #print(preds)
    preds = torch.clamp(preds, min=1e-7, max=1 - 1e-7)
    #print(preds.shape, labels.shape, preds.device, labels.device)
    bceloss = torch.nn.BCELoss()
    #print()
    #labels = labels.unsqueeze(-1).repeat(1, 1, preds.shape[2])

    # 生成一个与x形状相同的二值mask
    if fusion:
        p = 0.2
        mask2 = torch.rand_like(labels) < p
        # 使用mask翻转x
        labels[mask2] = 1 - labels[mask2]
    #print(labels.shape)
    try:
        loss = bceloss(preds, labels)
        a = loss.mean()
    #loss = -(labels * torch.log(preds) + 3 * (1 - labels) * torch.log(1 - preds)).mean()
    #print(loss.mean())
    except Exception as e:
        a = np.array(0)
    return a