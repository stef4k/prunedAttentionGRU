import torch
import numpy as np

def mixup(data, targets, alpha):
    rand_index = torch.randperm(data.size(0), device=data.device)
    data_a = data
    data_b = data[rand_index]
    target_a = targets
    target_b = targets[rand_index]

    lam = torch.tensor(np.random.beta(alpha, alpha), device=data.device, dtype=data.dtype)
    data = data_a * lam + data_b * (1 - lam)
    return data, target_a, target_b, lam
