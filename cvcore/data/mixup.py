import numpy as np
import torch
import cv2

def mixup_data(x, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lamb = np.random.beta(alpha + 1., alpha)
    else:
        lamb = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lamb * x + (1 - lamb) * x[index, :]
    return mixed_x