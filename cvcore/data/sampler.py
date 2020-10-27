import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

def class_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    sample_weights = np.zeros_like(labels).astype(np.float32)
    for idx, label in enumerate(labels):
        sample_weights[idx] = total_samples / class_counts[label]
    # return sample_weights
    sampler = WeightedRandomSampler(weights=sample_weights,
        num_samples=total_samples)
    return sampler