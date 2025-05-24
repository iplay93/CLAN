import numpy as np
import torch
from tsaug import *
import random

class Scale:
    """
    Scale: Applies random multiplicative scaling to each channel of a time-series.

    This augmentation simulates amplitude variation.
    Reference: https://arxiv.org/pdf/1706.00527.pdf

    Args:
        sigma (float): Standard deviation of the normal distribution.
        loc (float): Mean of the normal distribution (controls average scaling).
    """
    def __init__(self, sigma: float = 1.1, loc: float = 1.3):
        self.sigma = sigma
        self.loc = loc

    def augment(self, x: np.ndarray) -> np.ndarray:
        """
        Applies scaling to each time-step in the input sequence.

        Args:
            x (np.ndarray): Input array of shape (B, T, C)

        Returns:
            np.ndarray: Scaled array of shape (B, T, C)
        """
        B, T, C = x.shape
        factor = np.random.normal(loc=self.loc, scale=self.sigma, size=(B, C))  # (B, C)

        augmented = []
        for t in range(T):
            xt = x[:, t, :]                      # (B, C)
            xt_scaled = xt * factor              # Apply scaling factor
            augmented.append(xt_scaled[:, np.newaxis, :])  # (B, 1, C)

        return np.concatenate(augmented, axis=1)  # (B, T, C)


class Permute:
    """
    Permute: Temporally shuffles segments of a time-series.

    This augmentation introduces temporal disorder by randomly rearranging time segments.

    Args:
        min_segments (int): Minimum number of segments.
        max_segments (int): Maximum number of segments.
        seg_mode (str): 'random' for random split points, otherwise uniform split.
    """
    def __init__(self, min_segments: int = 2, max_segments: int = 15, seg_mode: str = "random"):
        self.min = min_segments
        self.max = max_segments
        self.seg_mode = seg_mode

    def augment(self, x: np.ndarray) -> np.ndarray:
        """
        Applies segment-wise permutation to the sequence.

        Args:
            x (np.ndarray): Input array of shape (B, T, C)

        Returns:
            np.ndarray: Permuted array of shape (B, T, C)
        """
        B, T, C = x.shape
        time_indices = np.arange(T)
        segment_counts = np.random.randint(self.min, self.max, size=B)

        augmented = np.zeros_like(x)

        for i in range(B):
            if segment_counts[i] > 1:
                if self.seg_mode == "random":
                    split_points = np.random.choice(T - 2, segment_counts[i] - 1, replace=False)
                    split_points.sort()
                    segments = np.split(time_indices, split_points)
                else:
                    segments = np.array_split(time_indices, segment_counts[i])

                permuted_indices = np.concatenate(np.random.permutation(segments)).ravel()
                augmented[i] = x[i, permuted_indices, :]
            else:
                augmented[i] = x[i]

        return augmented


def select_transformation(aug_method: str, seq_len: int):
    """
    Returns the augmentation object corresponding to the selected method.

    Args:
        aug_method (str): Name of the augmentation method.
        seq_len (int): Sequence length (not used in all transforms).

    Returns:
        An augmentation object with an `.augment()` method.

    Raises:
        ValueError: If the method name is not recognized.
    """
    if aug_method == 'AddNoise':
        return AddNoise(scale=0.01)

    elif aug_method == 'Convolve':
        return Convolve(window="flattop", size=11)

    elif aug_method == 'Crop':
        # Approximated via permutation of short segments
        return Permute(min_segments=1, max_segments=5, seg_mode="random")

    elif aug_method == 'Drift':
        return Drift(max_drift=0.7, n_drift_points=5)

    elif aug_method == 'Dropout':
        return Dropout(p=0.1, fill=0)

    elif aug_method == 'Pool':
        return Pool(kind='max', size=4)

    elif aug_method == 'Quantize':
        return Quantize(n_levels=20)

    elif aug_method == 'Resize':
        return Scale(sigma=1.1, loc=2.0)

    elif aug_method == 'Reverse':
        return Reverse()

    elif aug_method == 'TimeWarp':
        return TimeWarp(n_speed_change=5, max_speed_ratio=3)

    else:
        raise ValueError(f"Unsupported augmentation method: {aug_method}")
