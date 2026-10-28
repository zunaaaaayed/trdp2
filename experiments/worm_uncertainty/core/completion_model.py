"""Small deterministic sinogram CNN and finite-grid context checks."""

import numpy as np
import torch
from torch import nn


class CompletionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = [7, 16, 16, 16, 1]
        for i, dilation in enumerate([1, 2, 4, 8]):
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=dilation, dilation=dilation)
            )
            if i < 3:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)[:, 0]


def angular_connectivity(count=25):
    """Boolean paths through actual finite zero-padded layers; not infinite RF alone."""
    access = np.eye(count, dtype=bool)
    for dilation in [1, 2, 4, 8]:
        next_access = np.zeros_like(access)
        for row in range(count):
            for neighbour in [row - dilation, row, row + dilation]:
                if 0 <= neighbour < count:
                    next_access[row] |= access[neighbour]
        access = next_access
    return access
