"""Autograd equivalent of the existing fixed-grid parallel FBP; no target fitting."""

import numpy as np
import torch
from torch import nn
from scipy.fft import next_fast_len


class DifferentiableFBP(nn.Module):
    def __init__(self, operator, angle_step_degrees):
        super().__init__()
        self.size = operator.size
        self.detectors = operator.detectors
        self.nfft = next_fast_len(max(64, 2 * self.detectors))
        matrix = np.concatenate([a.toarray().T for a in operator.matrices], axis=1)
        self.register_buffer("backprojection", torch.from_numpy(matrix))
        ramp = 2 * np.abs(np.fft.rfftfreq(self.nfft, operator.du))
        self.register_buffer("ramp", torch.from_numpy(ramp))
        self.factor = np.deg2rad(angle_step_degrees) / 2 * operator.du / operator.spacing**2

    def forward(self, projections, mask):
        observed = torch.where(mask[..., None], projections, torch.zeros_like(projections))
        filtered = torch.fft.irfft(
            torch.fft.rfft(observed, n=self.nfft, dim=-1) * self.ramp.to(projections),
            n=self.nfft,
            dim=-1,
        )[..., : self.detectors]
        result = filtered.flatten(-2) @ self.backprojection.to(projections).T
        return result.reshape(*projections.shape[:-2], self.size, self.size) * self.factor


def completion_loss(prediction, measured, mask, reference, reconstruction, weight=0.0):
    """Worm auxiliary target is an FBP surrogate with identical measured noise."""
    if weight < 0:
        raise ValueError("nonnegative auxiliary weight required")
    completed = torch.where(mask[..., None], measured, prediction)
    missing = (~mask)[..., None].expand_as(prediction)
    count = missing.sum(dim=(-2, -1))
    if (count == 0).any():
        raise ValueError("missing-ray loss undefined for full acquisition")
    base = (((completed - reference).square() * missing).sum(dim=(-2, -1)) / count).mean()
    if weight == 0:
        return base
    target = torch.where(mask[..., None], measured, reference)
    full = torch.ones_like(mask)
    return (
        base
        + weight * (reconstruction(completed, full) - reconstruction(target, full)).abs().mean()
    )
