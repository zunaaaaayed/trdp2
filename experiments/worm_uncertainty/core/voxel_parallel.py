"""Pixel-driven parallel CT with an exact discrete transpose and fixed-grid FBP.

Arrays are (z,y,x) and (z,angle,detector). Coordinates are pixel centres about
(N-1)/2; u=x*cos(theta)-y*sin(theta). Linear detector splatting is a declared
discretization, not the unknown worm simulator. All lengths use the same units.
"""

import numpy as np
from scipy import sparse
from scipy.fft import rfft, irfft, rfftfreq, next_fast_len


class ParallelProjector:
    def __init__(self, size, angles, spacing=1.0, detector_count=None, detector_spacing=None):
        self.size = int(size)
        self.angles = np.asarray(angles, dtype=float)
        self.spacing = float(spacing)
        self.detectors = int(detector_count or size)
        self.du = float(detector_spacing or spacing)
        if self.size < 4 or self.spacing <= 0 or self.du <= 0:
            raise ValueError("positive physical grid required")
        q = (np.arange(size) - (size - 1) / 2) * self.spacing
        self.y, self.x = np.meshgrid(q, q, indexing="ij")
        self.support = self.x**2 + self.y**2 < (size * self.spacing / 2) ** 2
        pixels = np.flatnonzero(self.support)
        self.matrices = []
        for angle in np.deg2rad(self.angles):
            t = (
                self.x.ravel()[pixels] * np.cos(angle) - self.y.ravel()[pixels] * np.sin(angle)
            ) / self.du
            t += (self.detectors - 1) / 2
            left = np.floor(t).astype(int)
            frac = t - left
            row = np.concatenate([left, left + 1])
            col = np.tile(pixels, 2)
            val = np.concatenate([1 - frac, frac]) * self.spacing**2 / self.du
            valid = (row >= 0) & (row < self.detectors)
            self.matrices.append(
                sparse.csr_matrix(
                    (val[valid], (row[valid], col[valid])), shape=(self.detectors, size * size)
                )
            )

    def forward(self, volume):
        volume = np.asarray(volume)
        if volume.ndim != 3 or volume.shape[1:] != (self.size, self.size):
            raise ValueError("expected z,y,x volume")
        flat = volume.reshape(len(volume), -1).T
        return np.stack([(a @ flat).T for a in self.matrices], axis=1)

    def adjoint(self, projections):
        projections = np.asarray(projections)
        if projections.ndim != 3 or projections.shape[1:] != (len(self.angles), self.detectors):
            raise ValueError("expected z,angle,detector measurements")
        result = np.zeros(
            (self.size * self.size, len(projections)), dtype=np.result_type(projections, float)
        )
        for i, a in enumerate(self.matrices):
            result += a.T @ projections[:, i].T
        return result.T.reshape(-1, self.size, self.size)

    def fbp(self, observed, mask, angle_step_degrees=1.8):
        """Zero unobserved angles; retain original angular quadrature (no pi/span gain)."""
        observed = np.where(np.asarray(mask)[None, :, None], observed, 0)
        nfft = next_fast_len(max(64, 2 * self.detectors))
        ramp = 2 * np.abs(rfftfreq(nfft, self.du))
        filtered = irfft(rfft(observed, n=nfft, axis=-1) * ramp, n=nfft, axis=-1)[
            ..., : self.detectors
        ]
        return (
            self.adjoint(filtered)
            * (np.deg2rad(angle_step_degrees) / 2)
            * self.du
            / self.spacing**2
        )

    def sart(self, observed, mask, sweeps=5, relaxation=0.5):
        """Sequential angle-block SART; explicit row/column normalization and positivity."""
        ids = np.flatnonzero(mask)
        x = np.zeros((self.size * self.size, len(observed)))
        history = []
        for _ in range(sweeps):
            for i in ids:
                a = self.matrices[i]
                row = np.asarray(a.sum(axis=1)).ravel()
                col = np.asarray(a.sum(axis=0)).ravel()
                residual = observed[:, i].T - a @ x
                residual = np.divide(
                    residual, row[:, None], out=np.zeros_like(residual), where=row[:, None] > 1e-12
                )
                update = a.T @ residual
                x += relaxation * np.divide(
                    update, col[:, None], out=np.zeros_like(update), where=col[:, None] > 1e-12
                )
                np.maximum(x, 0, out=x)
            history.append(
                float(
                    np.linalg.norm(
                        self.forward(x.T.reshape(-1, self.size, self.size))[:, mask]
                        - observed[:, mask]
                    )
                )
            )
        return x.T.reshape(-1, self.size, self.size), history


def relative_l2(value, target):
    den = np.linalg.norm(np.asarray(target).ravel())
    return float(np.linalg.norm((np.asarray(value) - target).ravel()) / den) if den else None


def reconstruction_metrics(volume, clean, observed_mean, mask, operator, truth=None):
    predicted = operator.forward(volume)
    out = dict(
        acquired_clean_relative_l2=relative_l2(predicted[:, mask], clean[:, mask]),
        acquired_noisy_relative_l2=relative_l2(predicted[:, mask], observed_mean[:, mask]),
        acquired_noisy_rmse=float(
            np.sqrt(np.mean((predicted[:, mask] - observed_mean[:, mask]) ** 2))
        ),
        withheld_relative_l2=relative_l2(predicted[:, ~mask], clean[:, ~mask])
        if (~mask).any()
        else None,
        volume_relative_l2=None,
        negative_voxel_fraction=float(np.mean(volume < 0)),
    )
    if truth is not None:
        fg = truth > 0.1 * truth.max()
        out.update(
            volume_relative_l2=relative_l2(volume, truth),
            volume_rmse=float(np.sqrt(np.mean((volume - truth) ** 2))),
            foreground_relative_l2=relative_l2(volume[fg], truth[fg]),
        )
    return out, predicted
