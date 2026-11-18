"""Matched measured-only refinement contract for three initialization choices."""

import copy
import numpy as np
from experiments.worm_uncertainty.studies.worm_fit_controls import (
    multiscale_initialize,
    align_to_reference,
)


def matched_contracts(measured, mask, angles, uv, reference, cfg):
    """Independent copies of exactly one observation/prior/budget contract."""
    y = measured[mask].ravel()
    return {
        arm: dict(
            y=y.copy(),
            angles=angles[mask].copy(),
            uv=uv.copy(),
            reference=reference.copy(),
            cfg=copy.deepcopy(cfg),
        )
        for arm in ["acquired", "interpolation", "learned"]
    }


def initialize_arms(measured, interpolation, learned, mask, angles, uv, v, cfg, design):
    # All arguments are completion outputs with measured values already reinserted.
    for value in [interpolation, learned]:
        if not np.array_equal(value[mask], measured[mask]):
            raise ValueError("acquired observations differ")
    baseline, info = multiscale_initialize(measured[mask].ravel(), angles[mask], uv, v, cfg, design)
    initializers = {"acquired": baseline}
    infos = {"acquired": info}
    for arm, value in [("interpolation", interpolation), ("learned", learned)]:
        initial, info = multiscale_initialize(value.ravel(), angles, uv, v, cfg, design)
        initializers[arm] = align_to_reference(initial, baseline)
        infos[arm] = info
    return initializers, infos, matched_contracts(measured, mask, angles, uv, baseline, cfg)
