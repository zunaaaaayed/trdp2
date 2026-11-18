"""Pre-training CT/Fourier mismatch and crop diagnostic on a separate known object."""

import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector, relative_l2
from experiments.worm_uncertainty.core.icecream_adapter import (
    native_axes,
    canonical_axes,
    make_trainer,
    load_official,
)
from experiments.worm_uncertainty.data.alternative_reconstruction import phantom
from experiments.worm_uncertainty.data.completion_data import acquisition_mask
from experiments.worm_uncertainty.studies.icecream_pilot import (
    fresh_stage,
    write_json,
    DEFAULT_ROOT,
)


def audit(root):
    folder, _ = fresh_stage(root, "operator_audit")
    tick = time.monotonic()
    _, _, measure = load_official()
    truth = phantom(901, 64, 19)  # separate implementation object, not pilot evaluation
    smooth_q = (np.arange(64) - 31.5) / 32
    z, y, x = np.meshgrid(smooth_q, smooth_q, smooth_q, indexing="ij")
    smooth = 0.03 * np.exp(-0.5 * ((x / 0.18) ** 2 + (y / 0.28) ** 2 + (z / 0.3) ** 2))
    op = ParallelProjector(64, np.arange(100) * 1.8)
    rows = []
    for name, value in [("smooth", smooth), ("mixed", truth)]:
        projections = op.forward(value)
        native = torch.from_numpy(native_axes(value)).float()
        for start, span in [(0, 180), (0, 120), (0, 90), (36, 120)]:
            # No network inference/training here; only its configured Fourier operator.
            trainer, _ = make_trainer(64, start, span)
            visible = trainer.wedge_input_set[0]
            ideal = measure(native[None], visible)[0]
            fbp = op.fbp(projections, acquisition_mask(start, span))
            crop = slice(16, 48)
            small, _ = make_trainer(32, start, span)
            crop_after = ideal[crop, crop, crop]
            after_crop = measure(native[crop, crop, crop][None], small.wedge_input_set[0])[0]
            rows.append(
                dict(
                    object=name,
                    start=start,
                    span=span,
                    ct_fbp_vs_ideal_relative_l2=relative_l2(fbp, canonical_axes(ideal.numpy())),
                    central_ct_fbp_vs_ideal_relative_l2=relative_l2(
                        fbp[crop, crop, crop], canonical_axes(ideal.numpy())[crop, crop, crop]
                    ),
                    crop_noncommutation_relative_l2=relative_l2(
                        after_crop.numpy(), crop_after.numpy()
                    ),
                    valid_rotations=len(trainer.k_sets),
                )
            )
    report = dict(
        rows=rows,
        seconds=time.monotonic() - tick,
        interpretation="Approximation errors, not exact-operator equality. Smooth and mixed independent implementation objects; no tuning to pilot truth. Geometric correctness is verified separately.",
    )
    write_json(folder / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    torch.set_num_threads(2)
    audit(p.parse_args().root)
