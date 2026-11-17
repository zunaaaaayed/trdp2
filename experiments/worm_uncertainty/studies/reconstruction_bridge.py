"""Frozen, small V3 completion-to-FBP bridge; CPU execution, compact artifacts."""

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector, relative_l2
from experiments.worm_uncertainty.core.completion_model import CompletionCNN
from experiments.worm_uncertainty.data.acquisition_diversity import (
    read_design,
    mask_hash,
    noisy_bank,
)
from experiments.worm_uncertainty.data.matched_completion import load_clean
from experiments.worm_uncertainty.data.completion_data import (
    acquisition_mask,
    channels,
    interpolate_missing,
)
from experiments.worm_uncertainty.data.prepare import sha256


def run(root, data, additional=False):
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    d, manifest = read_design()
    names = d["roles"]["selection"][3:6] if additional else d["roles"]["selection"][:3]
    checkpoint = Path(
        "output/worm_uncertainty/acquisition_diversity_v3/primary/B_2026102901/best.pt"
    )
    recorded = json.loads(checkpoint.with_name("report.json").read_text())["checkpoint_sha256"]
    assert sha256(checkpoint) == recorded
    model = CompletionCNN().eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    assert sum(p.numel() for p in model.parameters()) == 5809
    angles = np.arange(25) * 7.2
    masks = []
    previous = {r["hash"] for rows in d["inventories"].values() for r in rows}
    for index, span in [(24, 90), (19, 120)] if additional else [(9, 90), (19, 90), (9, 120)]:
        mask = acquisition_mask(index * 7.2, span, angles)
        assert mask_hash(mask) not in previous
        masks.append(
            dict(
                start=index * 7.2,
                span=span,
                mask=mask.tolist(),
                hash=mask_hash(mask),
                count=int(mask.sum()),
            )
        )
    config = dict(
        run_id=root.name,
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        files=names,
        exposure="previous V3 selection deformations; new masks absent from all V3 inventories",
        masks=masks,
        checkpoint=str(checkpoint),
        checkpoint_sha256=recorded,
        primary="single frozen B seed 2026102901",
        sampling=d["sampling"],
        normalization=d["normalization"],
        noise_seed=d["noise_seeds"]["selection"],
        noise_sd=0.05,
        reconstruction=dict(
            size=62,
            spacing_um=5,
            detector_spacing_um=5,
            origin="symmetric pixel centres",
            sart_sweeps=5,
            relaxation=0.5,
        ),
        gates=dict(gain=0.02, residual_degradation=0.01, majority=True),
        calibration=dict(full_l2_max=0.15, scale_error_max=0.1, adjoint_relative_max=1e-11),
        elapsed_cap_seconds=600,
        artifact_cap_bytes=250 * 1024**2,
        quota_override="user authorized proceeding 2026-11-16; personal quota unknown",
    )
    (root / "protocol.json").write_text(json.dumps(config, indent=2))
    (root / "dirty.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    (root / "runner.py").write_bytes(Path(__file__).read_bytes())
    op = ParallelProjector(62, angles, spacing=5)
    truth = np.exp(-0.5 * (((op.x - 20) / 25) ** 2 + ((op.y + 15) / 40) ** 2))[None] * 0.01
    clean = op.forward(truth)
    fbp = op.fbp(clean, np.ones(25, bool), 7.2)
    rng = np.random.default_rng(42)
    x = rng.normal(size=truth.shape)
    y = rng.normal(size=clean.shape)
    lhs = np.vdot(op.forward(x), y)
    rhs = np.vdot(x, op.adjoint(y))
    calibration = dict(
        full_l2=relative_l2(fbp, truth),
        scale=float(np.vdot(fbp, truth) / np.vdot(truth, truth)),
        adjoint_relative=float(abs(lhs - rhs) / max(abs(lhs), abs(rhs))),
    )
    (root / "calibration.json").write_text(json.dumps(calibration, indent=2))
    if (
        calibration["full_l2"] > 0.15
        or abs(calibration["scale"] - 1) > 0.1
        or calibration["adjoint_relative"] > 1e-11
    ):
        raise RuntimeError("synthetic geometry gate failed; no worm scoring")
    # Verify reversal and repeated deterministic SART before outcomes.
    np.testing.assert_allclose(
        ParallelProjector(62, angles + 180, spacing=5).forward(truth), clean[..., ::-1], atol=1e-12
    )
    control_mask = np.array(masks[0]["mask"])
    a, _ = op.sart(clean, control_mask)
    b, _ = op.sart(clean, control_mask)
    np.testing.assert_array_equal(a, b)
    cfg = dict(d, subsets={"validation": names})
    references = load_clean(data, cfg, manifest, "validation")
    noisy = noisy_bank(references, d, names, d["noise_seeds"]["selection"], 0)
    records = []
    for fi, name in enumerate(names):
        for mi, r in enumerate(masks):
            if time.monotonic() - started > 600:
                raise TimeoutError("bridge wall cap")
            mask = np.array(r["mask"])
            observed = np.where(mask[None, :, None], noisy[fi], 0)
            acquired_hash = hashlib.sha256(observed[:, mask].tobytes()).hexdigest()
            inputs = np.stack([channels(v, mask, r["start"], r["span"]) for v in observed])
            with torch.no_grad():
                prediction = model(torch.from_numpy(inputs)).numpy()
            learned = np.where(mask[None, :, None], observed, prediction)
            interpolation = np.stack([interpolate_missing(v, mask) for v in observed])
            interpolation = np.where(mask[None, :, None], observed, interpolation)
            for filled in [learned, interpolation]:
                np.testing.assert_array_equal(filled[:, mask], observed[:, mask])
            volumes = {
                "acquired_fbp": op.fbp(observed, mask, 7.2),
                "sart": op.sart(observed, mask)[0],
                "interpolation_fbp": op.fbp(interpolation, np.ones(25, bool), 7.2),
                "learned_fbp": op.fbp(learned, np.ones(25, bool), 7.2),
            }
            for arm, volume in volumes.items():
                projected = op.forward(volume)
                filled = {"learned_fbp": learned, "interpolation_fbp": interpolation}.get(arm)
                records.append(
                    dict(
                        file=name,
                        start=r["start"],
                        span=r["span"],
                        arm=arm,
                        acquired_sha256=acquired_hash,
                        raw_error=relative_l2(
                            filled[:, ~mask], references[fi, :, ~mask].transpose(1, 0, 2)
                        )
                        if filled is not None
                        else None,
                        withheld_error=relative_l2(projected[:, ~mask], references[fi][:, ~mask]),
                        acquired_error=relative_l2(projected[:, mask], observed[:, mask]),
                        acquired_rmse=float(
                            np.sqrt(np.mean((projected[:, mask] - observed[:, mask]) ** 2))
                        ),
                    )
                )
            if fi == 0 and mi == 0:
                np.savez_compressed(
                    root / "fixture.npz",
                    clean=references[fi],
                    observed=observed,
                    mask=mask,
                    learned=learned,
                    interpolation=interpolation,
                    **volumes,
                )
    with (root / "metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    summary = {}
    for arm in volumes:
        rr = [r for r in records if r["arm"] == arm and r["span"] == 90]
        summary[arm] = {
            k: float(np.mean([r[k] for r in rr])) if rr[0][k] is not None else None
            for k in ["raw_error", "withheld_error", "acquired_error", "acquired_rmse"]
        }
    (root / "summary.json").write_text(
        json.dumps(
            dict(
                primary_narrow=summary,
                seconds=time.monotonic() - started,
                bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
            ),
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("../worm_data"))
    p.add_argument("--additional", action="store_true")
    a = p.parse_args()
    run(a.root, a.data, a.additional)
