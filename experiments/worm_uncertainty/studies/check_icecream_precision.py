"""One acquired-only numerical diagnosis after the preserved worm AMP failure."""

import json
import time
import numpy as np
import torch
from experiments.worm_uncertainty.core.icecream_adapter import crop_pair, make_trainer
from experiments.worm_uncertainty.studies.icecream_pilot import (
    DEFAULT_ROOT,
    fresh_stage,
    training_pair,
    gpu_check,
    write_json,
)


def main():
    torch.set_num_threads(2)
    root = DEFAULT_ROOT
    folder, d = fresh_stage(root, "precision_check")
    gpu_check()
    pair, _ = training_pair(root, "worm004_120", d)
    failure = json.loads((root / "runs/worm004_120/report.json").read_text())
    checkpoint = torch.load(
        root / "runs/worm004_120/failed_checkpoint.pt", map_location="cpu", weights_only=False
    )
    rng = np.random.default_rng(d["training"]["seed"])
    for _ in range(failure["updates"] + 1):
        crops, pos = crop_pair(pair, 48, rng, d["training"]["patch_roi_worm_xy"])
    rows = []
    for mixed in [False, True]:
        t, _ = make_trainer(48, 0, 120, "cuda", mixed)
        t.model.load_state_dict(checkpoint["model_state_dict"])
        np.random.seed(d["training"]["seed"] % (2**32))
        for _ in range(failure["updates"]):
            np.random.choice(len(t.k_sets))
        a, b = [v.cuda() for v in crops]
        loss = t.compute_loss(a, b, 0)
        # No loss scaling: isolates FP16 forward/backward range from GradScaler.
        loss.backward()
        grads = [p.grad for p in t.model.parameters() if p.grad is not None]
        rows.append(
            dict(
                mixed=mixed,
                loss=float(loss.detach()),
                loss_finite=bool(torch.isfinite(loss)),
                all_unscaled_gradients_finite=all(bool(torch.isfinite(g).all()) for g in grads),
                max_abs_gradient=max(float(torch.nan_to_num(g).abs().max()) for g in grads),
            )
        )
        del t, loss, a, b, grads
        torch.cuda.empty_cache()
    t, _ = make_trainer(48, 0, 120, "cuda", False)
    rng = np.random.default_rng(d["training"]["seed"])
    tick = time.monotonic()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(16):
        crops, _ = crop_pair(pair, 48, rng, d["training"]["patch_roi_worm_xy"])
        a, b = [v.cuda() for v in crops]
        t.optimizer.zero_grad(set_to_none=True)
        loss = t.compute_loss(a, b, 0)
        loss.backward()
        if not torch.isfinite(loss) or not all(
            torch.isfinite(p.grad).all() for p in t.model.parameters() if p.grad is not None
        ):
            raise ValueError("FP32 diagnosis failed; no further training authorized by this repair")
        t.optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    duration = time.monotonic() - tick
    report = dict(
        failed_after_updates=failure["updates"],
        failed_crop=pos,
        checkpoint_checks=rows,
        fp32_smoke_updates=16,
        fp32_losses=losses,
        fp32_seconds=duration,
        fp32_seconds_per_update=duration / 16,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        withheld_or_truth_access=False,
    )
    write_json(folder / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
