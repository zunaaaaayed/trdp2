"""Training-only auxiliary loss calibration; not a restoration performance experiment."""

import json
from pathlib import Path
import numpy as np
import torch
from experiments.worm_uncertainty.core.completion_model import CompletionCNN
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector
from experiments.worm_uncertainty.core.differentiable_fbp import DifferentiableFBP
from experiments.worm_uncertainty.data.acquisition_diversity import read_design
from experiments.worm_uncertainty.data.completion_data import channels
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    root = Path("output/worm_uncertainty/bridge_loss_scale_v1_20261116")
    root.mkdir(parents=True, exist_ok=False)
    cache = Path("output/worm_uncertainty/acquisition_diversity_v3/profile")
    protocol = dict(
        stage="training-only scale calibration",
        seed=2026111601,
        batch="first eight training slices",
        mask="first broad V3 mask",
        rule="lambda = missing-ray MSE / auxiliary L1 at initialization; equal initial scalar contributions",
        budget_seconds=60,
        artifact_cap_bytes=1048576,
        claims="no training or held-out performance claim",
    )
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    hashes = json.loads((cache / "cache_hashes.json").read_text())
    for name in ["train_clean.npy", "noise_seed0.npy"]:
        assert sha256(cache / name) == hashes[name]
    d, _ = read_design()
    r = d["inventories"]["B"][0]
    mask = torch.tensor(r["mask"], dtype=torch.bool)
    clean = np.load(cache / "train_clean.npy", mmap_mode="r").reshape(-1, 25, 62)[:8].copy()
    noisy = np.load(cache / "noise_seed0.npy", mmap_mode="r").reshape(16, -1, 25, 62)[0, :8].copy()
    measured = np.where(mask.numpy()[None, :, None], noisy, 0)
    x = torch.tensor(np.stack([channels(v, r["mask"], r["start"], r["span"]) for v in measured]))
    torch.set_num_threads(1)
    torch.manual_seed(protocol["seed"])
    model = CompletionCNN()
    pred = model(x)
    y = torch.tensor(measured, dtype=pred.dtype)
    target = torch.tensor(clean, dtype=pred.dtype)
    completed = torch.where(mask[:, None], y, pred)
    paired_target = torch.where(mask[:, None], y, target)
    fbp = DifferentiableFBP(ParallelProjector(62, np.arange(25) * 7.2, spacing=5), 7.2)
    loss = (pred[:, ~mask] - target[:, ~mask]).square().mean()
    auxiliary = (
        (fbp(completed, torch.ones_like(mask)) - fbp(paired_target, torch.ones_like(mask)))
        .abs()
        .mean()
    )
    weight = float((loss / auxiliary).detach())
    (loss + weight * auxiliary).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    result = dict(
        missing_mse=float(loss.detach()),
        surrogate_l1=float(auxiliary.detach()),
        lambda_value=weight,
        finite_gradients=True,
        source_hashes=hashes,
        evaluation_access=False,
    )
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
