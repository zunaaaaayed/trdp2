"""One frozen paired seed: missing-ray MSE versus MSE plus FBP surrogate."""

import json
import time
from pathlib import Path
import numpy as np
import torch
from experiments.worm_uncertainty.studies.acquisition_diversity import (
    templates,
    input_tensor,
    infer,
    missing_loss,
    provenance,
)
from experiments.worm_uncertainty.studies.train_completion import configure
from experiments.worm_uncertainty.data.acquisition_diversity import (
    read_design,
    load_role,
    evaluation_arrays,
    preserve,
    score,
    schedule,
)
from experiments.worm_uncertainty.core.completion_model import CompletionCNN
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector
from experiments.worm_uncertainty.core.differentiable_fbp import DifferentiableFBP
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    root = Path("output/worm_uncertainty/bridge_ablation_v1_20261116")
    root.mkdir(parents=True, exist_ok=False)
    d, m = read_design()
    cache = Path("output/worm_uncertainty/acquisition_diversity_v3/profile")
    seed = 2026102901
    calibration = Path("output/worm_uncertainty/bridge_loss_scale_v1_20261116/result.json")
    weight = json.loads(calibration.read_text())["lambda_value"]
    cfg = dict(
        seed=seed,
        updates=1000,
        batch=256,
        validation_frequency=200,
        arms={"A": 0, "B": weight},
        selection_files=d["roles"]["selection"],
        selection_masks=d["inventories"]["selection"],
        evaluation_files=d["roles"]["audit"][:3],
        evaluation_masks=[
            dict(start=64.8, span=90),
            dict(start=136.8, span=90),
            dict(start=64.8, span=120),
        ],
        exposure="previous V3 audit deformations; masks previously used in bridge on different deformations",
        selection_rule="original V3 mean per-file missing-ray relative L2; first checkpoint wins ties",
        gates=dict(withheld_gain=0.02, acquired_degradation=0.01, majority=True),
        cap_seconds_per_arm=1200,
        artifact_cap_bytes=250 * 1024**2,
        calibration_sha256=sha256(calibration),
        training_family=d["inventories"]["B"],
        normalization=d["normalization"],
        sampling=d["sampling"],
        purpose="worm FBP surrogate ablation; no voxel accuracy claim",
        second_seed="not automatic; first-seed feasibility only",
    )
    provenance(root, cfg)
    hashes = json.loads((cache / "cache_hashes.json").read_text())
    for name in hashes:
        assert sha256(cache / name) == hashes[name]
    target = torch.tensor(
        np.load(cache / "train_clean.npy").reshape(-1, 25, 62), dtype=torch.float32, device="cuda"
    )
    bank = torch.tensor(np.load(cache / "noise_seed0.npy").reshape(16, -1, 25, 62), device="cuda")
    val = load_role(Path("../worm_data"), d, m, "selection")
    vx, obs, _ = evaluation_arrays(val, d, "selection")
    vx = torch.tensor(vx, device="cuda")
    ids, mis, banks = schedule(len(target), 256, 1000, seed, d["inventories"]["B"])
    tmp = templates(d["inventories"]["B"], "cuda")
    fbp = (
        DifferentiableFBP(ParallelProjector(62, np.arange(25) * 7.2, spacing=5), 7.2).float().cuda()
    )
    full = torch.ones(25, dtype=torch.bool, device="cuda")
    for arm, lam in cfg["arms"].items():
        configure(seed)
        model = CompletionCNN().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        dest = root / arm
        dest.mkdir()
        best = float("inf")
        history = []
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(1000):
                if time.monotonic() - start > 1190:
                    raise TimeoutError("arm cap")
                if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > 240 * 1024**2:
                    raise RuntimeError("artifact safety margin")
                sample = torch.tensor(ids[step], device="cuda")
                mi = torch.tensor(mis[step], device="cuda")
                nb = torch.tensor(banks[step], device="cuda")
                xx = input_tensor(bank[nb, sample], mi, tmp)
                yy = target[sample]
                model.train()
                opt.zero_grad(set_to_none=True)
                # Fixed microbatches keep both arms paired while bounding temporary activations.
                total = 0.0
                for j in range(0, 256, 32):
                    x = xx[j : j + 32]
                    t = yy[j : j + 32]
                    pred = model(x)
                    loss = missing_loss(pred, t, x[:, 1])
                    mask = x[:, 1, :, 0].bool()
                    if lam:
                        complete = torch.where(mask[..., None], x[:, 0], pred)
                        paired = torch.where(mask[..., None], x[:, 0], t)
                        loss = loss + lam * (fbp(complete, full) - fbp(paired, full)).abs().mean()
                    (loss / 8).backward()
                    total += float(loss.detach()) / 8
                if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError("nonfinite gradient")
                opt.step()
                if (step + 1) % 200 == 0:
                    raw = infer(model, vx, 256)
                    pred = preserve(raw, obs, d["inventories"]["selection"])
                    metric = score(
                        pred, val, d["inventories"]["selection"], d["roles"]["selection"]
                    )["groups"]["all"]["mean"]
                    if metric < best:
                        best = metric
                        torch.save(model.state_dict(), dest / "best.pt")
                        best_step = step + 1
                    history.append(dict(update=step + 1, training_loss=total, selection=metric))
                    print(arm, history[-1], flush=True)
            report = dict(
                status="completed",
                seconds=time.monotonic() - start,
                best_update=best_step,
                best_selection=best,
                checkpoint_sha256=sha256(dest / "best.pt"),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
            )
        except Exception as exc:
            (dest / "failure.json").write_text(
                json.dumps(
                    dict(error=str(exc), seconds=time.monotonic() - start, completed_updates=step),
                    indent=2,
                )
            )
            raise
        (dest / "history.json").write_text(json.dumps(history, indent=2))
        (dest / "report.json").write_text(json.dumps(report, indent=2))
    print(
        "PAIRED TRAINING COMPLETE; evaluation settings already frozen in protocol.json", flush=True
    )


if __name__ == "__main__":
    main()
