"""Frozen synthetic density benchmark; small training and conditional measured-only SART."""

import argparse
import csv
import json
import platform
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector, relative_l2
from experiments.worm_uncertainty.core.completion_model import CompletionCNN
from experiments.worm_uncertainty.data.phantom_density import objects, density_metrics, FAMILIES
from experiments.worm_uncertainty.data.acquisition_diversity import read_design, schedule, mask_hash
from experiments.worm_uncertainty.data.completion_data import (
    acquisition_mask,
    channels,
    interpolate_missing,
)
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.studies.acquisition_diversity import (
    templates,
    input_tensor,
    missing_loss,
    infer,
)
from experiments.worm_uncertainty.studies.train_completion import configure

BASE = Path(__file__).resolve().parents[1]
PROTOCOL = BASE / "protocols/phantom_density_v1.json"
DEFAULT = Path("output/worm_uncertainty/phantom_density_v1_20261116")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def budget(root, cfg):
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if used > cfg["budget"]["run_artifact_cap_bytes"]:
        raise RuntimeError("artifact cap reached")
    return used


def inventory(rows):
    result = []
    for r in rows:
        mask = acquisition_mask(r["start"], r["span"], np.arange(25) * 7.2)
        result.append(
            dict(
                start=r["start"],
                span=r["span"],
                mask=mask.tolist(),
                hash=mask_hash(mask),
                count=int(mask.sum()),
            )
        )
    return result


def calibration(op, cfg):
    truth = 0.015 * np.exp(-0.5 * (((op.x - 20) / 20) ** 2 + ((op.y + 15) / 30) ** 2))[None]
    y = op.forward(truth)
    theta = np.deg2rad(op.angles)
    centre = 20 * np.cos(theta) + 15 * np.sin(theta)
    variance = (20 * np.cos(theta)) ** 2 + (30 * np.sin(theta)) ** 2
    u = (np.arange(62) - 30.5) * 5
    analytic = (
        0.015
        * np.sqrt(2 * np.pi)
        * 20
        * 30
        / np.sqrt(variance[:, None])
        * np.exp(-0.5 * (u[None] - centre[:, None]) ** 2 / variance[:, None])
    )
    fbp = op.fbp(y, np.ones(25, bool), 7.2)
    rng = np.random.default_rng(91)
    x = rng.normal(size=truth.shape)
    z = rng.normal(size=y.shape)
    lhs = np.vdot(op.forward(x), z)
    rhs = np.vdot(x, op.adjoint(z))
    pos = np.maximum(fbp, 0)[0]
    centroid = np.array([(pos * op.x).sum() / pos.sum(), (pos * op.y).sum() / pos.sum()])
    r = dict(
        analytic_projection_l2=relative_l2(y[0], analytic),
        full_fbp_density_l2=relative_l2(fbp, truth),
        scale_error=abs(float(np.vdot(fbp, truth) / np.vdot(truth, truth)) - 1),
        centroid_error_um=float(np.linalg.norm(centroid - [20, -15])),
        adjoint_relative=float(abs(lhs - rhs) / max(abs(lhs), abs(rhs))),
        reversal_max_abs=float(
            abs(
                ParallelProjector(62, op.angles + 180, spacing=5).forward(truth) - y[..., ::-1]
            ).max()
        ),
    )
    g = cfg["calibration_gates"]
    r["passed"] = (
        all(
            r[k] <= g["reversal_max_abs" if k == "reversal_max_abs" else k + "_max"]
            for k in [
                "analytic_projection_l2",
                "full_fbp_density_l2",
                "centroid_error_um",
                "adjoint_relative",
                "reversal_max_abs",
            ]
        )
        and r["scale_error"] <= g["full_fbp_scale_error_max"]
    )
    return r


def examples(clean, noisy, rows):
    obs = np.stack([np.where(np.array(r["mask"])[None, :, None], noisy, 0) for r in rows], axis=1)
    x = np.stack(
        [channels(v, r["mask"], r["start"], r["span"]) for ob in obs for v, r in zip(ob, rows)]
    )
    return x, obs


def aggregate(rows, method, span=90):
    chosen = [r for r in rows if r["method"] == method and r["span"] == span]
    fields = [
        "density_l2",
        "withheld_l2",
        "acquired_l2",
        "acquired_rmse",
        "boundary_um",
        "interior_bias",
        "interior_mae",
    ]
    # Every object has the same mask count. Explicit object-first aggregation.
    return {
        k: float(
            np.mean(
                [
                    np.mean([r[k] for r in chosen if r["object"] == i])
                    for i in sorted({r["object"] for r in chosen})
                ]
            )
        )
        if all(r[k] is not None for r in chosen)
        else None
        for k in fields
    }


def gate(rows, cfg):
    a = aggregate(rows, "interpolation_fbp")
    b = aggregate(rows, "learned_fbp")
    ids = sorted({r["object"] for r in rows})
    improvements = []
    for i in ids:
        rr = [r for r in rows if r["object"] == i and r["span"] == 90]
        improvements.append(
            float(
                np.mean([r["density_l2"] for r in rr if r["method"] == "interpolation_fbp"])
                - np.mean([r["density_l2"] for r in rr if r["method"] == "learned_fbp"])
            )
        )
    g = cfg["gate"]
    wins = sum(v > 0 for v in improvements)
    return dict(
        passed=bool(
            a["density_l2"] - b["density_l2"] >= g["density_gain_absolute"]
            and wins / len(ids) > g["object_win_fraction_strictly_above"]
            and b["acquired_l2"] - a["acquired_l2"] <= g["acquired_residual_degradation_max"]
        ),
        density_gain=a["density_l2"] - b["density_l2"],
        acquired_degradation=b["acquired_l2"] - a["acquired_l2"],
        wins=wins,
        objects=len(ids),
        per_object_gain=improvements,
    )


def train(root, cfg, op):
    d, _ = read_design()
    rows = d["inventories"]["B"]
    sel = inventory(cfg["selection_masks"])
    train_truth = objects(cfg["splits"]["train"])
    clean = op.forward(train_truth)
    val_truth = objects(cfg["splits"]["selection"])
    val = op.forward(val_truth)
    rng = np.random.default_rng(cfg["noise"]["training_seed"])
    bank = clean[None] + rng.normal(0, 0.05, (8, *clean.shape))
    vn = val + np.random.default_rng(cfg["noise"]["selection_seed"]).normal(0, 0.05, val.shape)
    x, obs = examples(val, vn, sel)
    vx = torch.tensor(x, device="cuda")
    params = cfg["training"]
    configure(params["seed"])
    model = CompletionCNN().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    target = torch.tensor(clean, dtype=torch.float32, device="cuda")
    bank = torch.tensor(bank, dtype=torch.float32, device="cuda")
    tmp = templates(rows, "cuda")
    ids, mi, nb = schedule(len(clean), params["batch"], params["updates"], params["seed"], rows)
    write(
        root / "input_hashes.json",
        {
            k: __import__("hashlib").sha256(a.tobytes()).hexdigest()
            for k, a in [
                ("training_truth", train_truth),
                ("training_clean", clean),
                ("selection_truth", val_truth),
            ]
        },
    )
    best = float("inf")
    history = []
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(params["updates"]):
        if time.monotonic() - start > params["gpu_cap_seconds"]:
            raise TimeoutError("training cap")
        ix = torch.tensor(ids[step], device="cuda")
        mm = torch.tensor(mi[step], device="cuda")
        bb = torch.tensor(nb[step] % 8, device="cuda")
        xx = input_tensor(bank[bb, ix], mm, tmp)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = missing_loss(model(xx), target[ix], xx[:, 1])
        loss.backward()
        if not torch.isfinite(loss) or not all(
            torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise ValueError("nonfinite training")
        opt.step()
        if (step + 1) % params["selection_frequency"] == 0:
            raw = infer(model, vx, 64).reshape(obs.shape)
            completed = np.stack(
                [
                    np.where(np.array(r["mask"])[None, :, None], obs[:, j], raw[:, j])
                    for j, r in enumerate(sel)
                ],
                axis=1,
            )
            current = float(
                np.mean(
                    [
                        np.mean(
                            [
                                relative_l2(
                                    completed[i, j, ~np.array(r["mask"])],
                                    val[i, ~np.array(r["mask"])],
                                )
                                for j, r in enumerate(sel)
                            ]
                        )
                        for i in range(len(val))
                    ]
                )
            )
            if current < best:
                best = current
                best_step = step + 1
                torch.save(model.state_dict(), root / "best.pt")
            history.append(
                dict(
                    update=step + 1,
                    loss=float(loss.detach()),
                    selection=current,
                    seconds=time.monotonic() - start,
                )
            )
            print(history[-1], flush=True)
            budget(root, cfg)
    report = dict(
        status="completed",
        seconds=time.monotonic() - start,
        updates=step + 1,
        best_update=best_step,
        best_selection=best,
        checkpoint_sha256=sha256(root / "best.pt"),
        parameters=sum(p.numel() for p in model.parameters()),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
    )
    write(root / "training.json", report)
    write(root / "history.json", history)
    model.cpu()
    model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=True))
    model.eval()
    torch.cuda.empty_cache()
    return model


def row_metrics(vol, truth, observed, clean, mask, op, object_id, family, r, method, raw=None):
    projected = op.forward(vol)
    return [
        dict(
            object=object_id + i,
            family=family[i],
            start=r["start"],
            span=r["span"],
            orientation="familiar" if r["start"] == 0 else "held",
            method=method,
            **density_metrics(v, t),
            withheld_l2=relative_l2(p[~mask], c[~mask]),
            acquired_l2=relative_l2(p[mask], y[mask]),
            acquired_rmse=float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2))),
            raw_completion_l2=relative_l2(raw[i, ~mask], c[~mask]) if raw is not None else None,
        )
        for i, (v, t, p, c, y) in enumerate(zip(vol, truth, projected, clean, observed))
    ]


def evaluate(root, cfg, op, model):
    start = time.monotonic()
    truth = objects(cfg["splits"]["evaluation"])
    clean = op.forward(truth)
    noisy = clean + np.random.default_rng(cfg["noise"]["evaluation_seed"]).normal(
        0, 0.05, clean.shape
    )
    masks = inventory(cfg["evaluation_masks"])
    rows = []
    saved = {}
    volumes = {}
    observations = []
    filled = []
    for j, r in enumerate(masks):
        if time.monotonic() - start > cfg["budget"]["cpu_evaluation_seconds"]:
            raise TimeoutError("CPU evaluation cap")
        mask = np.array(r["mask"])
        obs = np.where(mask[None, :, None], noisy, 0)
        observations.append(obs)
        x = torch.tensor(np.stack([channels(v, mask, r["start"], r["span"]) for v in obs]))
        with torch.no_grad():
            pred = model(x).numpy()
        completed = np.where(mask[None, :, None], obs, pred)
        interpolation = np.stack([interpolate_missing(v, mask) for v in obs])
        interpolation = np.where(mask[None, :, None], obs, interpolation)
        for a in [completed, interpolation]:
            np.testing.assert_array_equal(a[:, mask], obs[:, mask])
        arms = dict(
            acquired_fbp=op.fbp(obs, mask, 7.2),
            sart_zero=op.sart(obs, mask)[0],
            interpolation_fbp=op.fbp(interpolation, np.ones(25, bool), 7.2),
            learned_fbp=op.fbp(completed, np.ones(25, bool), 7.2),
        )
        filled.append(np.stack([interpolation, completed]))
        volumes[j] = arms
        for method, v in arms.items():
            rows += row_metrics(
                v,
                truth,
                obs,
                clean,
                mask,
                op,
                0,
                [FAMILIES[i % 4] for i in range(len(truth))],
                r,
                method,
                completed
                if method == "learned_fbp"
                else interpolation
                if method == "interpolation_fbp"
                else None,
            )
    initial_gate = gate(rows, cfg)
    write(root / "advancement_gate.json", initial_gate)
    if initial_gate["passed"]:
        tick = time.monotonic()
        for j, r in enumerate(masks):
            mask = np.array(r["mask"])
            obs = observations[j]
            for method, source in [
                ("sart_interpolation_init", "interpolation_fbp"),
                ("sart_learned_init", "learned_fbp"),
            ]:
                if time.monotonic() - tick > cfg["conditional_refinement"]["budget_seconds"]:
                    raise TimeoutError("refinement cap")
                v = op.sart(obs, mask, initial=volumes[j][source])[0]
                volumes[j][method] = v
                rows += row_metrics(
                    v,
                    truth,
                    obs,
                    clean,
                    mask,
                    op,
                    0,
                    [FAMILIES[i % 4] for i in range(len(truth))],
                    r,
                    method,
                )
        write(
            root / "refinement.json",
            dict(
                status="completed",
                seconds=time.monotonic() - tick,
                observation="identical acquired rays; no predicted rays in updates",
            ),
        )
    else:
        write(
            root / "refinement.json",
            dict(status="gated_off", reason="density/acquired-residual gate failed"),
        )
    for method in volumes[0]:
        saved[method] = np.stack([volumes[j][method] for j in range(len(masks))]).astype(np.float32)
    np.savez_compressed(
        root / "evaluation_arrays.npz",
        truth=truth,
        clean=clean,
        observed=np.stack(observations),
        filled=np.stack(filled),
        **saved,
    )
    with (root / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        primary={method: aggregate(rows, method) for method in volumes[0]},
        moderate={method: aggregate(rows, method, 120) for method in volumes[0]},
        gate=initial_gate,
        seconds=time.monotonic() - start,
        arrays_sha256=sha256(root / "evaluation_arrays.npz"),
    )
    write(root / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(PROTOCOL.read_text())
    write(root / "protocol.json", cfg)
    (root / "dirty.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    paths = list(BASE.rglob("*.py")) + [PROTOCOL]
    with zipfile.ZipFile(root / "sources.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, str(p.relative_to(Path.cwd())))
    write(
        root / "provenance.json",
        dict(
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            dirty=subprocess.check_output(["git", "status", "--porcelain"], text=True),
            sources={str(p): sha256(p) for p in paths},
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            host=platform.node(),
            reserved_access=False,
        ),
    )
    d, _ = read_design()
    train_rows = d["inventories"]["B"]
    sel = inventory(cfg["selection_masks"])
    ev = inventory(cfg["evaluation_masks"])
    assert not {r["hash"] for r in sel} & {r["hash"] for r in ev}
    assert not {r["hash"] for r in sel} & {r["hash"] for r in train_rows}
    assert all(r["hash"] not in {v["hash"] for v in train_rows} for r in ev if r["start"] != 0)
    write(root / "masks.json", dict(training=train_rows, selection=sel, evaluation=ev))
    torch.set_num_threads(1)
    op = ParallelProjector(62, np.arange(25) * 7.2, spacing=5)
    cal = calibration(op, cfg)
    write(root / "calibration.json", cal)
    if not cal["passed"]:
        raise ValueError("geometry calibration failed before training")
    model = train(root, cfg, op)
    evaluate(root, cfg, op, model)
    write(root / "status.json", dict(status="completed", artifact_bytes=budget(root, cfg)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT)
    args = p.parse_args()
    try:
        run(args.root)
    except Exception as exc:
        if args.root.is_dir() and not (args.root / "status.json").exists():
            write(args.root / "failure.json", dict(error=repr(exc)))
        raise
