"""Bounded per-acquisition ICECREAM pilot. Stages refuse existing destinations."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

from experiments.worm_uncertainty.core.voxel_parallel import (
    ParallelProjector,
    reconstruction_metrics,
    relative_l2,
)
from experiments.worm_uncertainty.core.icecream_adapter import (
    OFFICIAL,
    canonical_axes,
    crop_pair,
    infer_pair,
    load_official,
    make_trainer,
    method_record,
    native_axes,
    wedge,
)
from experiments.worm_uncertainty.data.alternative_reconstruction import (
    MANIFEST,
    load_slab,
    mask_record,
    noisy_pair,
    phantom,
)
from experiments.worm_uncertainty.data.completion_data import acquisition_mask
from experiments.worm_uncertainty.data.prepare import sha256

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
PROTOCOL = BASE / "protocols/icecream_pilot_v1.json"
DEFAULT_ROOT = REPO / "output/worm_uncertainty/icecream_pilot_v1"


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def source_record(destination):
    paths = [
        BASE / p
        for p in [
            "core/voxel_parallel.py",
            "core/icecream_adapter.py",
            "data/alternative_reconstruction.py",
            "data/completion_data.py",
            "data/prepare.py",
            "studies/icecream_pilot.py",
            "studies/run_icecream_pilot.py",
            "studies/audit_icecream_operator.py",
            "protocols/icecream_pilot_v1.json",
            "protocols/completion_manifest_v1.json",
        ]
    ]
    archive = destination / "source"
    archive.mkdir()
    hashes = {}
    for p in paths:
        rel = p.relative_to(REPO)
        target = archive / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        hashes[str(rel)] = sha256(p)
    return dict(
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        source_hashes=hashes,
        official_commit=subprocess.check_output(
            ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True
        ).strip(),
        official_files={
            str(p.relative_to(OFFICIAL)): sha256(p)
            for p in sorted((OFFICIAL / "src").rglob("*.py"))
        },
        python=sys.version,
        torch=torch.__version__,
        cuda_build=torch.version.cuda,
        host=platform.node(),
        platform=platform.platform(),
        cpu_threads=torch.get_num_threads(),
        protocol_sha256=sha256(PROTOCOL),
        manifest_sha256=sha256(MANIFEST),
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def read_protocol(root):
    d = json.loads(PROTOCOL.read_text())
    if sha256(MANIFEST) != d["manifest_sha256"]:
        raise ValueError("population manifest changed")
    frozen = root / "protocol.json"
    if frozen.exists() and sha256(frozen) != sha256(PROTOCOL):
        raise ValueError("frozen protocol changed")
    return d


def fresh_stage(root, name):
    root.mkdir(parents=True, exist_ok=True)
    d = read_protocol(root)
    if not (root / "protocol.json").exists():
        shutil.copyfile(PROTOCOL, root / "protocol.json")
    folder = root / name
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "provenance.json", source_record(folder))
    return folder, d


def geometry_checks(root):
    folder, _ = fresh_stage(root, "geometry")
    started = time.monotonic()
    n = 96
    angles = np.arange(100) * 1.8
    op = ParallelProjector(n, angles)
    centre = [11.0, -7.0]
    sx, sy = 7.0, 11.0
    truth = np.exp(-0.5 * (((op.x - centre[0]) / sx) ** 2 + ((op.y - centre[1]) / sy) ** 2))[None]
    actual = op.forward(truth)
    theta = np.deg2rad(angles)
    u = np.arange(n) - (n - 1) / 2
    var = sx * sx * np.cos(theta) ** 2 + sy * sy * np.sin(theta) ** 2
    means = centre[0] * np.cos(theta) - centre[1] * np.sin(theta)
    exact = (
        np.sqrt(2 * np.pi)
        * sx
        * sy
        / np.sqrt(var[:, None])
        * np.exp(-0.5 * (u[None] - means[:, None]) ** 2 / var[:, None])
    )
    reconstructed = op.fbp(actual, np.ones(100, bool))
    weight = np.maximum(reconstructed[0], 0)
    centroid = np.array([(weight * op.x).sum(), (weight * op.y).sum()]) / weight.sum()
    rng = np.random.default_rng(2026111000)
    v, y = rng.normal(size=(2, n, n)), rng.normal(size=(2, 100, n))
    lhs, rhs = np.vdot(op.forward(v), y), np.vdot(v, op.adjoint(y))
    shifted = ParallelProjector(n, angles + 180).forward(truth)
    report = dict(
        analytic_projection_relative_l2=relative_l2(actual[0], exact),
        adjoint_relative_error=float(abs(lhs - rhs) / max(abs(lhs), abs(rhs))),
        reversal_relative_l2=relative_l2(shifted, actual[..., ::-1]),
        full_angle_volume_relative_l2=relative_l2(reconstructed, truth),
        full_angle_scale_ratio=float(np.vdot(reconstructed, truth) / np.vdot(truth, truth)),
        centroid_error_pixels=float(np.linalg.norm(centroid - centre)),
        convention="pixel centres; u=x cos(theta)-y sin(theta), axial z; density per assumed length unit",
    )
    report["passed"] = bool(
        report["analytic_projection_relative_l2"] < 0.06
        and report["adjoint_relative_error"] < 1e-11
        and report["reversal_relative_l2"] < 1e-11
        and abs(report["full_angle_scale_ratio"] - 1) < 0.06
        and report["centroid_error_pixels"] < 0.5
    )
    report["seconds"] = time.monotonic() - started
    write_json(folder / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise ValueError("geometry acceptance failed")


def prepare(root, case_id):
    d = read_protocol(root)
    if not json.loads((root / "geometry/report.json").read_text())["passed"]:
        raise ValueError("geometry gate not passed")
    case = next(c for c in d["cases"] if c["id"] == case_id)
    folder, _ = fresh_stage(root, "inputs/" + case_id)
    start = time.monotonic()
    angles = np.array(d["angles_deg"])
    truth = None
    if case["kind"] == "phantom":
        truth = phantom(case["phantom_seed"], 64, case["object_rotation"])
        op = ParallelProjector(64, angles)
        clean = op.forward(truth)
        source = dict(generator_seed=case["phantom_seed"], rotation=case["object_rotation"])
    else:
        clean, source = load_slab(REPO.parent / "worm_data", case["file"])
        op = ParallelProjector(310, angles)
    mask = acquisition_mask(case["start"], case["span"], angles)
    pair = noisy_pair(clean, mask, d["noise"]["sigma_each"], case["noise_seeds"])
    fbp_start = time.monotonic()
    fbp = np.stack([op.fbp(v, mask) for v in pair])
    fbp_seconds = time.monotonic() - fbp_start
    # No evaluation metrics here. FBP/SART see only the identical observed pair.
    sart_start = time.monotonic()
    sart, history = op.sart(
        pair.mean(axis=0), mask, **{k: d["baselines"]["sart"][k] for k in ["sweeps", "relaxation"]}
    )
    sart_seconds = time.monotonic() - sart_start
    if not np.isfinite(fbp).all() or not np.isfinite(sart).all():
        raise ValueError("nonfinite preprocessing")
    np.save(folder / "observed_pair.npy", pair)
    np.save(folder / "fbp_pair.npy", fbp.astype(np.float32))
    np.save(folder / "sart.npy", sart.astype(np.float32))
    ev = folder / "evaluation_only"
    ev.mkdir()
    np.save(ev / "clean.npy", clean)
    if truth is not None:
        np.save(ev / "truth.npy", truth)
    # Diagnostic of observed structure anisotropy, not true rotational invariance.
    mean = fbp.mean(0)
    energies = [float(np.mean(np.diff(mean, axis=a) ** 2)) for a in range(3)]
    report = dict(
        case=case,
        source=source,
        mask=mask_record(case["start"], case["span"]),
        shape=list(mean.shape),
        observed_pair_sha256=sha256(folder / "observed_pair.npy"),
        fbp_pair_sha256=sha256(folder / "fbp_pair.npy"),
        sart_sha256=sha256(folder / "sart.npy"),
        clean_evaluation_sha256=sha256(ev / "clean.npy"),
        truth_evaluation_sha256=sha256(ev / "truth.npy") if truth is not None else None,
        fbp_seconds=fbp_seconds,
        sart_seconds=sart_seconds,
        sart_acquired_history=history,
        observed_fbp_axis_gradient_energy_zyx=energies,
        observed_fbp_max_min_axis_energy_ratio=max(energies) / max(min(energies), 1e-30),
        seconds=time.monotonic() - start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        training_information="only observed_pair and fbp_pair; evaluation_only never loaded by trainer",
    )
    write_json(folder / "report.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ["case", "shape", "seconds", "fbp_seconds", "sart_seconds"]}
        ),
        flush=True,
    )


def training_pair(root, case_id, d):
    p = root / "inputs" / case_id
    report = json.loads((p / "report.json").read_text())
    if sha256(p / "fbp_pair.npy") != report["fbp_pair_sha256"]:
        raise ValueError("training input changed")
    array = np.load(p / "fbp_pair.npy")
    return [
        torch.from_numpy(native_axes(v) * d["normalization"]["network_volume_gain"]) for v in array
    ], report


def gpu_check(required=4 * 1024**3):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; use the GPU execution permission, do not change scientific settings"
        )
    free, total = torch.cuda.mem_get_info()
    if free < required:
        raise RuntimeError("insufficient currently free VRAM including headroom")
    return dict(
        name=torch.cuda.get_device_name(),
        free_bytes=free,
        total_bytes=total,
        nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True),
    )


def profile(root):
    folder, d = fresh_stage(root, "profile")
    start = time.monotonic()
    gpu = gpu_check()
    pair, _ = training_pair(root, "phantom_base_120", d)
    rows = []
    for size in d["training"]["patch_candidates"]:
        t, _ = make_trainer(size, 0, 120, "cuda", False)
        crops, _ = crop_pair(pair, size, np.random.default_rng(19))
        a, b = [x.cuda() for x in crops]
        torch.cuda.reset_peak_memory_stats()
        # Same network/dropout/random rotation for the precision comparison.
        losses, grads = [], []
        for mixed in [False, True]:
            t.configs.use_mixed_precision = mixed
            t.autocast = torch.autocast("cuda", dtype=torch.float16)
            t.optimizer.zero_grad(set_to_none=True)
            np.random.seed(17)
            torch.manual_seed(17)
            loss = t.compute_loss(a, b, 0)
            loss.backward()
            grad = torch.cat(
                [p.grad.flatten().float() for p in t.model.parameters() if p.grad is not None]
            )
            if not torch.isfinite(loss) or not torch.isfinite(grad).all():
                raise ValueError("nonfinite precision smoke")
            losses.append(float(loss.detach()))
            grads.append(grad.detach().cpu())
        loss_error = abs(losses[1] - losses[0]) / max(abs(losses[0]), 1e-12)
        grad_error = float(
            torch.linalg.vector_norm(grads[1] - grads[0]) / torch.linalg.vector_norm(grads[0])
        )
        mixed = loss_error <= 0.02 and grad_error <= 0.05
        t.configs.use_mixed_precision = mixed
        scaler = torch.amp.GradScaler("cuda", enabled=mixed)
        times = []
        for _ in range(6):
            torch.cuda.synchronize()
            tick = time.monotonic()
            t.optimizer.zero_grad(set_to_none=True)
            loss = t.compute_loss(a, b, 0)
            scaler.scale(loss).backward()
            scaler.step(t.optimizer)
            scaler.update()
            torch.cuda.synchronize()
            times.append(time.monotonic() - tick)
        row = dict(
            patch=size,
            seconds_per_update=float(np.median(times[2:])),
            loss_fp32_amp=losses,
            loss_relative_error=loss_error,
            gradient_relative_error=grad_error,
            mixed_precision=mixed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            headroom_bytes=torch.cuda.mem_get_info()[0],
            parameters=sum(p.numel() for p in t.model.parameters()),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        del t, a, b, grad, grads, loss
        torch.cuda.empty_cache()
    eligible = [r for r in rows if r["headroom_bytes"] >= 3 * 1024**3]
    if not eligible:
        raise RuntimeError("no profiled patch meets memory headroom")
    chosen = eligible[-1]
    step = chosen["seconds_per_update"]
    updates = dict(
        phantom=min(2000, int(240 / step / 25) * 25), worm=min(4000, int(600 / step / 25) * 25)
    )
    if min(updates.values()) < 100:
        raise RuntimeError("throughput too low for frozen pilot budget")
    write_json(
        folder / "report.json",
        dict(
            gpu=gpu,
            profiles=rows,
            selected=chosen,
            updates=updates,
            seconds=time.monotonic() - start,
            selection="training-only throughput/precision; no clean or withheld evaluation loaded",
        ),
    )
    write_json(
        root / "training_freeze.json",
        dict(
            patch=chosen["patch"],
            mixed_precision=chosen["mixed_precision"],
            updates=updates,
            profile_sha256=sha256(folder / "report.json"),
            protocol_sha256=sha256(PROTOCOL),
            selection="fixed final update; settings identical for all acquisitions",
        ),
    )


def train(root, case_id, device, attempt=None):
    run_group = "runs" if attempt is None else "runs_fp32"
    folder, d = fresh_stage(root, run_group + "/" + case_id)
    case = next(c for c in d["cases"] if c["id"] == case_id)
    frozen = json.loads((root / "training_freeze.json").read_text())
    if frozen["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("training freeze changed")
    size = frozen["patch"]
    start = time.monotonic()
    gpu = gpu_check()
    pair, inp = training_pair(root, case_id, d)
    mixed = frozen["mixed_precision"]
    amendment = None
    if attempt is not None:
        amendment = json.loads((root / "precision_amendment.json").read_text())
        if (
            attempt != "fp32"
            or case["kind"] != "worm"
            or amendment["updates"] != updates_for_case(frozen, case)
        ):
            raise ValueError("invalid numerical amendment")
        mixed = False
    trainer, model_config = make_trainer(
        size, case["start"], case["span"], device, mixed, d["training"]["seed"]
    )
    write_json(folder / "method.json", method_record(trainer, model_config))
    write_json(
        folder / "input_contract.json",
        dict(
            input_report_sha256=sha256(root / "inputs" / case_id / "report.json"),
            observed_pair_sha256=inp["observed_pair_sha256"],
            fbp_pair_sha256=inp["fbp_pair_sha256"],
            training_freeze_sha256=sha256(root / "training_freeze.json"),
            seed=d["training"]["seed"],
            gpu=gpu,
            numerical_amendment_sha256=sha256(root / "precision_amendment.json")
            if amendment
            else None,
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=mixed)
    rng = np.random.default_rng(d["training"]["seed"])
    updates = frozen["updates"][case["kind"]]
    cap = d["caps"][case["kind"] + "_case_train_seconds"]
    roi = d["training"]["patch_roi_worm_xy"] if case["kind"] == "worm" else None
    torch.cuda.reset_peak_memory_stats()
    completed = 0
    try:
        with (folder / "losses.csv").open("x", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(
                [
                    "update",
                    "data_loss",
                    "equivariance_loss",
                    "total_loss",
                    "elapsed_seconds",
                    "crop_x",
                    "crop_y",
                    "crop_z",
                ]
            )
            for update in range(updates):
                if time.monotonic() - start > cap:
                    raise TimeoutError(
                        "training time cap; no automatic extension or partial-score selection"
                    )
                crops, pos = crop_pair(pair, size, rng, roi)
                a, b = [v.to(device) for v in crops]
                trainer.optimizer.zero_grad(set_to_none=True)
                loss = trainer.compute_loss(a, b, 0)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(trainer.optimizer)
                if not all(
                    torch.isfinite(p.grad).all()
                    for p in trainer.model.parameters()
                    if p.grad is not None
                ):
                    raise ValueError("nonfinite gradient")
                scaler.step(trainer.optimizer)
                scaler.update()
                completed = update + 1
                writer.writerow(
                    [
                        completed,
                        trainer.obs_loss_set[-1],
                        trainer.equi_loss_set[-1],
                        float(loss.detach()),
                        time.monotonic() - start,
                        *pos,
                    ]
                )
                if completed % 100 == 0:
                    f.flush()
                    print(
                        f"{case_id}: {completed}/{updates}, loss={float(loss):.6g}, {time.monotonic() - start:.1f}s",
                        flush=True,
                    )
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - start
        torch.save(
            dict(model_state_dict=trainer.model.state_dict(), updates=completed),
            folder / "checkpoint.pt",
        )
        inference_start = time.monotonic()
        estimate = infer_pair(
            trainer,
            pair,
            size,
            mixed,
            deadline=inference_start + d["caps"][case["kind"] + "_case_inference_seconds"],
        )
        volume = canonical_axes(estimate) / d["normalization"]["network_volume_gain"]
        np.save(folder / "volume.npy", volume)
        report = dict(
            status="completed",
            case=case,
            updates=completed,
            training_seconds=train_seconds,
            inference_seconds=time.monotonic() - inference_start,
            seconds=time.monotonic() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            checkpoint_sha256=sha256(folder / "checkpoint.pt"),
            volume_sha256=sha256(folder / "volume.npy"),
            fixed_checkpoint=True,
        )
    except Exception as e:
        torch.save(
            dict(model_state_dict=trainer.model.state_dict(), updates=completed),
            folder / "failed_checkpoint.pt",
        )
        report = dict(
            status="failed",
            case=case,
            updates=completed,
            seconds=time.monotonic() - start,
            reason=str(e),
            traceback=traceback.format_exc(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        write_json(folder / "report.json", report)
        raise
    write_json(folder / "report.json", report)
    print(json.dumps(report), flush=True)


def updates_for_case(frozen, case):
    return frozen["updates"][case["kind"]]


def evaluate(root, case_id, attempt=None):
    evaluation_group = "evaluation" if attempt is None else "evaluation_fp32"
    folder, d = fresh_stage(root, evaluation_group + "/" + case_id)
    inp = root / "inputs" / case_id
    info = json.loads((inp / "report.json").read_text())
    case = info["case"]
    run = root / ("runs" if attempt is None else "runs_fp32") / case_id
    run_info = json.loads((run / "report.json").read_text())
    if run_info["status"] != "completed" or not run_info["fixed_checkpoint"]:
        raise ValueError("only completed fixed-rule inference may be evaluated")
    if sha256(run / "volume.npy") != run_info["volume_sha256"]:
        raise ValueError("output changed")
    clean = np.load(inp / "evaluation_only/clean.npy")
    truth_path = inp / "evaluation_only/truth.npy"
    truth = np.load(truth_path) if truth_path.exists() else None
    pair = np.load(inp / "observed_pair.npy")
    mask = np.zeros(100, bool)
    mask[info["mask"]["indices"]] = True
    op = ParallelProjector(info["shape"][-1], np.array(d["angles_deg"]))
    rows = []
    for method, volume in [
        ("FBP", np.load(inp / "fbp_pair.npy").mean(0)),
        ("SART", np.load(inp / "sart.npy")),
        ("ICECREAM", np.load(run / "volume.npy")),
    ]:
        # Apply the declared cylindrical support equally; preserve raw network output.
        outside_energy = float(np.sum(volume[:, ~op.support] ** 2))
        volume = volume * op.support[None]
        values, prediction = reconstruction_metrics(volume, clean, pair.mean(0), mask, op, truth)
        values.update(
            method=method,
            raw_outside_support_squared_norm=outside_energy,
            input="same two independent acquired observations",
            regime="per-acquisition self-supervised"
            if method == "ICECREAM"
            else "deterministic pair-average reconstruction",
            training_seconds=run_info["training_seconds"] if method == "ICECREAM" else 0,
            inference_seconds=run_info["inference_seconds"]
            if method == "ICECREAM"
            else info[method.lower() + "_seconds"],
            peak_gpu_bytes=run_info["peak_allocated_bytes"] if method == "ICECREAM" else 0,
        )
        rows.append(values)
        np.save(folder / (method.lower() + "_volume.npy"), volume.astype(np.float32))
        np.save(folder / (method.lower() + "_projection.npy"), prediction.astype(np.float32))
    write_json(
        folder / "report.json",
        dict(
            case=case,
            rows=rows,
            input_report_sha256=sha256(inp / "report.json"),
            run_report_sha256=sha256(run / "report.json"),
            limitations=[
                "FBP/SART and ICECREAM have different priors and costs",
                "finite voxel projector is not the verified worm simulator",
                "no worm voxel truth",
                "per-acquisition training, not zero-shot transfer",
                "no posterior uncertainty",
            ],
        ),
    )
    print(json.dumps(rows, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["geometry", "prepare", "profile", "train", "evaluate"])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--case")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attempt", choices=["fp32"])
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.stage == "geometry":
        geometry_checks(args.root)
    elif args.stage == "prepare":
        prepare(args.root, args.case)
    elif args.stage == "profile":
        profile(args.root)
    elif args.stage == "train":
        train(args.root, args.case, args.device, args.attempt)
    elif args.stage == "evaluate":
        evaluate(args.root, args.case, args.attempt)


if __name__ == "__main__":
    main()
