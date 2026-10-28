"""Bounded matched-input completion pilot: familiar validation selects checkpoints."""

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from experiments.worm_uncertainty.core.completion_model import CompletionCNN, angular_connectivity
from experiments.worm_uncertainty.data.matched_completion import (
    PROTOCOL,
    read_protocol,
    grid,
    load_clean,
    examples,
    metrics,
    preserve_measured,
)
from experiments.worm_uncertainty.data.completion_data import acquisition_mask, interpolate_missing
from experiments.worm_uncertainty.data.prepare import sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def configure(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def context_checks(design):
    degrees, _, _ = grid(design)
    access = angular_connectivity(len(degrees))
    rows = []
    configure(design["seeds"]["training"])
    model = CompletionCNN()
    for start, span in design["familiar_masks"] + design["shift_masks"]:
        mask = acquisition_mask(start, span, degrees)
        if not access[~mask][:, mask].any(axis=1).all():
            raise ValueError("missing row lacks measured context")
        x = torch.ones(1, 7, 25, 62, requires_grad=True)
        with torch.no_grad():
            x[:, 0, ~mask] = 0
            x[:, 1] = torch.tensor(mask[:, None])
        # All missing rows, centre detector; geometry/mask stay fixed.
        out = model(x)
        responses = []
        for row in np.flatnonzero(~mask):
            grad = torch.autograd.grad(out[0, row, 31], x, retain_graph=True)[0][0, 0]
            responses.append(float(grad[mask].abs().sum()))
        if min(responses) <= 0:
            raise ValueError("tested random weights lack measured sensitivity")
        rows.append(
            dict(
                start=start,
                span=span,
                acquired_count=int(mask.sum()),
                missing_count=int((~mask).sum()),
                minimum_measured_gradient_l1=min(responses),
                all_missing_rows_connected=True,
            )
        )
    return rows


def predict(model, x, device, batch=8):
    result = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            result.append(model(torch.from_numpy(x[i : i + batch]).to(device)).cpu().numpy())
    return np.concatenate(result)


def completed_metrics(raw, example, design, acquisitions):
    _, target, measured, mask, ids = example
    result = np.array([preserve_measured(p, y, m) for p, y, m in zip(raw, measured, mask)])
    values = metrics(result, target, mask, ids, design["subsets"]["validation"], acquisitions)
    measured_pixels = np.broadcast_to(mask[:, :, None], result.shape)
    values["maximum_measured_change"] = float(
        np.max(np.abs(result[measured_pixels] - measured[measured_pixels]))
    )
    if values["maximum_measured_change"] != 0:
        raise ValueError("measured rays changed")
    return result, values


def evaluate_set(models, clean, design, acquisitions, device, folder):
    base = examples(clean, design, "validation", acquisitions, 0, "geometry")
    x, target, measured, mask, ids = base
    preds = dict(
        zero_fill=measured.copy(),
        interpolation=np.array([interpolate_missing(y, m) for y, m in zip(measured, mask)]),
    )
    for arm, model in models.items():
        inputs = x.copy()
        if arm == "mask":
            inputs[:, 2:] = 0
        preds[arm] = predict(model, inputs, device)
    summaries = {}
    arrays = dict(target=target, measured=measured, acquired_mask=mask, identities=ids)
    for arm, pred in preds.items():
        arrays[arm], summaries[arm] = completed_metrics(pred, base, design, acquisitions)
    np.savez_compressed(folder, **arrays)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--data", type=Path, default=Path("../worm_data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    design, manifest = read_protocol(args.protocol)
    if not (args.execute or args.smoke):
        print(json.dumps(design, indent=2))
        return
    if args.output.exists():
        raise ValueError("output exists")
    configure(design["seeds"]["training"])
    contexts = context_checks(design)
    args.output.mkdir(parents=True)
    write(args.output / "protocol.json", design)
    provenance = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p): sha256(p) for p in Path("experiments/worm_uncertainty").rglob("*.py")
        },
        protocol_sha256=sha256(args.protocol),
        manifest_sha256=design["manifest_sha256"],
        host=socket.gethostname(),
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        cuda_build=torch.version.cuda,
        source_inputs={
            r["name"]: r["sha256"]
            for r in manifest["files"]
            if r["name"] in design["subsets"]["train"] + design["subsets"]["validation"]
        },
        reserved_read=False,
    )
    write(args.output / "provenance.json", provenance)
    write(args.output / "context.json", contexts)
    if args.smoke:
        model = CompletionCNN()
        x = torch.ones(8, 7, 25, 62)
        target = torch.zeros(8, 25, 62)
        loss = (model(x) - target).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or not all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
        ):
            raise ValueError("nonfinite gradients")
        torch.optim.Adam(model.parameters(), lr=0.001).step()
        write(
            args.output / "report.json",
            dict(
                status="CPU gradient/context smoke passed",
                loss=float(loss),
                parameters=sum(p.numel() for p in model.parameters()),
                context_masks=len(contexts),
            ),
        )
        print("CPU gradient/context smoke passed", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: no automatic environment installation")
    device = torch.device("cuda")
    provenance["gpu"] = torch.cuda.get_device_name()
    write(args.output / "provenance.json", provenance)
    started = time.monotonic()
    train = load_clean(args.data, design, manifest, "train")
    val = load_clean(args.data, design, manifest, "validation")
    degrees, uv, v = grid(design)
    np.savez_compressed(
        args.output / "inputs.npz",
        train_clean=train,
        validation_clean=val,
        angles_degrees=degrees,
        uv=uv,
        v=v,
        angle_indices=design["sampling"]["angle_indices"],
        detector_indices=design["sampling"]["detector_indices"],
        z_indices=design["sampling"]["z_indices"],
    )
    models = {}
    histories = {}
    training = {}
    for arm in design["arms"]:
        configure(design["seeds"]["training"])
        torch.cuda.reset_peak_memory_stats()
        model = CompletionCNN().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=design["training"]["learning_rate"])
        val_example = examples(val, design, "validation", design["familiar_masks"], 0, arm)
        best = float("inf")
        updates = 0
        history = []
        arm_started = time.monotonic()
        stop = False
        for epoch in range(design["training"]["epochs"]):
            x, y, _, mask, _ = examples(
                train, design, "train", design["familiar_masks"], epoch, arm
            )
            order = np.random.default_rng(design["seeds"]["training"] + epoch).permutation(len(x))
            model.train()
            losses = []
            for first in range(0, len(order), design["training"]["batch_size"]):
                if time.monotonic() - arm_started > design["training"]["max_seconds_per_arm"] - 15:
                    stop = True
                    break
                idx = order[first : first + design["training"]["batch_size"]]
                tx = torch.from_numpy(x[idx]).to(device)
                ty = torch.tensor(y[idx], dtype=torch.float32, device=device)
                missing = torch.tensor(~mask[idx], device=device)[:, :, None].expand_as(ty)
                optim.zero_grad(set_to_none=True)
                pred = model(tx)
                loss = (pred - ty)[missing].square().mean()
                loss.backward()
                if not torch.isfinite(loss) or not all(
                    p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    raise ValueError("nonfinite training loss/gradient")
                optim.step()
                updates += 1
                losses.append(float(loss))
            raw = predict(model, val_example[0], device)
            _, metric = completed_metrics(raw, val_example, design, design["familiar_masks"])
            row = dict(
                epoch=epoch + 1,
                complete_epoch=not stop,
                updates=updates,
                train_mse=float(np.mean(losses)) if losses else None,
                validation=metric,
                elapsed_seconds=time.monotonic() - arm_started,
            )
            history.append(row)
            if metric["relative_l2"] < best:
                best = metric["relative_l2"]
                best_epoch = epoch + 1
                torch.save(
                    {k: t.detach().cpu() for k, t in model.state_dict().items()},
                    args.output / (arm + "_best.pt"),
                )
            write(args.output / (arm + "_history.json"), history)
            print(
                json.dumps(
                    dict(
                        arm=arm,
                        epoch=epoch + 1,
                        updates=updates,
                        validation_relative_l2=metric["relative_l2"],
                        elapsed=row["elapsed_seconds"],
                    )
                ),
                flush=True,
            )
            if stop:
                break
        model.load_state_dict(
            torch.load(args.output / (arm + "_best.pt"), map_location=device, weights_only=True)
        )
        models[arm] = model
        training[arm] = dict(
            best_epoch=best_epoch,
            updates=updates,
            completed_epochs=sum(r["complete_epoch"] for r in history),
            duration_seconds=time.monotonic() - arm_started,
            parameters=sum(p.numel() for p in model.parameters()),
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            time_limit_exit=stop,
        )
        histories[arm] = history
    familiar = evaluate_set(
        models, val, design, design["familiar_masks"], device, args.output / "familiar.npz"
    )
    selected = min(design["arms"], key=lambda a: familiar[a]["relative_l2"])
    baseline = familiar["interpolation"]
    chosen = familiar[selected]
    gate = design["integration_gate"]
    improved = sum(
        a["relative_l2"] < b["relative_l2"]
        for a, b in zip(chosen["per_file"], baseline["per_file"])
    )
    proceed = (
        chosen["relative_l2"]
        <= baseline["relative_l2"] * (1 - gate["minimum_mean_relative_improvement"])
        and improved >= gate["minimum_files_improved"]
        and chosen["mse"] <= baseline["mse"]
    )
    selection = dict(
        selected_arm=selected,
        checkpoints={a: sha256(args.output / (a + "_best.pt")) for a in design["arms"]},
        files_improved=improved,
        relative_improvement=1 - chosen["relative_l2"] / baseline["relative_l2"],
        integration_approved_by_frozen_rule=bool(proceed),
        familiar=familiar,
        training=training,
    )
    write(args.output / "selection.json", selection)
    # Settings/checkpoints committed to a hashed record before shift outcomes are computed.
    frozen_selection = sha256(args.output / "selection.json")
    shifts = evaluate_set(
        models, val, design, design["shift_masks"], device, args.output / "shifts.npz"
    )
    if frozen_selection != sha256(args.output / "selection.json"):
        raise ValueError("selection changed after shifts")
    report = dict(
        selection=selection,
        selection_sha256=frozen_selection,
        shifts=shifts,
        elapsed_seconds=time.monotonic() - started,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        reserved_evaluated=False,
        claim="deterministic supervised sparse completion on four validation deformations; no posterior calibration",
    )
    report["artifacts_sha256"] = {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()}
    write(args.output / "report.json", report)
    print(
        json.dumps(
            dict(
                selected=selected,
                integration=proceed,
                familiar={k: v["relative_l2"] for k, v in familiar.items()},
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
