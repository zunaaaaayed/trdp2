"""Fixed longer SART schedule using saved phantom inputs, with no retraining."""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
import zipfile
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector, relative_l2
from experiments.worm_uncertainty.data.phantom_density import density_metrics, FAMILIES
from experiments.worm_uncertainty.data.prepare import sha256

PROTOCOL = Path("experiments/worm_uncertainty/protocols/sart_duration_v1.json")
DEFAULT = Path("output/worm_uncertainty/sart_duration_v1_20261116")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def summarize(rows, arm, sweep, span):
    rr = [r for r in rows if r["arm"] == arm and r["sweep"] == sweep and r["span"] == span]
    ids = sorted({r["object"] for r in rr})
    keys = [
        "density_l2",
        "withheld_l2",
        "acquired_l2",
        "acquired_rmse",
        "boundary_um",
        "interior_bias",
        "interior_mae",
    ]
    return {
        key: float(np.mean([np.mean([r[key] for r in rr if r["object"] == i]) for i in ids]))
        if all(r[key] is not None for r in rr)
        else None
        for key in keys
    }


def comparisons(rows, cfg):
    output = {}
    for span in [90, 120]:
        output[str(span)] = {}
        for comparator in ["zero", "interpolation"]:
            at = {}
            for sweep in [5, 100]:
                a = summarize(rows, comparator, sweep, span)
                b = summarize(rows, "learned", sweep, span)
                differences = []
                for i in range(12):
                    values = {
                        arm: np.mean(
                            [
                                r["density_l2"]
                                for r in rows
                                if r["arm"] == arm
                                and r["sweep"] == sweep
                                and r["span"] == span
                                and r["object"] == i
                            ]
                        )
                        for arm in [comparator, "learned"]
                    }
                    differences.append(float(values[comparator] - values["learned"]))
                gain = a["density_l2"] - b["density_l2"]
                degradation = b["acquired_l2"] - a["acquired_l2"]
                wins = sum(v > 0 for v in differences)
                at[str(sweep)] = dict(
                    density_gain=gain,
                    acquired_degradation=degradation,
                    wins=wins,
                    objects=12,
                    per_object_gain=differences,
                    passed=bool(gain >= 0.02 and degradation <= 0.01 and wins > 6),
                )
            output[str(span)][comparator] = at
    return output


def measure(volume, truth, obs, clean, mask, op, arm, sweep, r):
    pred = op.forward(volume)
    return [
        dict(
            object=i,
            family=FAMILIES[i % 4],
            start=r["start"],
            span=r["span"],
            orientation="familiar" if r["start"] == 0 else "held",
            arm=arm,
            sweep=sweep,
            **density_metrics(x, t),
            withheld_l2=relative_l2(p[~mask], c[~mask]),
            acquired_l2=relative_l2(p[mask], y[mask]),
            acquired_rmse=float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2))),
        )
        for i, (x, t, p, c, y) in enumerate(zip(volume, truth, pred, clean, obs))
    ]


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cfg = json.loads(PROTOCOL.read_text())
    write(root / "protocol.json", cfg)
    parent = Path(cfg["parent"])
    parent_report = json.loads((parent / "summary.json").read_text())
    assert sha256(parent / "evaluation_arrays.npz") == parent_report["arrays_sha256"]
    source_paths = [
        PROTOCOL,
        Path(__file__),
        Path("experiments/worm_uncertainty/core/voxel_parallel.py"),
        Path("experiments/worm_uncertainty/data/phantom_density.py"),
    ]
    with zipfile.ZipFile(root / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in source_paths:
            archive.write(p, str(p))
    write(
        root / "provenance.json",
        dict(
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            dirty=subprocess.check_output(["git", "status", "--porcelain"], text=True),
            sources={str(p): sha256(p) for p in source_paths},
            parent_files={
                str(parent / name): sha256(parent / name)
                for name in [
                    "protocol.json",
                    "masks.json",
                    "evaluation_arrays.npz",
                    "metrics.csv",
                    "best.pt",
                ]
            },
            host=platform.node(),
            python=platform.python_version(),
            numpy=np.__version__,
            reserved_access=False,
        ),
    )
    (root / "dirty.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    arrays = np.load(parent / "evaluation_arrays.npz")
    truth = arrays["truth"]
    clean = arrays["clean"]
    observed = arrays["observed"]
    filled = arrays["filled"]
    masks = json.loads((parent / "masks.json").read_text())["evaluation"]
    op = ParallelProjector(62, np.arange(25) * 7.2, spacing=5)
    historical = list(csv.DictReader((parent / "metrics.csv").open()))
    rows = []
    diagnostics = []
    saved = {}
    separations = []
    input_hashes = []
    reported = cfg["schedule"]["reported_sweeps"]
    schedule = sorted(set(reported + [99]))
    mapping = {
        "zero": "sart_zero",
        "interpolation": "sart_interpolation_init",
        "learned": "sart_learned_init",
    }
    for mi, r in enumerate(masks):
        mask = np.array(r["mask"])
        obs = observed[mi]
        inputs = {
            "zero": np.zeros_like(truth),
            "interpolation": op.fbp(filled[mi, 0], np.ones(25, bool), 7.2),
            "learned": op.fbp(filled[mi, 1], np.ones(25, bool), 7.2),
        }
        terminal = {}
        for arm, initial in inputs.items():
            current = np.maximum(initial * op.support[None], 0)
            previous_sweep = 0
            at50 = None
            at99 = None
            input_hashes.append(
                dict(
                    mask_index=mi,
                    arm=arm,
                    observed_sha256=digest(obs),
                    acquired_sha256=digest(obs[:, mask]),
                    initializer_sha256=digest(current),
                )
            )
            for sweep in schedule:
                if time.monotonic() - started > cfg["budget"]["wall_seconds"]:
                    raise TimeoutError("fixed CPU budget exhausted; no extension")
                if sweep:
                    current, _ = op.sart(
                        obs,
                        mask,
                        sweeps=sweep - previous_sweep,
                        relaxation=cfg["schedule"]["relaxation"],
                        initial=current,
                    )
                previous_sweep = sweep
                if sweep == 50:
                    at50 = op.forward(current)[:, mask] - obs[:, mask]
                if sweep == 99:
                    at99 = current.copy()
                if sweep in reported:
                    rr = measure(current, truth, obs, clean, mask, op, arm, sweep, r)
                    rows.extend(rr)
                if sweep in [5, 100]:
                    saved[f"{arm}_m{mi}_s{sweep}"] = current.astype(np.float32)
                if sweep == 5:
                    for row in rr:
                        original = next(
                            v
                            for v in historical
                            if int(v["object"]) == row["object"]
                            and float(v["start"]) == r["start"]
                            and float(v["span"]) == r["span"]
                            and v["method"] == mapping[arm]
                        )
                        for key in [
                            "density_l2",
                            "withheld_l2",
                            "acquired_l2",
                            "boundary_um",
                            "interior_bias",
                            "interior_mae",
                        ]:
                            np.testing.assert_allclose(
                                row[key],
                                float(original[key]),
                                rtol=cfg["historical_parity"]["sweep5_metric_relative_tolerance"],
                                atol=cfg["historical_parity"]["absolute_tolerance"],
                            )
            residual = op.forward(current)[:, mask] - obs[:, mask]
            for i in range(len(truth)):
                change = float(
                    np.linalg.norm(current[i] - at99[i]) / max(np.linalg.norm(current[i]), 1e-12)
                )
                residual_change = float(
                    abs(np.linalg.norm(residual[i]) - np.linalg.norm(at50[i]))
                    / max(np.linalg.norm(at50[i]), 1e-12)
                )
                d = cfg["stability_diagnostics"]
                diagnostics.append(
                    dict(
                        object=i,
                        mask_index=mi,
                        arm=arm,
                        last_sweep_change=change,
                        residual_change_50_to_100=residual_change,
                        stable=bool(
                            change <= d["terminal_one_sweep_relative_change_max"]
                            and residual_change
                            <= d["acquired_residual_relative_change_50_to_100_max"]
                        ),
                    )
                )
            terminal[arm] = current
        for arm in ["interpolation", "learned"]:
            for i in range(len(truth)):
                separations.append(
                    dict(
                        object=i,
                        mask_index=mi,
                        arm=arm,
                        relative_density_distance_to_zero=float(
                            np.linalg.norm(terminal[arm][i] - terminal["zero"][i])
                            / max(np.linalg.norm(terminal["zero"][i]), 1e-12)
                        ),
                        projected_distance_to_zero=float(
                            np.linalg.norm(
                                op.forward((terminal[arm][i] - terminal["zero"][i])[None])[:, mask]
                            )
                            / max(np.linalg.norm(obs[i, mask]), 1e-12)
                        ),
                    )
                )
    with (root / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(root / "solutions.npz", **saved)
    write(root / "input_hashes.json", input_hashes)
    write(root / "stability.json", diagnostics)
    write(root / "separation.json", separations)
    summary = dict(
        curves={
            str(span): {
                str(sweep): {arm: summarize(rows, arm, sweep, span) for arm in inputs}
                for sweep in reported
            }
            for span in [90, 120]
        },
        comparisons=comparisons(rows, cfg),
        stable_cases=sum(r["stable"] for r in diagnostics),
        diagnostic_cases=len(diagnostics),
        seconds=time.monotonic() - started,
        solutions_sha256=sha256(root / "solutions.npz"),
        historical_sweep5_parity="passed",
    )
    write(root / "summary.json", summary)
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if used > cfg["budget"]["artifact_cap_bytes"]:
        raise RuntimeError("artifact cap exceeded")
    write(root / "status.json", dict(status="completed", artifact_bytes=used))
    print(
        json.dumps(
            dict(
                terminal90=summary["curves"]["90"]["100"],
                comparisons90=summary["comparisons"]["90"],
                stable_cases=summary["stable_cases"],
                cases=summary["diagnostic_cases"],
                seconds=summary["seconds"],
                artifact_bytes=used,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT)
    a = p.parse_args()
    try:
        run(a.root)
    except Exception as exc:
        if a.root.is_dir() and not (a.root / "status.json").exists():
            write(a.root / "failure.json", dict(error=repr(exc)))
        raise
