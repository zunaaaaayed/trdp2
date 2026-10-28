"""Frozen 16-Gaussian density audit and joint/staged development comparison."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from experiments.worm_uncertainty.core.cholesky_gaussian import projection
from experiments.worm_uncertainty.core.measured_initializer import perturb
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.core.staged_density import fit_stages, solve_density
from experiments.worm_uncertainty.studies.worm_fit_controls import stability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=resource_path("worm_staged_protocol_v1.json")
    )
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/worm_fit_controls_v1")
    )
    parser.add_argument(
        "--pilot", type=Path, default=Path("output/worm_uncertainty/worm_shape_pilot_v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if not args.execute:
        print(json.dumps(design, indent=2))
        return
    if args.output.exists():
        raise ValueError("output exists; preserve it and choose a new path")
    if sha256(args.parent / "report.json") != design["parent_report_sha256"]:
        raise ValueError("parent report hash mismatch")
    parent = json.loads((args.parent / "report.json").read_text())
    if sha256(args.pilot / "report.json") != parent["parent_report_sha256"]:
        raise ValueError("pilot report hash mismatch")
    pilot = json.loads((args.pilot / "report.json").read_text())
    for directory, report, names in [
        (args.parent, parent, ["fits.json", "references.npz"]),
        (args.pilot, pilot, ["inputs.npz"]),
    ]:
        for name in names:
            if sha256(directory / name) != report["artifacts_sha256"][name]:
                raise ValueError("artifact hash mismatch: " + name)
    cfg = dict(pilot["protocol"], components=design["components"])
    if cfg["components"] != 16 or design["seeds"] != cfg["initialization_seeds"]:
        raise ValueError("frozen 16-component/four-start comparison required")
    with np.load(args.pilot / "inputs.npz") as d:
        data = {key: d[key] for key in d.files}
    with np.load(args.parent / "references.npz") as d:
        reference = d["16"]
    angles, uv, measured = [data[key] for key in ["angles", "uv", "measured"]]
    mask = measured.reshape(len(angles), len(uv))[:, 0]
    if not np.array_equal(measured, np.repeat(mask, len(uv))):
        raise ValueError("complete acquired views required")
    y = data["noisy"][measured]
    acquired_angles = angles[mask]
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    source_names = [
        "worm_staged.py",
        "staged_density.py",
        "cholesky_gaussian.py",
        "measured_initializer.py",
        "worm_fit_controls.py",
        "prepare.py",
        "shape_gaussian.py",
        "rotation_sensitivity.py",
        "stress_3d.py",
        "diagnose_3d.py",
    ]
    provenance = dict(
        protocol=design,
        configuration=cfg,
        source_sha256={n: sha256(resource_path(n)) for n in source_names},
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        host=platform.node(),
        parent_report_sha256=sha256(args.parent / "report.json"),
        input_sha256=sha256(args.pilot / "inputs.npz"),
        references_sha256=sha256(args.parent / "references.npz"),
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    started = time.monotonic()
    audits = []
    old_fits = json.loads((args.parent / "fits.json").read_text())
    for fit in old_fits:
        if fit["arm"] != "new_start_new_prior_16_450":
            continue
        p = np.array(fit["parameters"])
        basis = projection(p, acquired_angles, uv)[1][:, 144:]
        a = p[144:]
        optimum, info = solve_density(
            basis, y, cfg["noise_sigma"], cfg["density_prior_sd"], cfg["density_bounds"]
        )

        def cost(density):
            return float(
                np.sum(((basis @ density - y) / cfg["noise_sigma"]) ** 2)
                + np.sum((density / cfg["density_prior_sd"]) ** 2)
            )

        before, after = cost(a), cost(optimum)
        if after > before + 1e-8 * max(1, before):
            raise RuntimeError("conditional optimum increased objective")
        audits.append(
            dict(
                seed=fit["seed"],
                before=before,
                after=after,
                improvement=before - after,
                solver=info,
            )
        )
    (args.output / "density_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    fits, rows = [], []
    for arm in design["arms"]:
        arm_fits = []
        for seed in design["seeds"]:
            fit = fit_stages(
                y, acquired_angles, uv, perturb(reference, seed, cfg), reference, cfg, arm["stages"]
            )
            fit.update(arm=arm["name"], seed=seed)
            fits.append(fit)
            arm_fits.append(fit)
            (args.output / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in fit.items() if k not in ["parameters", "trace"]}),
                flush=True,
            )
        selected, gate = stability(arm_fits, acquired_angles, uv, cfg["stability_gate"])
        prediction = projection(np.array(selected["parameters"]), angles, uv)[0]
        # Evaluation happens only after acquired-only selection; no feedback to fitting.
        clean = data["clean"]
        errors = {
            name: float(np.linalg.norm((prediction - clean)[m]) / np.linalg.norm(clean[m]))
            for name, m in [("measured_relative_l2", measured), ("withheld_relative_l2", ~measured)]
        }
        rows.append(
            dict(
                arm=arm["name"],
                gate=gate,
                **errors,
                objective=selected["objective"],
                elapsed_seconds=sum(f["elapsed_seconds"] for f in arm_fits),
                projection_jacobian_calls=sum(f["projection_jacobian_calls"] for f in arm_fits),
            )
        )
    report = dict(
        **provenance,
        rows=rows,
        density_audit=audits,
        elapsed_seconds=time.monotonic() - started,
        artifacts_sha256={
            n: sha256(args.output / n)
            for n in ["fits.json", "density_audit.json", "protocol.json", "provenance.json"]
        },
        interpretation="Development optimization diagnostic; no posterior or transfer claim.",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
