"""Summarize known-truth fits with exact Gaussian L2 and unchanged stationarity gate."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.worm_uncertainty.core.gaussian_l2 import relative_l2
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.input / "report.json").read_text())
    for n, h in report["artifacts_sha256"].items():
        if sha256(args.input / n) != h:
            raise ValueError("artifact mismatch: " + n)
    fits = json.loads((args.input / "fits.json").read_text())
    data = np.load(args.input / "inputs.npz")
    rows = []
    for f in fits:
        r = {k: v for k, v in f.items() if k not in ["parameters", "before", "after"]}
        r.update(
            exact_volume_relative_l2=relative_l2(
                np.array(f["parameters"]), data[f["shape"] + "_truth"]
            ),
            projected_gradient_inf=f["after"]["projected_gradient_inf"],
        )
        rows.append(r)
    groups = [
        (shape, span, noisy)
        for shape in ["moderate", "elongated"]
        for span in [180, 120]
        for noisy in [False, True]
    ]
    labels = [f"{s[:3]} {span}°\n" + ("noise" if n else "clean") for s, span, n in groups]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for seed, marker, label in [(0, "o", "Exact start"), (81, "x", "Perturbed start")]:
        rs = [
            next(r for r in rows if (r["shape"], r["span"], r["noisy"], r["seed"]) == (*g, seed))
            for g in groups
        ]
        axes[0].plot(
            range(8),
            [max(r["exact_volume_relative_l2"] * 100, 1e-8) for r in rs],
            marker + "-",
            label=label,
        )
        axes[1].plot(range(8), [r["projected_gradient_inf"] for r in rs], marker + "-", label=label)
    axes[0].set(
        yscale="log",
        ylabel="Exact full-space density L2 error (%) — lower is better",
        title="Known truth: no voxel-grid truncation",
    )
    axes[1].axhline(1e-6, color="red", ls="--", label="Frozen threshold")
    axes[1].set(
        yscale="log",
        ylabel="Projected-gradient norm — lower is better",
        title="Stationarity is distinct from reconstruction accuracy",
    )
    for ax in axes:
        ax.set_xticks(range(8), labels, rotation=35)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Oracle geometry prior · four Gaussian phantom · one noise draw · no calibration claim"
    )
    fig.tight_layout()
    args.output.mkdir(parents=True)
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    summary = dict(
        rows=rows,
        report_sha256=sha256(args.input / "report.json"),
        metric_code_sha256=sha256(resource_path("gaussian_l2.py")),
        plot_code_sha256=sha256(__file__),
        protocol=report["protocol"],
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in rows:
        print(
            r["shape"],
            r["span"],
            r["noisy"],
            r["seed"],
            "pass",
            r["converged"],
            "L2%",
            round(100 * r["exact_volume_relative_l2"], 5),
            "gradient",
            r["projected_gradient_inf"],
        )


if __name__ == "__main__":
    main()
