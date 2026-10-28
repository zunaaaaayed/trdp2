"""Finite-step sensitivity and linearization error; not posterior uncertainty."""

from experiments.worm_uncertainty.layout import verify_source

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.input / "report.json").read_text())
    for name, h in report["source_sha256"].items():
        verify_source(name, h)
    groups = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for shape in ["moderate", "elongated"]:
        for mode in ["density", "joint"]:
            series = []
            for radius in [0.001, 0.01, 0.1]:
                rows = [
                    p
                    for r in report["rows"]
                    if r["shape"] == shape
                    for p in r["probes"]
                    if p["mode"] == mode and p["requested_step"] == radius and not p["skipped"]
                ]
                if len(rows) != 30:
                    raise ValueError("expected all 30 sign/case probes")
                summary = dict(
                    shape=shape,
                    mode=mode,
                    radius=radius,
                    median_density_change=float(
                        np.median([p["density_relative_change"] for p in rows])
                    ),
                    median_acquired_sigma_rms=float(
                        np.median([p["acquired_change_sigma_rms"] for p in rows])
                    ),
                    median_withheld_change=float(
                        np.median([p["withheld_relative_change"] for p in rows])
                    ),
                    median_linearization_relative_error=float(
                        np.median([p["linearization_relative_error"] for p in rows])
                    ),
                )
                groups.append(summary)
                series.append(summary)
            label = f"{shape} / {mode}"
            axes[0].plot(
                [s["median_acquired_sigma_rms"] for s in series],
                [100 * s["median_density_change"] for s in series],
                "o-",
                label=label,
            )
            axes[1].plot(
                [s["radius"] for s in series],
                [s["median_linearization_relative_error"] for s in series],
                "o-",
                label=label,
            )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Acquired projection change RMS / noise SD",
        ylabel="Exact density change (%)",
        title="More density change for less measured change\nmeans weaker sensitivity, not better accuracy",
    )
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Step norm in prior-SD parameter units",
        ylabel="Relative finite-step linearization error",
        title="Lower means more accurate linear prediction\nNear-null directions can be nonlinear at tiny changes",
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(
        "Medians across 15 cases × two signs · data-only directions · not posterior intervals"
    )
    fig.tight_layout()
    args.output.mkdir(parents=True)
    fig.savefig(args.output / "sensitivity.png", dpi=160)
    plt.close(fig)
    (args.output / "summary.json").write_text(
        json.dumps(
            dict(
                groups=groups,
                protocol=report["protocol"],
                report_sha256=sha256(args.input / "report.json"),
                plot_code_sha256=sha256(__file__),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
