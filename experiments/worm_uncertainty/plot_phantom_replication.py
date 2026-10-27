"""Summarize declared prior arms without selecting or filtering by test errors."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from prepare import sha256


def summarize(rows):
    summaries = []
    for shape in ["moderate", "elongated"]:
        for offset in [0, 28.8, 57.6]:
            for tau in [10.0, 100.0]:
                selected = [
                    r
                    for r in rows
                    if r["shape"] == shape
                    and r["offset"] == offset
                    and r["tau"] == tau
                    and r["seed"] == 0
                ]
                clean = [r for r in selected if r["noise_seed"] is None]
                noisy = [r for r in selected if r["noise_seed"] is not None]
                if len(clean) != 1 or len(noisy) != 5:
                    raise ValueError("incomplete replication group")
                errors = np.array([r["exact_volume_relative_l2"] for r in noisy])
                summaries.append(
                    dict(
                        shape=shape,
                        offset=offset,
                        tau=tau,
                        clean_density_error=clean[0]["exact_volume_relative_l2"],
                        noisy_density_mean=float(errors.mean()),
                        noisy_density_min=float(errors.min()),
                        noisy_density_max=float(errors.max()),
                        noisy_withheld_mean=float(
                            np.mean([r["withheld_relative_l2"] for r in noisy])
                        ),
                        stationary_noisy=sum(r["converged"] for r in noisy),
                        stationary_clean=clean[0]["converged"],
                    )
                )
    return summaries


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
    rows = report["rows"]
    keys = {(r["shape"], r["offset"], r["noise_seed"], r["tau"], r["seed"]) for r in rows}
    if len(rows) != 144 or len(keys) != 144:
        raise ValueError("144 unique fits required")
    summaries = summarize(rows)
    paired = []
    for r in rows:
        if r["tau"] != 10.0 or r["seed"] != 0:
            continue
        other = next(
            q
            for q in rows
            if q["shape"] == r["shape"]
            and q["offset"] == r["offset"]
            and q["noise_seed"] == r["noise_seed"]
            and q["tau"] == 100.0
            and q["seed"] == 0
        )
        paired.append(
            dict(
                shape=r["shape"],
                offset=r["offset"],
                noise_seed=r["noise_seed"],
                density_error_change=other["exact_volume_relative_l2"]
                - r["exact_volume_relative_l2"],
            )
        )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, shape in enumerate(["moderate", "elongated"]):
        for tau, shift in [(10.0, -0.7), (100.0, 0.7)]:
            group = [s for s in summaries if s["shape"] == shape and s["tau"] == tau]
            x = np.array([s["offset"] for s in group]) + shift
            means = np.array([s["noisy_density_mean"] for s in group]) * 100
            lower = np.array([s["noisy_density_min"] for s in group]) * 100
            upper = np.array([s["noisy_density_max"] for s in group]) * 100
            axes[i, 0].errorbar(
                x,
                means,
                yerr=[means - lower, upper - means],
                marker="o",
                capsize=4,
                label=f"Density prior SD {tau:g}",
            )
            axes[i, 1].plot(
                x,
                [max(s["clean_density_error"] * 100, 1e-8) for s in group],
                "o-",
                label=f"Density prior SD {tau:g}",
            )
        axes[i, 0].set(
            title=f"{shape.capitalize()}: noisy mean and min–max (5 draws)",
            ylabel="Exact density L2 error (%) — lower is better",
        )
        axes[i, 1].set(
            title=f"{shape.capitalize()}: clean control",
            ylabel="Exact density L2 error (%) — lower is better",
            yscale="log",
        )
        for ax in axes[i]:
            ax.set(xlabel="120° window start (degrees)", xticks=[0, 28.8, 57.6])
            ax.legend(fontsize=8)
    fig.suptitle(
        "Exact-start primary series · all fits retained · min–max bars are not confidence intervals"
    )
    fig.tight_layout()
    args.output.mkdir(parents=True)
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    summary = dict(
        groups=summaries,
        paired_prior_changes=paired,
        rows=rows,
        report_sha256=sha256(args.input / "report.json"),
        protocol=report["protocol"],
        plot_code_sha256=sha256(__file__),
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for s in summaries:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
