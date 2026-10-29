"""Frozen development summaries and sinogram/Fourier/reconstruction diagnostics."""

import argparse
import csv
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.worm_uncertainty.data.acquisition_diversity import (
    read_design,
    score,
    paired,
    preserve,
)
from experiments.worm_uncertainty.data.matched_completion import grid
from experiments.worm_uncertainty.core.completion_evaluation import (
    filtered_backprojection,
    polar_spectrum,
    density_slice,
)
from experiments.worm_uncertainty.data.prepare import sha256

COLORS = {"A": "#777777", "B": "#2075a9", "C": "#c87922", "interpolation": "#613e8d"}


def groups_table(results, baseline, groups):
    out = []
    for group in groups:
        row = dict(group=group, interpolation=baseline["groups"][group]["mean"])
        for arm in "ABC":
            values = [r["metrics"]["groups"][group]["mean"] for r in results if r["arm"] == arm]
            row[arm] = dict(
                mean=float(np.mean(values)),
                seed_sd=float(np.std(values, ddof=1)),
                seed_values=values,
            )
        out.append(row)
    return out


def draw_views(values, labels, u, angles, path, title):
    fig, axes = plt.subplots(
        3, len(values), figsize=(3.4 * len(values), 9.8), constrained_layout=True
    )
    vmax = max(np.quantile(v, 0.995) for v in values)
    vmin = min(0, min(np.min(v) for v in values))
    spectra = [polar_spectrum(v, u, angles) for v in values]
    fmax = max(v[2].max() for v in spectra)
    images = [filtered_backprojection(v, u, angles) for v in values]
    imax = max(np.quantile(v, 0.995) for v in images)
    imin = min(np.quantile(v, 0.005) for v in images)
    for i, (v, label, spec, image) in enumerate(zip(values, labels, spectra, images)):
        im = axes[0, i].imshow(
            v,
            aspect="auto",
            origin="lower",
            extent=[u[0] - (u[1] - u[0]) / 2, u[-1] + (u[1] - u[0]) / 2, -3.6, 176.4],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        axes[0, i].set_title(label)
        axes[0, i].set_xlabel("Detector u")
        axes[0, i].set_ylabel("Projection angle (degrees)")
        sc = axes[1, i].scatter(
            spec[0].ravel(),
            spec[1].ravel(),
            c=spec[2].ravel(),
            s=5,
            cmap="magma",
            vmin=0,
            vmax=fmax,
        )
        axes[1, i].set_aspect("equal")
        axes[1, i].set_xlabel("Frequency x", fontsize=9)
        axes[1, i].set_ylabel("Frequency y", fontsize=9)
        ri = axes[2, i].imshow(
            image, origin="lower", extent=[-1, 1, -1, 1], vmin=imin, vmax=imax, cmap="gray"
        )
        axes[2, i].set_xlabel("x (normalized)")
        axes[2, i].set_ylabel("y (normalized)")
    fig.colorbar(im, ax=axes[0, :], label="Normalized projection")
    fig.colorbar(sc, ax=axes[1, :], label="log(1 + |detector FT|)")
    fig.colorbar(ri, ax=axes[2, :], label="FBP value (normalized)")
    fig.suptitle(
        title + "\nFourier samples follow detector reversal geometry; clean FBP is not voxel truth",
        fontsize=13,
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return images


def examples(root, d, audit, out):
    folder = root / "audit"
    a = np.load(folder / "inputs.npz")
    clean = np.load(folder / "clean.npy")
    arm = audit["selected_arm"] or "B"
    preds = np.array([np.load(folder / f"{arm}_{seed}.npy") for seed in d["training_seeds"]])
    pred = preserve(preds.mean(0), a["observed"], d["inventories"]["audit"])
    rows = d["inventories"]["audit"]
    names = d["roles"]["audit"]
    mi = next(
        i for i, r in enumerate(rows) if r["group"] == "new_in_distribution" and r["count"] == 13
    )
    means = np.mean(
        [
            [r["relative_l2"] for r in item["metrics"]["per_case"]]
            for item in audit["results"]
            if item["arm"] == arm
        ],
        axis=0,
    )
    worst = int(np.argmax(means))
    cases = [("fixed", 0, mi), ("worst", worst // len(rows), worst % len(rows))]
    degrees, uv, _ = grid(d)
    u = np.unique(uv[:, 0])
    angles = np.deg2rad(degrees)
    zi = d["sampling"]["z_indices"].index(256)
    metadata = []
    for tag, fi, mi in cases:
        row = rows[mi]
        target = clean[fi, zi]
        values = [
            target,
            a["observed"][fi, mi, zi],
            a["interpolation"][fi, mi, zi],
            pred[fi, mi, zi],
        ]
        title = f"{tag.capitalize()}: {names[fi]}, z256, acquired {row['span']}° from {row['start']}° ({row['count']}/25 rows)"
        draw_views(
            values,
            [
                "Clean projections",
                "Acquired values (missing=0)",
                "Interpolation",
                f"{arm}: 3-seed mean",
            ],
            u,
            angles,
            out / f"{tag}_views.png",
            title,
        )
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
        mask = np.broadcast_to(np.array(row["mask"])[:, None], target.shape)
        ax[0].imshow(mask, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="gray")
        ax[0].set_title("Acquired mask: white = measured")
        errors = [np.abs(values[i] - target) for i in [2, 3]]
        vmax = max(np.quantile(e, 0.995) for e in errors)
        for j, e in enumerate(errors):
            im = ax[j + 1].imshow(
                e, origin="lower", aspect="auto", vmin=0, vmax=vmax, cmap="inferno"
            )
            ax[j + 1].set_title(["Interpolation absolute error", f"{arm} absolute error"][j])
        disagreement = np.std(preds[:, fi, mi, zi], axis=0, ddof=1)
        disagreement[mask.astype(bool)] = 0
        if audit["selected_arm"] is not None:
            imstd = ax[3].imshow(disagreement, origin="lower", aspect="auto", vmin=0, cmap="magma")
            ax[3].set_title("3-model disagreement\nNot calibrated uncertainty")
            fig.colorbar(imstd, ax=ax[3], label="Sample SD, normalized units")
        else:
            ax[3].axis("off")
            ax[3].text(0.05, 0.5, "No eligible uncertainty analysis")
        fig.colorbar(im, ax=ax[1:3], label="Absolute projection error — lower is better")
        fig.suptitle(title + "\nError scales shared; display clipping only, metrics use all values")
        fig.savefig(out / f"{tag}_errors.png", dpi=150)
        plt.close(fig)
        metadata.append(dict(panel=tag, file=names[fi], mask=row, arm=arm, z=256))
    return metadata


def refinements(root, d, out):
    folder = root / "refinement"
    if not (folder / "report.json").exists():
        return None
    report = json.loads((folder / "report.json").read_text())
    if report.get("skipped"):
        return report
    fig, axes = plt.subplots(
        1,
        len(report["cases"]),
        figsize=(6 * len(report["cases"]), 4),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, case in zip(axes[0], report["cases"]):
        for row, color in zip(case["rows"], ["#777777", "#c87922", "#2075a9"]):
            stages = [s for s in ["completion", "initialization", "refined"] if s in row["errors"]]
            ax.plot(
                stages,
                [100 * row["errors"][s]["withheld"] for s in stages],
                marker="o",
                label=row["arm"],
                color=color,
            )
        ax.set_title(case["file"])
        ax.set_ylabel("Withheld projection relative L2 (%) — lower is better")
        ax.legend()
    fig.suptitle(
        "Completion → Gaussian initialization → acquired-only refinement\nAcceptance failures remain exploratory; see per-case gates"
    )
    fig.savefig(out / "gaussian_stages.png", dpi=160)
    plt.close(fig)
    case = report["cases"][0]
    a = np.load(folder / Path(case["file"]).stem / "inputs.npz")
    fits = json.loads((folder / Path(case["file"]).stem / "fits.json").read_text())
    zi = d["sampling"]["z_indices"].index(256)
    z = a["v"][zi]
    density = []
    for row in case["rows"]:
        if row["selected_seed"] is not None:
            p = np.asarray(
                next(
                    f["parameters"]
                    for f in fits
                    if f["arm"] == row["arm"] and f["seed"] == row["selected_seed"]
                )
            )
            density.append((row["arm"], density_slice(p, z)))
    if density:
        fig, axes = plt.subplots(
            1, len(density), figsize=(4 * len(density), 4), constrained_layout=True, squeeze=False
        )
        vmax = max(np.quantile(v, 0.995) for _, v in density)
        for ax, (name, v) in zip(axes[0], density):
            im = ax.imshow(v, origin="lower", extent=[-1, 1, -1, 1], vmin=0, vmax=vmax, cmap="gray")
            ax.set_title(name)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        fig.colorbar(im, ax=axes[0, :], label="Model density (normalized)")
        fig.suptitle(
            case["file"]
            + " z256: fitted Gaussian slices\nAll selected fits fail acceptance; no deformed voxel truth"
        )
        fig.savefig(out / "gaussian_slices.png", dpi=160)
        plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("output/worm_uncertainty/acquisition_diversity_v3")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/worm_uncertainty/acquisition_diversity_v3"),
    )
    args = parser.parse_args()
    d, _ = read_design()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads((args.root / "primary/selection.json").read_text())
    baseline = json.loads((args.root / "primary/baselines.json").read_text())["interpolation"]
    audit = json.loads((args.root / "audit/report.json").read_text())
    tables = dict(
        selection=groups_table(
            selection["results"], baseline, ["familiar", "primary_narrow", "all"]
        ),
        audit=groups_table(
            audit["results"],
            audit["baselines"]["interpolation"],
            [
                "familiar",
                "primary_narrow",
                "held_orientation",
                "held_count",
                "extrapolation",
                "all",
            ],
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True, sharey=True)
    for ax, (role, table) in zip(axes, tables.items()):
        x = np.arange(len(table))
        width = 0.19
        for i, arm in enumerate(["interpolation", "A", "B", "C"]):
            values = [r[arm] if arm == "interpolation" else r[arm]["mean"] for r in table]
            err = None if arm == "interpolation" else [100 * r[arm]["seed_sd"] for r in table]
            ax.bar(
                x + (i - 1.5) * width,
                100 * np.array(values),
                width,
                label=arm,
                color=COLORS[arm],
                yerr=err,
                capsize=2,
            )
        ax.set_xticks(x, [r["group"].replace("_", "\n") for r in table], fontsize=8)
        ax.set_title(role.capitalize())
        ax.set_ylabel("Missing relative L2 (%) — lower is better")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Acquisition diversity: A restricted, B broad + geometry, C broad + mask\nBars average deformations; error bars show 3-seed SD, not organism confidence intervals"
    )
    fig.savefig(out / "grouped_errors.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for arm in "ABC":
        for i, seed in enumerate(d["training_seeds"]):
            h = json.loads((args.root / "primary" / f"{arm}_{seed}" / "history.json").read_text())
            axes[0].plot(
                [r["update"] for r in h],
                [r["loss"] for r in h],
                color=COLORS[arm],
                alpha=0.55,
                label=arm if i == 0 else None,
            )
            axes[1].plot(
                [r["update"] for r in h],
                [100 * r["selection_relative_l2"] for r in h],
                color=COLORS[arm],
                alpha=0.55,
            )
    axes[0].set_title("Training missing-ray MSE")
    axes[1].set_title("Selection missing-ray relative L2 (%)")
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Lower is better")
    axes[0].legend()
    fig.suptitle("Paired seeds and common budget; training mask difficulty differs by design")
    fig.savefig(out / "learning_curves.png", dpi=170)
    plt.close(fig)
    if audit["risk"] is not None:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for key, color in [
            ("disagreement", "#2075a9"),
            ("angular_distance", "#c87922"),
            ("random", "#777777"),
        ]:
            curves = np.array([r[key] for r in audit["risk"]])
            ax.plot(
                np.array(d["analysis"]["fractions"]) * 100,
                curves.mean(0),
                marker="o",
                color=color,
                label=key.replace("_", " "),
            )
        ax.set_xlabel("Lowest-score missing rays retained (%)")
        ax.set_ylabel("Mean absolute clean-ray error — lower is better")
        ax.legend()
        ax.set_title(
            "Error-risk ranking across 12 audit deformations\n3-model disagreement is not calibrated uncertainty"
        )
        fig.savefig(out / "risk_retention.png", dpi=170)
        plt.close(fig)
    panels = examples(args.root, d, audit, out)
    refinement = refinements(args.root, d, out)
    summary = dict(
        tables=tables,
        panels=panels,
        risk=audit["risk"],
        selected_arm=selection["selected_arm"],
        selection_results=selection["results"],
        audit_results=audit["results"],
        refinement=refinement,
        capacity=json.loads((args.root / "analysis_freeze.json").read_text())["capacity"],
        profile=json.loads((args.root / "profile/report.json").read_text()),
        historical_completion=json.loads(
            (args.root / "historical_completion/report.json").read_text()
        ),
        parent_hashes={
            str(p): sha256(p)
            for p in [
                args.root / "primary/selection.json",
                args.root / "audit/report.json",
                args.root / "training_budget.json",
                args.root / "analysis_freeze.json",
            ]
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = [
        "role",
        "arm",
        "seed",
        "file",
        "mask_hash",
        "group",
        "start",
        "span",
        "count",
        "relative_l2",
        "mse",
        "boundary_mse",
        "center_mse",
    ]
    with (out / "per_case_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for role, results, baselines in [
            (
                "selection",
                selection["results"],
                json.loads((args.root / "primary/baselines.json").read_text()),
            ),
            ("audit", audit["results"], audit["baselines"]),
        ]:
            for result in results + [dict(arm=k, seed="", metrics=v) for k, v in baselines.items()]:
                for row in result["metrics"]["per_case"]:
                    writer.writerow(dict(role=role, arm=result["arm"], seed=result["seed"], **row))
    for name in [
        "descriptive_summary.json",
        "state_before.json",
        "training_budget.json",
        "analysis_freeze.json",
        "context.json",
    ]:
        if (args.root / name).exists():
            (out / name).write_bytes((args.root / name).read_bytes())
    print(json.dumps(tables, indent=2))


if __name__ == "__main__":
    main()
