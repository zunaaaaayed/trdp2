"""Post-audit tie sensitivity and explicit high-error/low-disagreement diagnostics.

Primary frozen risk results remain unchanged. This evaluates the expected geometry
baseline under uniform ordering within tied angular distances; it tunes no model.
"""

import argparse
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.data.acquisition_diversity import (
    read_design,
    angular_distance,
    score,
    paired,
)
from experiments.worm_uncertainty.studies.acquisition_diversity import write
from experiments.worm_uncertainty.data.prepare import sha256


def tied_risk_curve(error, risk, fractions):
    """Expected retained error under uniform tie breaking, including fractional cutoff."""
    e = np.asarray(error).ravel()
    s = np.asarray(risk).ravel()
    order = np.argsort(s, kind="stable")
    e = e[order]
    s = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    result = []
    for fraction in fractions:
        count = max(1, int(len(s) * fraction))
        total = 0.0
        for start, end in zip(starts, ends):
            take = min(max(count - start, 0), end - start)
            if take:
                total += take * float(np.mean(e[start:end]))
            if end >= count:
                break
        result.append(total / count)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("output/worm_uncertainty/acquisition_diversity_v3")
    )
    args = parser.parse_args()
    folder = args.root / "audit"
    dest = args.root / "risk_sensitivity.json"
    if dest.exists():
        raise ValueError("output exists")
    if not (folder / "ensemble.npz").exists():
        raise ValueError("no eligible ensemble")
    d, _ = read_design()
    clean = np.load(folder / "clean.npy")
    a = np.load(folder / "ensemble.npz")
    inputs = np.load(folder / "inputs.npz")
    rows = d["inventories"]["audit"]
    names = d["roles"]["audit"]
    curves = []
    worst = None
    largest = -np.inf
    failures = []
    for f, name in enumerate(names):
        error = []
        std = []
        distance = []
        for mi, r in enumerate(rows):
            missing = ~np.array(r["mask"], bool)
            error.append(np.abs(a["mean"][f, mi] - clean[f])[:, missing].ravel())
            std.append(a["std"][f, mi][:, missing].ravel())
            distance.append(
                np.broadcast_to(angular_distance(~missing)[None, :, None], clean[f].shape)[
                    :, missing
                ].ravel()
            )
        e = np.concatenate(error)
        s = np.concatenate(std)
        dist = np.concatenate(distance)
        curves.append(
            dict(
                file=name,
                geometry_tie_expected=tied_risk_curve(e, dist, d["analysis"]["fractions"]),
                disagreement_tie_expected=tied_risk_curve(e, s, d["analysis"]["fractions"]),
                random_expected=[float(np.mean(e))] * 5,
            )
        )
        low = np.quantile(s, 0.25)
        high = np.quantile(e, 0.9)
        bad = (s <= low) & (e >= high)
        failures.append(
            dict(
                file=name,
                low_disagreement_threshold=float(low),
                high_error_threshold=float(high),
                fraction_of_missing_rays=float(np.mean(bad)),
                fraction_of_top_error_rays=float(np.sum(bad) / np.sum(e >= high)),
                maximum_error_in_low_disagreement=float(e[s <= low].max()),
            )
        )
        for mi, r in enumerate(rows):
            missing = np.broadcast_to(~np.array(r["mask"], bool)[None, :, None], clean[f].shape)
            err = np.abs(a["mean"][f, mi] - clean[f])
            eligible = missing & (a["std"][f, mi] <= low)
            loc = np.unravel_index(np.where(eligible, err, -np.inf).argmax(), err.shape)
            if err[loc] > largest:
                largest = float(err[loc])
                worst = dict(
                    file=name,
                    file_index=f,
                    mask_index=mi,
                    mask=r,
                    slice_index=int(loc[0]),
                    z=d["sampling"]["z_indices"][int(loc[0])],
                    angle_index=int(loc[1]),
                    detector_index=int(loc[2]),
                    absolute_error=largest,
                    disagreement=float(a["std"][f, mi][loc]),
                    low_disagreement_threshold=float(low),
                    selection_rule="post-audit diagnostic: maximum absolute missing error among rays below their file-pooled25th disagreement percentile",
                )
    ensemble = score(a["mean"], clean, rows, names)
    baseline = score(inputs["interpolation"], clean, rows, names)
    write(
        dest,
        dict(
            status="post-audit sensitivity; primary frozen results unchanged; no tuning",
            audit_report_sha256=sha256(folder / "report.json"),
            curves=curves,
            curve_means={
                key: np.mean([r[key] for r in curves], axis=0).tolist()
                for key in ["geometry_tie_expected", "disagreement_tie_expected", "random_expected"]
            },
            failures=failures,
            worst_low_disagreement=worst,
            ensemble_metrics=ensemble,
            paired=paired(ensemble, baseline),
            expected_new_independent_noisy_ray_mse={
                g: r["mse"] + d["noise_sd"] ** 2 for g, r in ensemble["groups"].items()
            },
            noisy_mse_note="Analytic clean-target MSE + fixed noise variance0.0025 under declared independent Gaussian noise; not empirical noisy coverage or calibrated intervals.",
        ),
    )
    print("Saved tie sensitivity, ensemble metrics and high-error/low-disagreement failures")


if __name__ == "__main__":
    main()
