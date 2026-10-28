"""Audit multistart stability using only convergence and acquired predictions."""

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import projection
from experiments.worm_uncertainty.studies.stress_3d import acquisition_mask
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    report = json.loads((args.input / "report.json").read_text())
    cfg = report["protocol"]
    gate = cfg["stability_gate"]
    if sha256(args.input / "draws.json") != report["draws_sha256"]:
        raise ValueError("draw checksum mismatch")
    draws = json.loads((args.input / "draws.json").read_text())
    degrees = np.arange(cfg["angles_deg"]["count"]) * cfg["angles_deg"]["step"]
    h, w = cfg["detector_shape"]
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * 2 / w - 1, (np.arange(h) + 0.5) * 2 / h - 1)
    uv = np.c_[uu.ravel(), vv.ravel()]
    records = []
    rows = []
    for case in cfg["cases"]:
        angles = np.deg2rad(degrees[acquisition_mask(degrees, case["start"], case["span"])])
        selected = []
        for repeat in range(cfg["noise_repeats"]):
            candidates = [
                r
                for r in draws
                if r["case"] == case["name"] and r["noise_repeat"] == repeat and r["converged"]
            ]
            candidates.sort(key=lambda r: r["objective"])
            record = dict(
                case=case["name"],
                noise_repeat=repeat,
                converged_candidates=len(candidates),
                passed=False,
            )
            if len(candidates) >= gate["minimum_converged"]:
                first, second = candidates[:2]
                a = projection(np.array(first["parameters"]), angles, uv)[0]
                b = projection(np.array(second["parameters"]), angles, uv)[0]
                gap = abs(first["objective"] - second["objective"]) / max(
                    1, abs(first["objective"])
                )
                disagreement = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-15))
                passed = (
                    not (first["active_indices"] or second["active_indices"])
                    and gap <= gate["relative_objective_gap"]
                    and disagreement <= gate["relative_prediction_disagreement"]
                )
                record.update(
                    passed=passed,
                    relative_objective_gap=gap,
                    relative_prediction_disagreement=disagreement,
                    selected_seed=first["initialization_seed"],
                    second_seed=second["initialization_seed"],
                    metrics=first["metrics"],
                )
                selected.append(first)
            records.append(record)
        row = dict(
            case=case["name"],
            passes=sum(r["passed"] for r in records if r["case"] == case["name"]),
            draws=cfg["noise_repeats"],
            selected_metrics={},
        )
        if selected:
            row["selected_metrics"] = {
                domain: {
                    metric: float(np.mean([r["metrics"][domain][metric] for r in selected]))
                    for metric in selected[0]["metrics"][domain]
                }
                for domain in selected[0]["metrics"]
            }
        rows.append(row)
    output = dict(
        source_report_sha256=sha256(args.input / "report.json"),
        code_sha256=sha256(__file__),
        gate=gate,
        all_passed=all(r["passed"] for r in records),
        rows=rows,
        records=records,
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    print("all_passed", output["all_passed"])
    if not output["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
