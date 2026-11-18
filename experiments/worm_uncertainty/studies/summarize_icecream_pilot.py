"""Export compact, hash-linked pilot records without copying volumes or checkpoints."""

import argparse
import csv
import json
from pathlib import Path

from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.studies.icecream_pilot import DEFAULT_ROOT
from experiments.worm_uncertainty.studies.verify_icecream_pilot import verify


def summarize(root, output):
    verify(root)
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        "protocol.json",
        "training_freeze.json",
        "precision_amendment.json",
        "geometry/report.json",
        "profile/report.json",
        "operator_audit/report.json",
        "precision_check/report.json",
        "resource_snapshot.json",
        "runs/worm004_120/report.json",
    ]
    paths += [str(p.relative_to(root)) for p in sorted(root.glob("inputs/*/report.json"))]
    for group in ["evaluation", "evaluation_fp32", "runs", "runs_fp32"]:
        paths += [str(p.relative_to(root)) for p in sorted((root / group).glob("*/report.json"))]
    for group in ["runs", "runs_fp32"]:
        for name in ["method.json", "input_contract.json"]:
            paths += [str(p.relative_to(root)) for p in sorted((root / group).glob("*/" + name))]
    records = {p: json.loads((root / p).read_text()) for p in sorted(set(paths))}
    hashes = {p: sha256(root / p) for p in records}
    provenance = {
        str(p.relative_to(root)): json.loads(p.read_text())
        for p in sorted(root.rglob("provenance.json"))
    }
    rows = []
    for p, record in records.items():
        if p.startswith(("evaluation/", "evaluation_fp32/")):
            rows.extend(dict(case=record["case"]["id"], **r) for r in record["rows"])
    events = []
    for ledger in ["workflow.jsonl", "workflow_fp32.jsonl"]:
        events += [
            dict(ledger=ledger, **json.loads(line))
            for line in (root / ledger).read_text().splitlines()
        ]
    training_process_seconds = sum(e["seconds"] for e in events if e["stage"] == "train")
    cpu_stage_process_seconds = sum(e["seconds"] for e in events if e["stage"] != "train")
    summary = dict(
        source_root=str(root),
        report_hashes=hashes,
        records=records,
        provenance=provenance,
        workflow_events=events,
        resource_accounting=dict(
            training_and_inference_process_seconds_including_failure=training_process_seconds,
            workflow_cpu_stage_process_seconds=cpu_stage_process_seconds,
            input_preparation_seconds=sum(
                record["seconds"] for p, record in records.items() if p.startswith("inputs/")
            ),
            geometry_seconds=records["geometry/report.json"]["seconds"],
            operator_audit_seconds=records["operator_audit/report.json"]["seconds"],
            gpu_profile_seconds=records["profile/report.json"]["seconds"],
            precision_smoke_training_seconds=records["precision_check/report.json"]["fp32_seconds"],
            note="Process wall time, not utilization-weighted GPU time. Add profile and precision diagnostics; CPU verification and plotting are separate.",
        ),
        limitations=[
            "one organism",
            "no worm voxel truth",
            "unresolved simulator scaling",
            "Fourier/discrete-CT mismatch",
            "short per-scan training",
            "one training seed",
            "no uncertainty calibration",
        ],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["resource_accounting"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.output)
