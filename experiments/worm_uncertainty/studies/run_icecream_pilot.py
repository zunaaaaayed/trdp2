"""Persistent, resumable-by-completed-stage orchestration; never retry a failed stage."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.worm_uncertainty.studies.icecream_pilot import DEFAULT_ROOT, read_protocol


def run(root, attempt=None):
    d = read_protocol(root)
    if not (root / "training_freeze.json").is_file():
        raise ValueError("run geometry, CPU tests, preparation and training-only profile first")
    if not (root / "operator_audit/report.json").is_file():
        raise ValueError("Fourier/cropping audit must exist before training")
    budget = dict(phantom=0.0, worm=0.0, prepare=0.0)
    ledger = root / ("workflow.jsonl" if attempt is None else "workflow_fp32.jsonl")
    if attempt is not None and not (root / "precision_amendment.json").exists():
        raise ValueError("freeze and document numerical amendment first")
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            event = json.loads(line)
            if event["status"] != "completed":
                raise ValueError("previous stage failed; no automatic retry")
            budget[event["budget_group"]] += event["seconds"]
    for case in d["cases"]:
        if attempt is not None and case["kind"] != "worm":
            continue
        for stage, parent in [("prepare", "inputs"), ("train", "runs"), ("evaluate", "evaluation")]:
            if attempt is not None and parent != "inputs":
                parent += "_fp32"
            folder = root / parent / case["id"]
            if folder.exists():
                path = folder / "report.json"
                if (
                    not path.exists()
                    or json.loads(path.read_text()).get("status", "completed") != "completed"
                ):
                    raise ValueError("existing incomplete/failed stage: " + str(folder))
                continue
            group = "prepare" if stage != "train" else case["kind"]
            cap = {"phantom": 1800, "worm": 6300, "prepare": 1800}[group]
            remaining = cap - budget[group]
            if remaining < 30:
                raise RuntimeError("stage budget exhausted: " + group)
            command = [
                sys.executable,
                "-u",
                "-m",
                "experiments.worm_uncertainty.studies.icecream_pilot",
                stage,
                "--root",
                str(root),
                "--case",
                case["id"],
            ]
            if attempt is not None and stage != "prepare":
                command += ["--attempt", attempt]
            before = time.monotonic()
            print("RUN", " ".join(command), flush=True)
            try:
                completed = subprocess.run(command, timeout=remaining)
                status = "completed" if completed.returncode == 0 else "failed"
                error = None if status == "completed" else "child exit " + str(completed.returncode)
            except subprocess.TimeoutExpired:
                status, error = "failed", "hard workflow budget timeout"
            elapsed = time.monotonic() - before
            with ledger.open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            case=case["id"],
                            stage=stage,
                            budget_group=group,
                            seconds=elapsed,
                            status=status,
                            error=error,
                        )
                    )
                    + "\n"
                )
            budget[group] += elapsed
            if status != "completed":
                raise RuntimeError(error)
    print("Completed frozen pilot", json.dumps(budget), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--attempt", choices=["fp32"])
    args = parser.parse_args()
    run(args.root, args.attempt)
