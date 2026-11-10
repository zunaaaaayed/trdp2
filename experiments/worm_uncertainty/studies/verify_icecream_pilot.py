"""Regenerate ICECREAM pilot metrics and check inputs, snapshots and budget records."""

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.studies.icecream_pilot import DEFAULT_ROOT, read_protocol
from experiments.worm_uncertainty.core.voxel_parallel import (
    ParallelProjector,
    reconstruction_metrics,
)
from experiments.worm_uncertainty.data.alternative_reconstruction import (
    allowed_record,
    noisy_pair,
    mask_record,
)
from experiments.worm_uncertainty.data.completion_data import acquisition_mask
from experiments.worm_uncertainty.data.prepare import sha256


def verify(root, require_complete=True):
    design = read_protocol(root)
    frozen = json.loads((root / "training_freeze.json").read_text())
    count = 0
    for provenance in root.rglob("provenance.json"):
        record = json.loads(provenance.read_text())
        for relative, digest in record["source_hashes"].items():
            if sha256(provenance.parent / "source" / relative) != digest:
                raise ValueError("executed source archive mismatch")
    for case in design["cases"]:
        if case["kind"] == "worm":
            allowed_record(case["file"])
        amended = case["kind"] == "worm" and (root / "precision_amendment.json").exists()
        evaluation_group = "evaluation_fp32" if amended else "evaluation"
        run_group = "runs_fp32" if amended else "runs"
        folder = root / evaluation_group / case["id"]
        if not (folder / "report.json").exists():
            if require_complete:
                raise ValueError("missing evaluated case: " + case["id"])
            continue
        info = json.loads((root / "inputs" / case["id"] / "report.json").read_text())
        run = json.loads((root / run_group / case["id"] / "report.json").read_text())
        contract = json.loads((root / run_group / case["id"] / "input_contract.json").read_text())
        if amended and contract["numerical_amendment_sha256"] != sha256(
            root / "precision_amendment.json"
        ):
            raise ValueError("numerical amendment changed")
        evaluation = json.loads((folder / "report.json").read_text())
        if evaluation["input_report_sha256"] != sha256(
            root / "inputs" / case["id"] / "report.json"
        ):
            raise ValueError("evaluation input provenance changed")
        if evaluation["run_report_sha256"] != sha256(root / run_group / case["id"] / "report.json"):
            raise ValueError("evaluation training provenance changed")
        for filename, key in [
            ("checkpoint.pt", "checkpoint_sha256"),
            ("volume.npy", "volume_sha256"),
        ]:
            if sha256(root / run_group / case["id"] / filename) != run[key]:
                raise ValueError("training artifact changed: " + filename)
        if run["status"] != "completed" or run["updates"] != frozen["updates"][case["kind"]]:
            raise ValueError("incomplete fixed-budget run")
        if contract["observed_pair_sha256"] != info["observed_pair_sha256"]:
            raise ValueError("acquired pair differs")
        if contract["training_freeze_sha256"] != sha256(root / "training_freeze.json"):
            raise ValueError("checkpoint rule changed")
        inp = root / "inputs" / case["id"]
        for name, key in [
            ("observed_pair.npy", "observed_pair_sha256"),
            ("fbp_pair.npy", "fbp_pair_sha256"),
            ("sart.npy", "sart_sha256"),
            ("evaluation_only/clean.npy", "clean_evaluation_sha256"),
        ]:
            if sha256(inp / name) != info[key]:
                raise ValueError("input artifact changed: " + name)
        if mask_record(case["start"], case["span"]) != info["mask"]:
            raise ValueError("mask changed")
        clean = np.load(inp / "evaluation_only/clean.npy")
        pair = np.load(inp / "observed_pair.npy")
        mask = acquisition_mask(case["start"], case["span"])
        np.testing.assert_array_equal(
            pair, noisy_pair(clean, mask, design["noise"]["sigma_each"], case["noise_seeds"])
        )
        clean_for_isolation = clean.copy()
        clean_for_isolation[:, ~mask] = np.nan
        np.testing.assert_array_equal(
            pair,
            noisy_pair(
                clean_for_isolation, mask, design["noise"]["sigma_each"], case["noise_seeds"]
            ),
        )
        tp = inp / "evaluation_only/truth.npy"
        truth = np.load(tp) if tp.exists() else None
        op = ParallelProjector(info["shape"][-1], np.array(design["angles_deg"]))
        for row in evaluation["rows"]:
            method = row["method"].lower()
            vol = np.load(folder / (method + "_volume.npy"))
            values, prediction = reconstruction_metrics(vol, clean, pair.mean(0), mask, op, truth)
            for key, value in values.items():
                if value is None:
                    assert row[key] is None
                else:
                    np.testing.assert_allclose(
                        value,
                        row[key],
                        rtol=2e-7,
                        atol=1e-9,
                        err_msg=case["id"] + " " + method + " " + key,
                    )
            np.testing.assert_allclose(
                prediction, np.load(folder / (method + "_projection.npy")), rtol=1e-6, atol=1e-7
            )
            np.testing.assert_array_equal(vol[:, ~op.support], 0)
        count += 1
    print(
        f"PASS: {count} cases; archived source, input/noise/mask identity, withheld isolation, fixed updates, shared support and regenerated metrics"
    )
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    verify(a.root, not a.allow_partial)
