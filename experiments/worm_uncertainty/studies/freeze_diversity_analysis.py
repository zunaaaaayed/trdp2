"""Freeze capacity/ensemble decisions from selection results before audit access."""

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.data.acquisition_diversity import read_design
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.studies.acquisition_diversity import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("output/worm_uncertainty/acquisition_diversity_v3")
    )
    args = parser.parse_args()
    d, _ = read_design()
    dest = args.root / "analysis_freeze.json"
    if dest.exists() or (args.root / "audit").exists():
        raise ValueError("freeze/audit already exists")
    source = args.root / "primary/selection.json"
    s = json.loads(source.read_text())
    b = [r for r in s["results"] if r["arm"] == "B"]
    loss_changes = []
    gaps = []
    for r in b:
        history = json.loads(
            (args.root / "primary" / f"B_{r['seed']}" / "history.json").read_text()
        )
        quarter = [h for h in history if h["update"] >= 0.75 * r["planned_updates"]]
        loss_changes.append((quarter[0]["loss"] - quarter[-1]["loss"]) / quarter[0]["loss"])
        gaps.append(
            abs(quarter[-1]["selection_mse"] - quarter[-1]["loss"])
            / max(quarter[-1]["loss"], 1e-20)
        )
    narrow = float(np.mean([r["metrics"]["groups"]["primary_narrow"]["mean"] for r in b]))
    consider = (
        all(r["gate"] for r in b)
        and narrow > 0.35
        and float(np.mean(loss_changes)) < 0.01
        and float(np.mean(gaps)) < 0.1
    )
    decision = dict(
        eligible=consider,
        B_all_seeds_pass=all(r["gate"] for r in b),
        B_narrow_mean=narrow,
        mean_final_quarter_loss_reduction=float(np.mean(loss_changes)),
        mean_train_selection_mse_gap=float(np.mean(gaps)),
        reason="prospective capacity rule evaluated on selection only",
    )
    if consider:
        write(args.root / "capacity_eligible.json", decision)
        raise RuntimeError(
            "capacity condition triggered; freeze architecture and run the authorized conditional arm before audit"
        )
    write(
        dest,
        dict(
            selection_sha256=sha256(source),
            checkpoints={f"{r['arm']}_{r['seed']}": r["checkpoint_sha256"] for r in s["results"]},
            capacity=dict(tested=False, **decision),
            analysis=d["analysis"],
            selected_arm=s["selected_arm"],
            ensemble_enabled=s["selected_arm"] is not None,
            audit_inventory=d["inventories"]["audit"],
            audit_files=d["roles"]["audit"],
            reserved_read=False,
            source_sha256=sha256(Path(__file__)),
        ),
    )
    print(
        json.dumps(
            dict(capacity_skipped=True, selected_arm=s["selected_arm"], capacity_rule=decision)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
