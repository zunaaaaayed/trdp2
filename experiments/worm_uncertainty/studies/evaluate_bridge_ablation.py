"""Frozen evaluation of both selected checkpoints after paired training completes."""

import csv
import json
from pathlib import Path
import numpy as np
import torch
from experiments.worm_uncertainty.data.acquisition_diversity import read_design, noisy_bank
from experiments.worm_uncertainty.data.matched_completion import load_clean
from experiments.worm_uncertainty.data.completion_data import (
    acquisition_mask,
    channels,
    interpolate_missing,
)
from experiments.worm_uncertainty.core.completion_model import CompletionCNN
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector, relative_l2
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    root = Path("output/worm_uncertainty/bridge_ablation_v1_20261116")
    cfg = json.loads((root / "protocol.json").read_text())
    models = {}
    for arm in ["A", "B"]:
        info = json.loads((root / arm / "report.json").read_text())
        assert info["status"] == "completed"
        assert sha256(root / arm / "best.pt") == info["checkpoint_sha256"]
        model = CompletionCNN().eval()
        model.load_state_dict(
            torch.load(root / arm / "best.pt", map_location="cpu", weights_only=True)
        )
        models[arm] = model
    out = root / "evaluation"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    d, m = read_design()
    names = cfg["evaluation_files"]
    clean = load_clean(
        Path("../worm_data"), dict(d, subsets={"validation": names}), m, "validation"
    )
    noisy = noisy_bank(clean, d, names, d["noise_seeds"]["audit"], 0)
    op = ParallelProjector(62, np.arange(25) * 7.2, spacing=5)
    rows = []
    for fi, name in enumerate(names):
        for r in cfg["evaluation_masks"]:
            mask = acquisition_mask(r["start"], r["span"], np.arange(25) * 7.2)
            obs = np.where(mask[None, :, None], noisy[fi], 0)
            x = torch.tensor(np.stack([channels(v, mask, r["start"], r["span"]) for v in obs]))
            fills = {"interpolation": np.stack([interpolate_missing(v, mask) for v in obs])}
            with torch.no_grad():
                for arm, model in models.items():
                    fills[arm] = model(x).numpy()
            volumes = {"SART": op.sart(obs, mask)[0]}
            for arm, pred in fills.items():
                fills[arm] = np.where(mask[None, :, None], obs, pred)
                np.testing.assert_array_equal(fills[arm][:, mask], obs[:, mask])
                volumes[arm] = op.fbp(fills[arm], np.ones(25, bool), 7.2)
            for arm, vol in volumes.items():
                proj = op.forward(vol)
                rows.append(
                    dict(
                        file=name,
                        **r,
                        arm=arm,
                        raw=relative_l2(fills[arm][:, ~mask], clean[fi][:, ~mask])
                        if arm in fills
                        else None,
                        withheld=relative_l2(proj[:, ~mask], clean[fi][:, ~mask]),
                        acquired=relative_l2(proj[:, mask], obs[:, mask]),
                        rmse=float(np.sqrt(np.mean((proj[:, mask] - obs[:, mask]) ** 2))),
                    )
                )
    with (out / "metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        arm: {
            key: float(np.mean([r[key] for r in rows if r["arm"] == arm and r["span"] == 90]))
            for key in ["withheld", "acquired", "rmse"]
        }
        for arm in volumes
    }
    wins = sum(
        np.mean(
            [r["withheld"] for r in rows if r["file"] == n and r["arm"] == "B" and r["span"] == 90]
        )
        < np.mean(
            [r["withheld"] for r in rows if r["file"] == n and r["arm"] == "A" and r["span"] == 90]
        )
        for n in names
    )
    passed = (
        summary["A"]["withheld"] - summary["B"]["withheld"] >= 0.02
        and summary["B"]["acquired"] - summary["A"]["acquired"] <= 0.01
        and wins > len(names) / 2
    )
    result = dict(primary=summary, wins=int(wins), gate=bool(passed))
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
