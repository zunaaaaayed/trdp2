"""Independent regeneration of saved density benchmark metrics and acquired identity."""

import argparse
import csv
import json
import zipfile
from pathlib import Path
import numpy as np
import torch
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector, relative_l2
from experiments.worm_uncertainty.core.completion_model import CompletionCNN
from experiments.worm_uncertainty.data.phantom_density import objects, density_metrics
from experiments.worm_uncertainty.data.completion_data import channels
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.studies.phantom_density import DEFAULT, aggregate, gate


def verify(root):
    cfg = json.loads((root / "protocol.json").read_text())
    prov = json.loads((root / "provenance.json").read_text())
    training = json.loads((root / "training.json").read_text())
    assert sha256(root / "best.pt") == training["checkpoint_sha256"]
    assert training["updates"] == cfg["training"]["updates"]
    with zipfile.ZipFile(root / "sources.zip") as archive:
        for path, digest in prov["sources"].items():
            relative = Path(path).relative_to(Path.cwd())
            assert __import__("hashlib").sha256(archive.read(str(relative))).hexdigest() == digest
    summary = json.loads((root / "summary.json").read_text())
    assert sha256(root / "evaluation_arrays.npz") == summary["arrays_sha256"]
    arrays = np.load(root / "evaluation_arrays.npz")
    truth = objects(cfg["splits"]["evaluation"])
    np.testing.assert_array_equal(truth, arrays["truth"])
    op = ParallelProjector(62, np.arange(25) * 7.2, spacing=5)
    clean = op.forward(truth)
    np.testing.assert_array_equal(clean, arrays["clean"])
    noisy = clean + np.random.default_rng(cfg["noise"]["evaluation_seed"]).normal(
        0, 0.05, clean.shape
    )
    masks = json.loads((root / "masks.json").read_text())["evaluation"]
    model = CompletionCNN().eval()
    model.load_state_dict(torch.load(root / "best.pt", map_location="cpu", weights_only=True))
    torch.set_num_threads(1)
    records = list(csv.DictReader((root / "metrics.csv").open()))
    assert len(records) == 12 * 6 * len(summary["primary"])
    numeric = []
    for j, r in enumerate(masks):
        mask = np.array(r["mask"])
        obs = np.where(mask[None, :, None], noisy, 0)
        np.testing.assert_array_equal(obs, arrays["observed"][j])
        for k in [0, 1]:
            np.testing.assert_array_equal(arrays["filled"][j, k][:, mask], obs[:, mask])
        # No truth/reference enters this prediction regeneration.
        x = torch.tensor(np.stack([channels(v, mask, r["start"], r["span"]) for v in obs]))
        with torch.no_grad():
            pred = model(x).numpy()
        np.testing.assert_array_equal(
            np.where(mask[None, :, None], obs, pred), arrays["filled"][j, 1]
        )
        for method in summary["primary"]:
            volume = arrays[method][j]
            projection = op.forward(volume)
            for i in range(len(truth)):
                row = next(
                    z
                    for z in records
                    if int(z["object"]) == i
                    and float(z["start"]) == r["start"]
                    and float(z["span"]) == r["span"]
                    and z["method"] == method
                )
                metrics = density_metrics(volume[i], truth[i])
                metrics.update(
                    withheld_l2=relative_l2(projection[i, ~mask], clean[i, ~mask]),
                    acquired_l2=relative_l2(projection[i, mask], obs[i, mask]),
                    acquired_rmse=float(
                        np.sqrt(np.mean((projection[i, mask] - obs[i, mask]) ** 2))
                    ),
                )
                if method in ["interpolation_fbp", "learned_fbp"]:
                    k = 0 if method == "interpolation_fbp" else 1
                    metrics["raw_completion_l2"] = relative_l2(
                        arrays["filled"][j, k, i, ~mask], clean[i, ~mask]
                    )
                for key, value in metrics.items():
                    if key == "boundary_defined":
                        assert str(value) == row[key]
                    elif value is None:
                        assert row[key] == ""
                    else:
                        np.testing.assert_allclose(
                            value, float(row[key]), rtol=2e-6, atol=2e-9, err_msg=method + " " + key
                        )
                rr = dict(row)
                rr.update(object=i, start=r["start"], span=r["span"], **metrics)
                numeric.append(rr)
    for method in summary["primary"]:
        for section, span in [("primary", 90), ("moderate", 120)]:
            metrics = aggregate(numeric, method, span)
            for key, value in metrics.items():
                np.testing.assert_allclose(
                    value, summary[section][method][key], rtol=2e-6, atol=2e-9
                )
    assert gate(numeric, cfg)["passed"] == summary["gate"]["passed"]
    assert (
        sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        < cfg["budget"]["run_artifact_cap_bytes"]
    )
    print(
        "PASS: 432 method/object/mask rows; archived sources, checkpoint, noise, measured identity, observed-only predictions, density/boundary/projection metrics and gate"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT)
    a = p.parse_args()
    verify(a.root)
