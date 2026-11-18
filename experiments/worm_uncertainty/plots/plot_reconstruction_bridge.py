import json
import csv
import shutil
import hashlib
import platform
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = Path("output/worm_uncertainty")
dest = Path("docs/results/worm_uncertainty/reconstruction_bridge_v1")
dest.mkdir(parents=True, exist_ok=False)
for key, folder in [
    ("primary", "reconstruction_bridge_v1_20261116"),
    ("additional", "reconstruction_bridge_additional_v1_20261116"),
]:
    root = base / folder
    for file in ["protocol.json", "calibration.json", "metrics.csv", "summary.json"]:
        shutil.copyfile(root / file, dest / (key + "_" + file))
    a = np.load(root / "fixture.npz")
    vols = [a[k][4] for k in ["acquired_fbp", "sart", "interpolation_fbp", "learned_fbp"]]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), layout="constrained")
    lo = min(v.min() for v in vols)
    hi = max(v.max() for v in vols)
    for ax, v, title in zip(
        axs, vols, ["Acquired FBP", "SART", "Interpolation + FBP", "Learned + FBP"]
    ):
        im = ax.imshow(v, vmin=lo, vmax=hi, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axs, label="Model intensity; no worm voxel truth")
    fig.savefig(dest / (key + "_slices.png"), dpi=120)
    plt.close(fig)
    # Independent metric regeneration from the first scored example.
    rows = list(csv.DictReader(open(root / "metrics.csv")))
    from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector, relative_l2

    op = ParallelProjector(62, np.arange(25) * 7.2, spacing=5)
    mask = a["mask"]
    for method in ["acquired_fbp", "sart", "interpolation_fbp", "learned_fbp"]:
        row = next(r for r in rows if r["arm"] == method)
        pred = op.forward(a[method])
        np.testing.assert_allclose(
            relative_l2(pred[:, ~mask], a["clean"][:, ~mask]),
            float(row["withheld_error"]),
            rtol=1e-12,
        )
    print(key, "fixture metrics PASS", json.load(open(root / "summary.json"))["seconds"])
provenance = {
    "python": platform.python_version(),
    "sources": {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path("experiments/worm_uncertainty/studies/reconstruction_bridge.py"),
            Path("experiments/worm_uncertainty/core/voxel_parallel.py"),
            Path("experiments/worm_uncertainty/core/completion_model.py"),
            Path("experiments/worm_uncertainty/data/acquisition_diversity.py"),
            Path("experiments/worm_uncertainty/data/matched_completion.py"),
            Path("experiments/worm_uncertainty/data/completion_data.py"),
        ]
    },
}
(dest / "report_provenance.json").write_text(json.dumps(provenance, indent=2))
print(
    "total run bytes",
    sum(
        p.stat().st_size
        for name in [
            "reconstruction_bridge_v1_20261116",
            "reconstruction_bridge_additional_v1_20261116",
        ]
        for p in (base / name).rglob("*")
        if p.is_file()
    ),
)
