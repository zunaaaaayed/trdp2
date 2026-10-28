"""Plot the bounded initialization comparison without hiding failed fit gates."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("output/worm_uncertainty/completion_refinement_v2_retry1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/worm_uncertainty/completion_matched_v2"),
    )
    args = parser.parse_args()
    report_path = args.source / "report.json"
    report = json.loads(report_path.read_text())
    for name, digest in report["artifacts_sha256"].items():
        if sha256(args.source / name) != digest:
            raise ValueError("artifact changed: " + name)
    rows = report["rows"]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = [r["arm"] + "\n" + str(r["completed_fits"]) + " starts" for r in rows]
    for ax, key in zip(axes, ["acquired", "withheld"]):
        bars = ax.bar(
            x,
            [100 * r["metrics"][key] for r in rows],
            color=["#777777", "#d58b20", "#287ca5"][: len(rows)],
        )
        ax.bar_label(bars, fmt="%.2f%%", padding=3)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Latent projection relative L2 (%) — lower is better")
        ax.set_title(key.capitalize() + " rays after refinement")
        ax.set_ylim(0, max(100 * r["metrics"][key] for r in rows) * 1.25)
    fig.suptitle(
        "worm153, start0° / span120°: bounded feasibility\nAll selected fits fail acceptance; start counts differ"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "refinement.png", dpi=160)
    plt.close(fig)
    (args.output / "refinement_summary.json").write_text(
        json.dumps(
            dict(source=str(report_path), source_sha256=sha256(report_path), report=report),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
