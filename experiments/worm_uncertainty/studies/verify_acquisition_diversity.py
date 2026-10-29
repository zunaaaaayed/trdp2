"""Regenerate saved metrics, paired arrays, checkpoint selection and provenance."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.data.acquisition_diversity import (
    read_design,
    evaluation_arrays,
    score,
    paired,
    schedule,
    disagreement,
    preserve,
)
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.core.completion_evaluation import common_starts


def archive_hashes(folder):
    meta = json.loads((folder / "provenance.json").read_text())
    with zipfile.ZipFile(folder / "executed_sources.zip") as z:
        for name, digest in meta["sources"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("executed source archive changed")


def verify(root):
    d, _ = read_design()
    primary = root / "primary"
    selection = json.loads((primary / "selection.json").read_text())
    budget = json.loads((root / "training_budget.json").read_text())
    if sha256(root / "profile/report.json") != budget["profile_sha256"]:
        raise ValueError("profile changed")
    for name, digest in json.loads((root / "profile/cache_hashes.json").read_text()).items():
        if sha256(root / "profile" / name) != digest:
            raise ValueError("training cache changed")
    for folder in [root / "profile", primary, root / "audit"]:
        archive_hashes(folder)
    frozen = json.loads((root / "analysis_freeze.json").read_text())
    if sha256(primary / "selection.json") != frozen["selection_sha256"]:
        raise ValueError("selection changed after freeze")
    audit = json.loads((root / "audit/report.json").read_text())
    for name, digest in audit["artifacts_sha256"].items():
        if sha256(root / "audit" / name) != digest:
            raise ValueError("audit artifact changed")
    if audit["reserved_read"]:
        raise ValueError("reserved access recorded")
    for role in ["selection", "audit"]:
        folder = primary if role == "selection" else root / "audit"
        clean = np.load(folder / ("selection_clean.npy" if role == "selection" else "clean.npy"))
        inputs = np.load(folder / ("selection_inputs.npz" if role == "selection" else "inputs.npz"))
        _, observed, interp = evaluation_arrays(clean, d, role)
        np.testing.assert_array_equal(observed, inputs["observed"])
        np.testing.assert_array_equal(interp, inputs["interpolation"])
        baseline = score(interp, clean, d["inventories"][role], d["roles"][role])
        stored_base = (
            json.loads((primary / "baselines.json").read_text())["interpolation"]
            if role == "selection"
            else audit["baselines"]["interpolation"]
        )
        if baseline != stored_base:
            raise ValueError("baseline metric mismatch")
        results = selection["results"] if role == "selection" else audit["results"]
        predictions = {}
        for r in results:
            key = f"{r['arm']}_{r['seed']}"
            path = (
                primary / key / "selection_prediction.npy"
                if role == "selection"
                else folder / (key + ".npy")
            )
            p = np.load(path)
            np.testing.assert_array_equal(p, preserve(p, observed, d["inventories"][role]))
            computed = score(p, clean, d["inventories"][role], d["roles"][role])
            if computed != r["metrics"] or paired(computed, baseline) != r["paired"]:
                raise ValueError("metric mismatch " + role + key)
            if role == "selection":
                if (
                    sha256(path) != r["prediction_sha256"]
                    or sha256(primary / key / "best.pt") != r["checkpoint_sha256"]
                ):
                    raise ValueError("selected artifact changed")
                history = json.loads((primary / key / "history.json").read_text())
                if (
                    min(history, key=lambda x: x["selection_relative_l2"])["update"]
                    != r["best_update"]
                ):
                    raise ValueError("checkpoint rule mismatch")
                original = np.load(primary / key / "schedule.npz")
                ids, masks, banks = schedule(
                    768,
                    budget["batch_size"],
                    budget["updates"],
                    r["seed"],
                    d["inventories"][r["arm"]],
                )
                for a, b in [
                    (original["samples"], ids),
                    (original["masks"], masks),
                    (original["noise_banks"], banks),
                ]:
                    np.testing.assert_array_equal(a, b)
            predictions.setdefault(r["arm"], []).append(p)
        if role == "audit" and selection["selected_arm"]:
            _, _, risk = disagreement(
                np.array(predictions[selection["selected_arm"]]),
                clean,
                d["inventories"][role],
                d["roles"][role],
            )
            if risk != audit["risk"]:
                raise ValueError("risk regeneration mismatch")
    for seed in d["training_seeds"]:
        schedules = [np.load(primary / f"{a}_{seed}" / "schedule.npz") for a in "ABC"]
        for r in schedules[1:]:
            np.testing.assert_array_equal(r["samples"], schedules[0]["samples"])
            np.testing.assert_array_equal(r["noise_banks"], schedules[0]["noise_banks"])
        hashes = {r["noise_sha256"] for r in selection["results"] if r["seed"] == seed}
        if len(hashes) != 1:
            raise ValueError("different training noise across paired arms")
    ref = root / "refinement/report.json"
    if ref.exists():
        rr = json.loads(ref.read_text())
        if not rr.get("skipped"):
            for case in rr["cases"]:
                folder = ref.parent / Path(case["file"]).stem
                a = np.load(folder / "inputs.npz")
                fits = json.loads((folder / "fits.json").read_text())
                if (
                    common_starts(fits, ["acquired", "interpolation", "learned"])
                    != case["common_starts"]
                ):
                    raise ValueError("start intersection changed")
                for arm in ["acquired", "interpolation", "learned"]:
                    np.testing.assert_array_equal(
                        a[arm + "_observations"], a["observed"][a["mask"]].ravel()
                    )
                for arm in ["interpolation", "learned"]:
                    np.testing.assert_array_equal(a[arm][a["mask"]], a["observed"][a["mask"]])
                for name, digest in case["artifacts_sha256"].items():
                    if sha256(folder / name) != digest:
                        raise ValueError("refinement artifact changed")
    print(
        "PASS: executed source archives, mask/noise identity, paired schedules, checkpoint rule, all per-case metrics, risk diagnostics, acquired preservation and available refinement artifacts; no reserved reads"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("output/worm_uncertainty/acquisition_diversity_v3")
    )
    args = parser.parse_args()
    verify(args.root)


if __name__ == "__main__":
    main()
