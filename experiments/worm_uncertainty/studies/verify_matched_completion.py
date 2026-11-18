"""Read-only verification of matched observations, selection and saved results."""

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.data.matched_completion import (
    evaluation_masks,
    make_observation,
    metrics,
    read_protocol,
)


def verify_hashes(folder, report):
    for name, h in report["artifacts_sha256"].items():
        if sha256(folder / name) != h:
            raise ValueError("artifact changed: " + str(folder / name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("output/worm_uncertainty/completion_matched_v2")
    )
    parser.add_argument(
        "--refinement",
        type=Path,
        default=Path("output/worm_uncertainty/completion_refinement_v2_retry1"),
    )
    args = parser.parse_args()
    report = json.loads((args.source / "report.json").read_text())
    verify_hashes(args.source, report)
    d = json.loads((args.source / "protocol.json").read_text())
    read_protocol(args.source / "protocol.json")
    inputs = np.load(args.source / "inputs.npz")
    if report["reserved_evaluated"]:
        raise ValueError("unexpected reserved evaluation")
    for kind, acquisitions in [("familiar", d["familiar_masks"]), ("shifts", d["shift_masks"])]:
        arrays = np.load(args.source / (kind + ".npz"))
        mask = evaluation_masks(arrays, d, acquisitions)
        for i, (f, mi, zi) in enumerate(arrays["identities"]):
            clean = inputs["validation_clean"][f, zi]
            np.testing.assert_array_equal(clean, arrays["target"][i])
            y, m = make_observation(
                clean,
                d,
                d["subsets"]["validation"][f],
                d["sampling"]["z_indices"][zi],
                0,
                "validation",
                *acquisitions[mi],
            )
            np.testing.assert_array_equal(m, mask[i])
            np.testing.assert_array_equal(y, arrays["measured"][i])
        observed = np.broadcast_to(mask[:, :, None], arrays["target"].shape)
        for arm in ["zero_fill", "interpolation", "mask", "geometry"]:
            np.testing.assert_array_equal(arrays[arm][observed], arrays["measured"][observed])
            evaluated = metrics(
                arrays[arm],
                arrays["target"],
                mask,
                arrays["identities"],
                d["subsets"]["validation"],
                acquisitions,
            )
            stored = (
                report["selection"]["familiar"][arm]
                if kind == "familiar"
                else report["shifts"][arm]
            )
            for key in ["relative_l2", "mse", "per_file", "per_mask_file"]:
                if evaluated[key] != stored[key]:
                    raise ValueError("metric differs: " + arm + "/" + key)
    for arm in d["arms"]:
        history = json.loads((args.source / (arm + "_history.json")).read_text())
        best = min(history, key=lambda r: r["validation"]["relative_l2"])
        if best["epoch"] != report["selection"]["training"][arm]["best_epoch"]:
            raise ValueError("checkpoint selection changed")
        if sha256(args.source / (arm + "_best.pt")) != report["selection"]["checkpoints"][arm]:
            raise ValueError("checkpoint hash changed")
    if sha256(args.source / "selection.json") != report["selection_sha256"]:
        raise ValueError("selection changed after shifts")
    if (args.refinement / "report.json").exists():
        rr = json.loads((args.refinement / "report.json").read_text())
        verify_hashes(args.refinement, rr)
        a = np.load(args.refinement / "inputs.npz")
        y = a["measured"][a["mask"]].ravel()
        for arm in ["acquired", "interpolation", "learned"]:
            np.testing.assert_array_equal(a[arm + "_observations"], y)
        for arm in ["interpolation", "learned"]:
            np.testing.assert_array_equal(a[arm][a["mask"]], a["measured"][a["mask"]])
        protocol = json.loads((args.refinement / "protocol.json").read_text())
        if (
            protocol["configuration"]["budget_nfev"] != 450
            or protocol["configuration"]["noise_sigma"] != d["noise_sd"]
        ):
            raise ValueError("refinement budget/noise changed")
        print("PASS: refinement artifacts and exact acquired arrays across all three arms")
    print(
        "PASS: input noise/grid identity, preserved acquired values, all per-file/mask metrics, checkpoint selection and artifact hashes; reserved outcomes untouched"
    )


if __name__ == "__main__":
    main()
