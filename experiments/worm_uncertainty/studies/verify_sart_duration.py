"""Verify saved longer-SART metrics, parent identity and the complete fixed schedule."""

import argparse
import csv
import json
import zipfile
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.core.voxel_parallel import ParallelProjector
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.studies.sart_duration import (
    DEFAULT,
    digest,
    measure,
    summarize,
    comparisons,
)


def verify(root):
    cfg = json.loads((root / "protocol.json").read_text())
    prov = json.loads((root / "provenance.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    with zipfile.ZipFile(root / "sources.zip") as archive:
        for name, checksum in prov["sources"].items():
            assert (
                __import__("hashlib").sha256(archive.read(name.lstrip("/"))).hexdigest() == checksum
            )
    for name, checksum in prov["parent_files"].items():
        assert sha256(Path(name)) == checksum
    assert sha256(root / "solutions.npz") == summary["solutions_sha256"]
    parent = Path(cfg["parent"])
    arrays = np.load(parent / "evaluation_arrays.npz")
    saved = np.load(root / "solutions.npz")
    hashes = json.loads((root / "input_hashes.json").read_text())
    masks = json.loads((parent / "masks.json").read_text())["evaluation"]
    op = ParallelProjector(62, np.arange(25) * 7.2, spacing=5)
    rows = list(csv.DictReader((root / "metrics.csv").open()))
    assert len(rows) == 1512
    regenerated = []
    for mi, r in enumerate(masks):
        mask = np.array(r["mask"])
        obs = arrays["observed"][mi]
        truth = arrays["truth"]
        clean = arrays["clean"]
        initializers = dict(
            zero=np.zeros_like(truth),
            interpolation=op.fbp(arrays["filled"][mi, 0], np.ones(25, bool), 7.2),
            learned=op.fbp(arrays["filled"][mi, 1], np.ones(25, bool), 7.2),
        )
        for arm, initial in initializers.items():
            state = np.maximum(initial * op.support[None], 0)
            previous = 0
            record = next(v for v in hashes if v["mask_index"] == mi and v["arm"] == arm)
            assert digest(state) == record["initializer_sha256"]
            assert digest(obs) == record["observed_sha256"]
            assert digest(obs[:, mask]) == record["acquired_sha256"]
            for sweep in cfg["schedule"]["reported_sweeps"]:
                if sweep:
                    state, _ = op.sart(obs, mask, sweeps=sweep - previous, initial=state)
                previous = sweep
                rr = measure(state, truth, obs, clean, mask, op, arm, sweep, r)
                regenerated.extend(rr)
                if sweep in [5, 100]:
                    np.testing.assert_array_equal(
                        state.astype(np.float32), saved[f"{arm}_m{mi}_s{sweep}"]
                    )
                    assert np.isfinite(state).all() and state.min() >= 0
                    np.testing.assert_array_equal(state[:, ~op.support], 0)
                for row in rr:
                    original = next(
                        v
                        for v in rows
                        if int(v["object"]) == row["object"]
                        and int(v["sweep"]) == sweep
                        and v["arm"] == arm
                        and float(v["start"]) == r["start"]
                        and float(v["span"]) == r["span"]
                    )
                    for key, value in row.items():
                        if isinstance(value, bool):
                            assert str(value) == original[key]
                        elif value is None:
                            assert original[key] == ""
                        elif isinstance(value, (float, int)):
                            np.testing.assert_allclose(
                                value, float(original[key]), rtol=1e-10, atol=1e-12
                            )
                        else:
                            assert value == original[key]
    for span in [90, 120]:
        for sweep in cfg["schedule"]["reported_sweeps"]:
            for arm in ["zero", "interpolation", "learned"]:
                metrics = summarize(regenerated, arm, sweep, span)
                for key, value in metrics.items():
                    target = summary["curves"][str(span)][str(sweep)][arm][key]
                    if value is None:
                        assert target is None
                    else:
                        np.testing.assert_allclose(value, target, rtol=1e-12)
    assert comparisons(regenerated, cfg) == summary["comparisons"]
    assert (
        sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        < cfg["budget"]["artifact_cap_bytes"]
    )
    print(
        "PASS: complete schedule independently replayed;1512 rows, parent/source hashes, shared acquired inputs, five-sweep parity, saved endpoints, aggregates and gain decisions"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT)
    a = p.parse_args()
    verify(a.root)
