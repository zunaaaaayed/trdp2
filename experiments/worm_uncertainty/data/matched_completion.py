"""Exact inherited point sampling and shared observations for completion/refinement."""

import json
from pathlib import Path
import numpy as np
import tifffile
from experiments.worm_uncertainty.layout import resource_path
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.data.completion_data import (
    acquisition_mask,
    channels,
    validate_manifest,
)
from experiments.worm_uncertainty.studies.worm_shape_pilot import sampling_geometry

PROTOCOL = Path(__file__).resolve().parents[1] / "protocols/completion_matched_protocol_v2.json"


def read_protocol(path=PROTOCOL):
    design = json.loads(Path(path).read_text())
    manifest_path = resource_path("completion_manifest_v1.json")
    if (
        sha256(manifest_path) != design["manifest_sha256"]
        or sha256(resource_path("worm_shape_protocol_v1.json"))
        != design["sampling_protocol_sha256"]
    ):
        raise ValueError("parent protocol/manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    return design, manifest


def grid(design):
    cfg = design["sampling"]
    angles, uv, v = sampling_geometry(cfg)
    return np.rad2deg(angles), uv, v


def sparse_noise(design, name, z, epoch, role):
    identity = int(name.split("_")[1])
    seed = np.random.SeedSequence(
        [design["seeds"]["training" if role == "train" else role], identity, z, epoch]
    )
    noise = np.random.default_rng(seed).normal(0, design["noise_sd"], (100, 310))
    cfg = design["sampling"]
    return noise[np.ix_(cfg["angle_indices"], cfg["detector_indices"])]


def make_observation(clean, design, name, z, epoch, role, start, span):
    degrees, _, _ = grid(design)
    mask = acquisition_mask(start, span, degrees)
    noisy = clean + sparse_noise(design, name, z, epoch, role)
    measured = np.where(mask[:, None], noisy, 0.0)
    return measured, mask


def make_channels(measured, mask, start, span, arm):
    x = channels(measured, mask, start, span)
    if arm == "mask":
        x[2:] = 0
    elif arm != "geometry":
        raise ValueError("unknown arm")
    return x


def preserve_measured(prediction, measured, mask):
    if prediction.shape != measured.shape:
        raise ValueError("shape mismatch")
    return np.where(mask[:, None], measured, prediction.astype(np.float64))


def load_clean(data_root, design, manifest, role):
    if role not in ["train", "validation"]:
        raise ValueError("reserved outcomes unavailable in development loader")
    records = {r["name"]: r for r in manifest["files"]}
    arrays = []
    for name in design["subsets"][role]:
        record = records[name]
        if record["role"] != role:
            raise ValueError("population role mismatch")
        path = data_root / name
        if sha256(path) != record["sha256"]:
            raise ValueError("source hash mismatch: " + name)
        cfg = design["sampling"]
        with tifffile.TiffFile(path) as tif:
            if list(tif.series[0].shape) != cfg["source_shape"]:
                raise ValueError("shape mismatch")
            arrays.append(
                np.array(
                    [
                        tif.pages[z].asarray()[
                            np.ix_(cfg["angle_indices"], cfg["detector_indices"])
                        ]
                        for z in cfg["z_indices"]
                    ]
                )
                / design["normalization"]["value"]
            )
    result = np.array(arrays)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite source")
    return result


def examples(clean, design, role, masks, epoch, arm):
    xs, targets, measured, masks_out, identities = [], [], [], [], []
    for f, name in enumerate(design["subsets"][role]):
        for mi, (start, span) in enumerate(masks):
            for zi, z in enumerate(design["sampling"]["z_indices"]):
                obs, mask = make_observation(
                    clean[f, zi], design, name, z, epoch, role, start, span
                )
                xs.append(make_channels(obs, mask, start, span, arm))
                targets.append(clean[f, zi])
                measured.append(obs)
                masks_out.append(mask)
                identities.append([f, mi, zi])
    return (
        np.array(xs),
        np.array(targets),
        np.array(measured),
        np.array(masks_out),
        np.array(identities),
    )


def metrics(prediction, target, mask, identities, names, acquisitions):
    rows = []
    for f, name in enumerate(names):
        for mi, (start, span) in enumerate(acquisitions):
            selected = (identities[:, 0] == f) & (identities[:, 1] == mi)
            missing = np.broadcast_to(~mask[selected, :, None], target[selected].shape)
            diff = (prediction[selected] - target[selected])[missing]
            truth = target[selected][missing]
            energy = float(truth @ truth)
            if energy <= 1e-20:
                raise ValueError("empty target energy; no silent eligibility changes")
            rows.append(
                dict(
                    file=name,
                    start=start,
                    span=span,
                    relative_l2=float(np.linalg.norm(diff) / np.sqrt(energy)),
                    mse=float(np.mean(diff**2)),
                )
            )
    per_file = [
        dict(
            file=name,
            relative_l2=float(np.mean([r["relative_l2"] for r in rows if r["file"] == name])),
            mse=float(np.mean([r["mse"] for r in rows if r["file"] == name])),
        )
        for name in names
    ]
    return dict(
        relative_l2=float(np.mean([r["relative_l2"] for r in per_file])),
        mse=float(np.mean([r["mse"] for r in per_file])),
        per_file=per_file,
        per_mask_file=rows,
    )


def evaluation_masks(arrays, design, acquisitions):
    """Recover exact acquisition masks from saved identities and frozen geometry.

    Early v2 artifacts used 'mask' for both an arm and metadata; retain their bytes
    and recover geometry here. New artifacts use the unambiguous acquired_mask.
    """
    degrees, _, _ = grid(design)
    expected = np.array(
        [
            acquisition_mask(*acquisitions[int(row[1])], angles=degrees)
            for row in arrays["identities"]
        ]
    )
    if "acquired_mask" in arrays and not np.array_equal(expected, arrays["acquired_mask"]):
        raise ValueError("saved acquisition mask disagrees with protocol")
    return expected
