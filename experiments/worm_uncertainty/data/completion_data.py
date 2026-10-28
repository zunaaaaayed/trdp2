"""Leakage-aware supervised projection-completion inputs; no training or test scoring."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from experiments.worm_uncertainty.data.prepare import sha256


def acquisition_mask(start, span, angles=None):
    angles = np.arange(100) * 1.8 if angles is None else np.asarray(angles)
    if not 0 < span <= 180 or not np.isfinite(start):
        raise ValueError("finite start and span in (0,180] required")
    return np.mod(angles - start, 180) < span - 1e-10


def angular_extension(sino):
    """p(theta+pi,u)=p(theta,-u), with symmetric detector pixel centres."""
    return np.concatenate([sino[..., ::-1], sino, sino[..., ::-1]], axis=-2)


def interpolate_missing(measured, mask):
    """Linear angular comparator; only observed rays enter interpolation."""
    measured, mask = np.asarray(measured), np.asarray(mask, dtype=bool)
    if measured.ndim != 2 or mask.shape != (len(measured),) or mask.sum() < 2:
        raise ValueError("2D sinogram and >=2 acquired angles required")
    a, d = measured.shape
    extended = angular_extension(np.where(mask[:, None], measured, 0))
    available = np.tile(mask, 3)
    indices = np.arange(-a, 2 * a)
    result = np.column_stack(
        [np.interp(np.arange(a), indices[available], extended[available, i]) for i in range(d)]
    )
    result[mask] = measured[mask]
    return result


def channels(measured, mask, start, span):
    """Caller supplies measured data only; hidden clean values cannot affect channels."""
    measured = np.asarray(measured)
    a, d = measured.shape
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (a,) or not np.isfinite(measured[mask]).all():
        raise ValueError("valid measured data and mask required")
    theta = np.arange(a) * np.pi / a

    def expand(v):
        return np.broadcast_to(np.asarray(v)[:, None], (a, d))

    return np.stack(
        [
            np.where(mask[:, None], measured, 0),
            expand(mask),
            expand(np.sin(theta)),
            expand(np.cos(theta)),
            np.full((a, d), np.sin(np.deg2rad(start))),
            np.full((a, d), np.cos(np.deg2rad(start))),
            np.full((a, d), span / 180),
        ]
    ).astype(np.float32)


def observation(clean, scale, noise_sd, seed, start, span):
    if scale <= 0 or noise_sd < 0 or clean.shape != (100, 310) or not np.isfinite(clean).all():
        raise ValueError("positive fixed scale and finite 100x310 sinogram required")
    mask = acquisition_mask(start, span)
    # Full seeded noise makes shared rays identical across masks; no per-case normalization.
    noise = np.random.default_rng(seed).normal(0, noise_sd, clean.shape)
    measured = np.where(mask[:, None], clean / scale + noise, 0)
    return channels(measured, mask, start, span), mask


def validate_manifest(manifest):
    records = manifest["files"]
    names = [r["name"] for r in records]
    if len(names) != len(set(names)):
        raise ValueError("duplicate deformation")
    groups = {}
    for r in records:
        groups.setdefault(r["group"], set()).add(r["role"])
    if any(len(v) != 1 for v in groups.values()):
        raise ValueError("group leaks across population roles")
    if next(r["role"] for r in records if r["name"] == "worm_004_gt.tiff") != "train":
        raise ValueError("historical development scan must stay development")
    hashes = {}
    for r in records:
        hashes.setdefault(r["sha256"], set()).add(r["role"])
    if any(len(v) > 1 for v in hashes.values()):
        raise ValueError("exact duplicate leaks between roles")


def load_slice(root, manifest, name, z, role):
    record = next(r for r in manifest["files"] if r["name"] == name)
    if record["role"] != role or role == "reserved":
        raise ValueError("reserved targets unavailable to development loader")
    if z not in manifest["z_indices"]:
        raise ValueError("slice outside frozen pilot sampling")
    path = root / name
    if sha256(path) != record["sha256"]:
        raise ValueError("source hash mismatch")
    with tifffile.TiffFile(path) as tif:
        return tif.pages[z].asarray()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../worm_data"))
    parser.add_argument(
        "--manifest", type=Path, default=resource_path("completion_manifest_v1.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    clean = load_slice(args.data, manifest, "worm_004_gt.tiff", 256, "train")
    x, mask = observation(clean, manifest["normalization"]["value"], 0.05, 2026102802, 0, 120)
    simple = interpolate_missing(x[0], mask)
    # Development smoke only. No reserved TIFF contents opened.
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "development_example.npz",
        inputs=x,
        mask=mask,
        interpolation=simple,
        target=clean / manifest["normalization"]["value"],
    )
    summary = dict(
        status="data smoke passed; no neural training",
        roles={
            r: sum(f["role"] == r for f in manifest["files"])
            for r in ["train", "validation", "reserved"]
        },
        input_shape=list(x.shape),
        acquired_angles=int(mask.sum()),
        development_interpolation_withheld_relative_l2=float(
            np.linalg.norm((simple - clean / manifest["normalization"]["value"])[~mask])
            / np.linalg.norm((clean / manifest["normalization"]["value"])[~mask])
        ),
        manifest_sha256=sha256(args.manifest),
        code_sha256=sha256(Path(__file__)),
        source_sha256=next(
            r["sha256"] for r in manifest["files"] if r["name"] == "worm_004_gt.tiff"
        ),
        reserved_pixels_read=False,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
