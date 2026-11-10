"""Permitted contiguous slabs and independent noisy pairs for the ICECREAM pilot."""

import hashlib
import json
from pathlib import Path
import numpy as np
import tifffile
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.data.completion_data import acquisition_mask, validate_manifest

BASE = Path(__file__).resolve().parents[1]
MANIFEST = BASE / "protocols/completion_manifest_v1.json"


def allowed_record(name):
    manifest = json.loads(MANIFEST.read_text())
    validate_manifest(manifest)
    record = next((r for r in manifest["files"] if r["name"] == name), None)
    if record is None or record["role"] == "reserved":
        raise ValueError("reserved or unknown deformation forbidden before file access")
    return record, manifest


def load_slab(root, name, z_start=224, z_stop=288):
    record, manifest = allowed_record(name)
    if (z_start, z_stop) != (224, 288):
        raise ValueError("slab must match frozen contiguous 64-plane range")
    path = Path(root) / name
    if sha256(path) != record["sha256"]:
        raise ValueError("TIFF checksum mismatch")
    with tifffile.TiffFile(path) as tif:
        if tif.series[0].shape != (510, 100, 310):
            raise ValueError("unexpected source axes")
        clean = np.stack([tif.pages[z].asarray() for z in range(z_start, z_stop)])
    return clean.astype(np.float64) / manifest["normalization"]["value"], record


def noisy_pair(clean, mask, sigma, seeds):
    """Generate two full-grid independent noises, then discard every withheld value."""
    if len(set(seeds)) != 2 or sigma <= 0:
        raise ValueError("distinct pair seeds and positive noise required")
    if np.asarray(mask).shape != (clean.shape[1],) or not np.isfinite(clean[:, mask]).all():
        raise ValueError("invalid acquired data")
    pair = np.stack(
        [
            np.where(
                mask[None, :, None],
                clean + np.random.default_rng(s).normal(0, sigma, clean.shape),
                0,
            )
            for s in seeds
        ]
    )
    return pair


def mask_record(start, span):
    angles = np.arange(100) * 1.8
    mask = acquisition_mask(start, span, angles)
    return dict(
        start=float(start),
        span=float(span),
        count=int(mask.sum()),
        indices=np.flatnonzero(mask).tolist(),
        mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
    )


def phantom(seed, size=64, rotation_degrees=0):
    """Mixed smooth, sharp, hollow and elongated objects; no Gaussian-only truth."""
    rng = np.random.default_rng(seed)
    q = (np.arange(size) - (size - 1) / 2) / (size / 2)
    z, y, x = np.meshgrid(q, q, q, indexing="ij")
    a = np.deg2rad(rotation_degrees)
    xx, yy = np.cos(a) * x - np.sin(a) * y, np.sin(a) * x + np.cos(a) * y
    out = np.zeros_like(x)
    for _ in range(3):
        centre = rng.uniform(-0.4, 0.4, 3)
        widths = rng.uniform(0.08, 0.19, 3)
        out += rng.uniform(0.006, 0.014) * np.exp(
            -0.5
            * (
                ((xx - centre[0]) / widths[0]) ** 2
                + ((yy - centre[1]) / widths[1]) ** 2
                + ((z - centre[2]) / widths[2]) ** 2
            )
        )
    out += 0.023 * (((xx + 0.17) / 0.12) ** 2 + ((yy - 0.08) / 0.42) ** 2 + (z / 0.55) ** 2 < 1)
    ball = (xx - 0.25) ** 2 + (yy + 0.26) ** 2 + (z + 0.14) ** 2
    out += 0.018 * ((ball < 0.22**2) & (ball > 0.15**2))
    out += 0.016 * (
        (np.abs(xx + 0.32) < 0.12) & (np.abs(yy + 0.26) < 0.16) & (np.abs(z - 0.28) < 0.12)
    )
    return np.where((x * x + y * y < 0.85**2) & (np.abs(z) < 0.85), out, 0)
