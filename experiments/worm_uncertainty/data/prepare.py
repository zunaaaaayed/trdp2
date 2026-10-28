"""Prepare one development sinogram in stored units; never modify source TIFFs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile


def angle_mask(start, span):
    if not np.isfinite([start, span]).all() or not 0 <= start < 180 or not 0 < span <= 180:
        raise ValueError("require 0 <= start < 180 and 0 < span <= 180 degrees")
    angles = np.arange(100, dtype=np.float64) * 1.8
    return angles, ((angles - start) % 180) < span


def noisy_copy(clean, scale, fraction, seed):
    if not np.isfinite(clean).all() or not np.isfinite([scale, fraction]).all():
        raise ValueError("inputs must be finite")
    if scale <= 0 or fraction < 0 or seed < 0:
        raise ValueError("positive scale, nonnegative fraction and seed required")
    sigma = scale * fraction
    if not np.isfinite(sigma):
        raise ValueError("noise sigma overflow")
    result = np.array(clean, dtype=np.float64, copy=True)
    result += np.random.Generator(np.random.PCG64(seed)).normal(0, sigma, result.shape)
    if not np.isfinite(result).all():
        raise ValueError("noisy values overflow")
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--z", type=int, default=255)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--span", type=float, default=120)
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="fixed development-derived signal scale in stored units",
    )
    parser.add_argument("--scale-provenance", required=True)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    angles, measured = angle_mask(args.start, args.span)
    if not 0 <= args.z < 510:
        raise ValueError("z must be in [0, 510)")
    if args.output.exists():
        raise ValueError("output exists; use a fresh destination")
    with tifffile.TiffFile(args.source) as tif:
        if tif.series[0].shape != (510, 100, 310) or len(tif.pages) != 510:
            raise ValueError("expected a 510-page (z, angle, detector) stack")
        clean = tif.pages[args.z].asarray().astype(np.float64)
    if clean.shape != (100, 310) or (clean < 0).any():
        raise ValueError("unexpected clean sinogram")
    noisy = noisy_copy(clean, args.scale, args.fraction, args.seed)
    record = dict(
        kind="worm_development_noise_preparation",
        status="prepared",
        source=str(args.source.resolve()),
        source_sha256=sha256(args.source),
        z=args.z,
        axes=["angle", "detector"],
        geometry="parallel",
        detector_spacing_um=1,
        z_spacing_um=1,
        angle_convention="100 samples over [0,180), step 1.8 degrees",
        start_deg=args.start,
        span_deg=args.span,
        measured_views=int(measured.sum()),
        withheld_views=int((~measured).sum()),
        signal_scale=args.scale,
        scale_provenance=args.scale_provenance,
        noise_fraction=args.fraction,
        sigma=args.scale * args.fraction,
        seed=args.seed,
        rng="PCG64",
        numpy_version=np.__version__,
        units="original stored projection units; physical normalization unresolved",
        clipping=False,
        specimen_scope="deformations of one worm",
        scientific_performance_evidence=False,
        script_sha256=sha256(__file__),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(
        args.output / "sinogram.npz",
        clean=clean,
        noisy=noisy,
        angles_deg=angles,
        measured_mask=measured,
    )
    record["artifact_sha256"] = sha256(args.output / "sinogram.npz")
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
