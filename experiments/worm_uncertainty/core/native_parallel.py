"""Standalone parallel-beam adapter and analytic CUDA correctness check."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from experiments.worm_uncertainty.data.prepare import sha256

REPO = Path(__file__).resolve().parents[3]
PIPE = SimpleNamespace(compute_cov3D_python=False, debug=False)


def camera_basis(angle):
    c, s = np.cos(angle), np.sin(angle)
    # Detector u = x*cos(theta)-y*sin(theta); detector v = z.
    return np.array([[c, 0, -s], [-s, 0, -c], [0, 1, 0]], dtype=np.float32)


def camera(angle, size):
    import torch

    sys.path.insert(0, str(REPO))
    from r2_gaussian.dataset.cameras import Camera

    return Camera(
        0,
        {},
        camera_basis(angle),
        np.array([0, 0, 5], dtype=np.float32),
        angle,
        0,
        0.0,
        0.0,
        torch.zeros((1, size, size), device="cuda"),
        "parallel",
        0,
    )


def model_from_arrays(xyz, scales, densities):
    import torch

    sys.path.insert(0, str(REPO))
    from r2_gaussian.gaussian import GaussianModel

    model = GaussianModel()
    model._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device="cuda"))
    model._scaling = torch.nn.Parameter(
        torch.log(torch.tensor(scales, dtype=torch.float32, device="cuda"))
    )
    rotation = np.tile([1.0, 0.0, 0.0, 0.0], (len(xyz), 1))
    model._rotation = torch.nn.Parameter(torch.tensor(rotation, dtype=torch.float32, device="cuda"))
    model._density = torch.nn.Parameter(
        model.density_inverse_activation(
            torch.tensor(densities, dtype=torch.float32, device="cuda").reshape(-1, 1)
        )
    )
    return model


def analytic_projection(xyz, scales, densities, angle, size):
    coord = (np.arange(size) + 0.5) * 2 / size - 1
    u, v = np.meshgrid(coord, coord)
    c, s = np.cos(angle), np.sin(angle)
    result = np.zeros((size, size))
    for centre, scale, density in zip(xyz, scales, densities):
        projected_var = c * c * scale[0] ** 2 + s * s * scale[1] ** 2
        # Integrating a diagonal 3D covariance along the beam.
        conditional_var = scale[0] ** 2 * scale[1] ** 2 / projected_var
        centre_u = c * centre[0] - s * centre[1]
        result += (
            density
            * np.sqrt(2 * np.pi * conditional_var)
            * np.exp(
                -0.5 * ((u - centre_u) ** 2 / projected_var + (v - centre[2]) ** 2 / scale[2] ** 2)
            )
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    import torch

    sys.path.insert(0, str(REPO))
    from r2_gaussian.gaussian import render, query

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    xyz = np.array([[0.23, -0.17, 0.12], [-0.31, 0.21, -0.19], [0.05, 0.11, 0.34]])
    scales = np.array([[0.12, 0.18, 0.14], [0.16, 0.09, 0.11], [0.09, 0.13, 0.08]])
    densities = np.array([0.8, 0.5, 0.65])
    model = model_from_arrays(xyz, scales, densities)
    rows = []
    arrays = {}
    for degrees in [0, 37, 90, 145]:
        angle = np.deg2rad(degrees)
        value = render(camera(angle, 64), model, PIPE)["render"][0]
        expected = analytic_projection(xyz, scales, densities, angle, 64)
        actual = value.detach().cpu().numpy()
        error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        if error > 0.02:
            raise ValueError("analytic projection mismatch " + str(error))
        rows.append(dict(angle_deg=degrees, relative_l2=error))
        arrays[f"actual_{degrees}"] = actual
        arrays[f"analytic_{degrees}"] = expected
    volume = query(model, [0, 0, 0], [32, 32, 32], [2, 2, 2], PIPE)["vol"]
    coordinates = (np.arange(32) + 0.5) * 2 / 32 - 1
    grid = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), axis=-1)
    expected_volume = sum(
        d * np.exp(-0.5 * np.sum(((grid - m) / s) ** 2, axis=-1))
        for m, s, d in zip(xyz, scales, densities)
    )
    voxel_error = float(
        np.linalg.norm(volume.detach().cpu().numpy() - expected_volume)
        / np.linalg.norm(expected_volume)
    )
    if voxel_error > 0.02:
        raise ValueError("voxel query mismatch")
    value.sum().backward()
    for name in ["_xyz", "_scaling", "_rotation", "_density"]:
        grad = getattr(model, name).grad
        if grad is None or not torch.isfinite(grad).all():
            raise ValueError("invalid gradient " + name)
    autodiff = float(model._density.grad[0, 0])
    eps = 0.001
    with torch.no_grad():
        original = model._density[0, 0].clone()
        model._density[0, 0] = original + eps
        upper = float(render(camera(np.deg2rad(145), 64), model, PIPE)["render"].sum())
        model._density[0, 0] = original - eps
        lower = float(render(camera(np.deg2rad(145), 64), model, PIPE)["render"].sum())
        model._density[0, 0] = original
    numerical = (upper - lower) / (2 * eps)
    gradient_error = abs(numerical - autodiff) / max(abs(numerical), 1e-8)
    if gradient_error > 0.02:
        raise ValueError("density gradient mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    arrays["volume"] = volume.detach().cpu().numpy()
    arrays["analytic_volume"] = expected_volume
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    report = dict(
        kind="native_parallel_gaussian_correctness",
        projection_checks=rows,
        voxel_relative_l2=voxel_error,
        density_gradient_relative_error=gradient_error,
        finite_gradients=True,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        code_sha256=sha256(__file__),
        arrays_sha256=sha256(args.output / "arrays.npz"),
        convention="pixel centres -1+(2*i+1)/N; detector u=x*cos(theta)-y*sin(theta), v=z",
        limitation="analytic axis-aligned Gaussian phantom; does not establish worm simulator calibration",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
