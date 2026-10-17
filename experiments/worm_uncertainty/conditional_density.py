"""Plug-in Gaussian density posterior with fixed learned geometry; no positivity."""

import argparse
import json
from pathlib import Path
import numpy as np
from prepare import sha256
from limited_gaussian import projection_errors


def covariance_from_parameters(scales, quaternions):
    q = np.asarray(quaternions, dtype=float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rotation = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    return (rotation * np.asarray(scales)[:, None, :] ** 2) @ rotation.transpose(0, 2, 1)


def projection_basis(xyz, cov, angle, uv):
    """Torch arrays: unit-peak 3D Gaussians integrated along infinite rays."""
    import torch

    c, s = np.cos(angle), np.sin(angle)
    plane = torch.tensor([[c, -s, 0], [0, 0, 1]], dtype=xyz.dtype, device=xyz.device)
    projected = plane @ cov @ plane.T
    centres = xyz @ plane.T
    det = projected[:, 0, 0] * projected[:, 1, 1] - projected[:, 0, 1] ** 2
    if bool((det <= 0).any()):
        raise ValueError("nonpositive projected covariance")
    du = uv[:, 0, None] - centres[None, :, 0]
    dv = uv[:, 1, None] - centres[None, :, 1]
    exponent = (
        projected[None, :, 1, 1] * du**2
        - 2 * projected[None, :, 0, 1] * du * dv
        + projected[None, :, 0, 0] * dv**2
    ) / det
    factor = torch.sqrt(2 * np.pi * torch.linalg.det(cov) / det)
    return torch.exp(-0.5 * exponent) * factor


def voxel_basis(xyz, cov, points):
    import torch

    delta = points[:, None, :] - xyz[None, :, :]
    inv = torch.linalg.inv(cov)
    return torch.exp(-0.5 * torch.einsum("pki,kij,pkj->pk", delta, inv, delta))


def solve_posterior(gram, rhs, sigma, tau):
    import torch

    precision = (
        gram / sigma**2 + torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device) / tau**2
    )
    factor = torch.linalg.cholesky(precision)
    mean = torch.cholesky_solve((rhs / sigma**2)[:, None], factor)[:, 0]
    covariance = torch.cholesky_inverse(factor)
    return mean, covariance


def coverage_summary(mean, sd, truth, selected):
    if not selected.any():
        return None
    error = (mean - truth)[selected]
    width = (2 * 1.95996398454 * sd)[selected]
    return dict(
        coverage95=float(np.mean(abs(error) <= width / 2)),
        mean_width95=float(width.mean()),
        rmse=float(np.sqrt(np.mean(error**2))),
        mean_sd=float(sd[selected].mean()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    if not args.execute:
        print("Dry run: fixed-geometry density posterior for sigma=0.05 cases")
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cfg = json.loads((args.input / "protocol.json").read_text())
    parent = json.loads((args.input / "report.json").read_text())
    if sha256(args.input / "inputs.npz") != parent["inputs_sha256"]:
        raise ValueError("input checksum mismatch")
    inputs = np.load(args.input / "inputs.npz")
    clean = inputs["clean"]
    noise = inputs["unit_noise"]
    angles = inputs["angles_deg"]
    sigma = cfg["posterior_noise_sigma"]
    tau = cfg["posterior_prior_sd"]
    noisy = clean + sigma * noise
    depth, width = clean.shape[1:]
    spacing = 2 / width
    u = (np.arange(width) + 0.5 - width / 2) * spacing
    v = (np.arange(depth) + 0.5 - depth / 2) * spacing
    uu, vv = np.meshgrid(u, v)
    uv = torch.tensor(
        np.stack([uu.ravel(), vv.ravel()], axis=1), device="cuda", dtype=torch.float64
    )
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for record in parent["results"]:
        if record["sigma"] != sigma:
            continue
        key = record["case"]
        case = args.input / key
        destination = args.output / key
        destination.mkdir()
        for name, h in record["artifacts"].items():
            if sha256(case / name) != h:
                raise ValueError("baseline artifact mismatch")
        fit = np.load(case / "gaussians.npz")
        baseline = np.load(case / "arrays.npz")
        mask = baseline["mask"]
        xyz = torch.tensor(fit["xyz"], device="cuda", dtype=torch.float64)
        cov = torch.tensor(
            covariance_from_parameters(fit["scales"], fit["rotation"]),
            device="cuda",
            dtype=torch.float64,
        )
        weights = torch.tensor(fit["density"].ravel(), device="cuda", dtype=torch.float64)
        k = len(weights)
        gram = torch.zeros((k, k), device="cuda", dtype=torch.float64)
        rhs = torch.zeros(k, device="cuda", dtype=torch.float64)
        analytic = np.empty_like(clean)
        for i, angle in enumerate(angles):
            basis = projection_basis(xyz, cov, np.deg2rad(angle), uv)
            analytic[i] = (basis @ weights).cpu().numpy().reshape(depth, width)
            if mask[i]:
                gram += basis.T @ basis
                rhs += basis.T @ torch.tensor(noisy[i].ravel(), device="cuda", dtype=torch.float64)
        discrepancy = float(
            np.linalg.norm(analytic - baseline["prediction"])
            / np.linalg.norm(baseline["prediction"])
        )
        if discrepancy > 0.02:
            raise ValueError(
                "native versus analytic projector discrepancy exceeds 2%: " + str(discrepancy)
            )
        mean, posterior_cov = solve_posterior(gram, rhs, sigma, tau)
        prediction = np.empty_like(clean)
        latent_sd = np.empty_like(clean)
        for i, angle in enumerate(angles):
            basis = projection_basis(xyz, cov, np.deg2rad(angle), uv)
            prediction[i] = (basis @ mean).cpu().numpy().reshape(depth, width)
            variance = torch.sum((basis @ posterior_cov) * basis, dim=1)
            if float(variance.min()) < -1e-9:
                raise ValueError("negative variance")
            latent_sd[i] = torch.sqrt(variance.clamp_min(0)).cpu().numpy().reshape(depth, width)
        # Exact conditional variance for the central 100x100 voxel crop, not MC.
        coord = (np.arange(105, 205) + 0.5 - width / 2) * spacing
        xx, yy = np.meshgrid(coord, coord)
        points = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, v[depth // 2])], axis=1)
        voxel_mean = []
        voxel_sd = []
        for start in range(0, len(points), 512):
            basis = voxel_basis(
                xyz,
                cov,
                torch.tensor(points[start : start + 512], device="cuda", dtype=torch.float64),
            )
            voxel_mean.append((basis @ mean).cpu().numpy())
            voxel_sd.append(
                torch.sqrt(torch.sum((basis @ posterior_cov) * basis, dim=1).clamp_min(0))
                .cpu()
                .numpy()
            )
        foreground = clean > 0.05
        measures = {}
        for label, angle_mask in [("measured", mask), ("withheld", ~mask)]:
            selected = np.broadcast_to(angle_mask[:, None, None], clean.shape) & foreground
            measures[label + "_latent_clean"] = coverage_summary(
                prediction, latent_sd, clean, selected
            )
            measures[label + "_predictive_noisy"] = coverage_summary(
                prediction, np.sqrt(latent_sd**2 + sigma**2), noisy, selected
            )
        np.savez_compressed(
            destination / "arrays.npz",
            prediction=prediction,
            latent_sd=latent_sd,
            analytic_baseline=analytic,
            density_mean=mean.cpu().numpy(),
            density_covariance=posterior_cov.cpu().numpy(),
            voxel_mean=np.concatenate(voxel_mean).reshape(100, 100),
            voxel_sd=np.concatenate(voxel_sd).reshape(100, 100),
        )
        summary = dict(
            case=key,
            span=record["span"],
            sigma=sigma,
            prior_sd=tau,
            native_analytic_relative_l2=discrepancy,
            negative_density_mean_fraction=float((mean < 0).double().mean()),
            **projection_errors(prediction, clean, mask),
            coverage=measures,
            arrays_sha256=sha256(destination / "arrays.npz"),
        )
        (destination / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    report = dict(
        kind="conditional_density_posterior_development",
        baseline_report_sha256=sha256(args.input / "report.json"),
        results=summaries,
        foreground_threshold_normalized=0.05,
        posterior="exact linear Gaussian given fitted geometry, zero-mean prior; all amplitudes free",
        limitations=cfg["limitations"]
        + [
            "credible intervals may be overconfident due to plug-in geometry and model mismatch",
            "latent coverage uses clean projections; predictive coverage adds known measurement noise",
            "measured predictive coverage is in-sample; only withheld angles test prediction",
            "Gaussian density posterior allows negative amplitudes; no clipping applied",
        ],
        code_sha256=sha256(__file__),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
