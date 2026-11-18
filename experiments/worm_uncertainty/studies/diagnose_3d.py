"""Controlled known-target 3D Gaussian representation/optimization/UQ diagnosis."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import cho_factor, cho_solve
from experiments.worm_uncertainty.data.prepare import sha256


def ray_basis(centres, scales, angles, uv):
    c = np.cos(angles)[:, None, None]
    s = np.sin(angles)[:, None, None]
    du = uv[None, :, None, 0] - (c * centres[None, None, :, 0] - s * centres[None, None, :, 1])
    dv = uv[None, :, None, 1] - centres[None, None, :, 2]
    variance = c * c * scales[None, None, :, 0] ** 2 + s * s * scales[None, None, :, 1] ** 2
    amplitude = np.sqrt(
        2 * np.pi * scales[None, None, :, 0] ** 2 * scales[None, None, :, 1] ** 2 / variance
    )
    basis = amplitude * np.exp(-0.5 * (du**2 / variance + dv**2 / scales[None, None, :, 2] ** 2))
    derivatives = (
        np.stack(
            [
                du / variance * c,
                -du / variance * s,
                np.broadcast_to(dv / scales[None, None, :, 2] ** 2, basis.shape),
            ],
            axis=-1,
        )
        * basis[:, :, :, None]
    )
    return basis.reshape(-1, len(centres)), derivatives.reshape(-1, len(centres), 3)


def voxel_features(centres, scales, points):
    delta = points[:, None, :] - centres[None, :, :]
    basis = np.exp(-0.5 * np.sum((delta / scales[None, :, :]) ** 2, axis=-1))
    derivative = basis[:, :, None] * delta / scales[None, :, :] ** 2
    return basis, derivative


def features(parameters, scales, angles, uv):
    k = len(scales)
    centres = parameters[: 3 * k].reshape(k, 3)
    density = parameters[3 * k :]
    basis, derivative = ray_basis(centres, scales, angles, uv)
    jac = np.concatenate(
        [(derivative * density[None, :, None]).reshape(len(basis), 3 * k), basis], axis=1
    )
    return basis @ density, jac, basis


def invert_precision(matrix):
    return cho_solve(cho_factor(matrix, lower=True), np.eye(len(matrix)))


def optimize(y, angles, uv, scales, initial, sigma, cfg, budget):
    k = len(scales)
    prior_mean = np.r_[initial[: 3 * k], np.zeros(k)]
    prior_sd = np.r_[np.full(3 * k, cfg["position_prior_sd"]), np.full(k, cfg["density_prior_sd"])]

    def residual(p):
        return np.r_[(features(p, scales, angles, uv)[0] - y) / sigma, (p - prior_mean) / prior_sd]

    def jacobian(p):
        return np.vstack([features(p, scales, angles, uv)[1] / sigma, np.diag(1 / prior_sd)])

    lower = np.r_[np.full(3 * k, cfg["position_bounds"][0]), np.full(k, cfg["density_bounds"][0])]
    upper = np.r_[np.full(3 * k, cfg["position_bounds"][1]), np.full(k, cfg["density_bounds"][1])]
    fit = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=(lower, upper),
        max_nfev=budget,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    return fit, prior_sd


def diagnostic(mean, sd, truth, selected):
    error = mean[selected] - truth[selected]
    sd = sd[selected]
    return dict(
        rmse=float(np.sqrt(np.mean(error**2))),
        coverage95=float(np.mean(abs(error) <= 1.95996398454 * sd)),
        mean_width95=float(np.mean(2 * 1.95996398454 * sd)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, default=resource_path("phantom_3d_protocol_v1.json"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    cfg = json.loads(args.protocol.read_text())
    centres = np.array(cfg["centres"])
    scales = np.array(cfg["scales"])
    density = np.array(cfg["densities"])
    k = len(density)
    angles = np.deg2rad(np.arange(cfg["angles_deg"]["count"]) * cfg["angles_deg"]["step"])
    height, width = cfg["detector_shape"]
    u = (np.arange(width) + 0.5) * 2 / width - 1
    v = (np.arange(height) + 0.5) * 2 / height - 1
    uu, vv = np.meshgrid(u, v)
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    coords = [(np.arange(n) + 0.5) * 2 / n - 1 for n in cfg["voxel_shape"]]
    points = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1).reshape(-1, 3)
    basis, _ = ray_basis(centres, scales, angles, uv)
    clean = basis @ density
    vb, _ = voxel_features(centres, scales, points)
    truth = vb @ density
    foreground = truth > cfg["foreground_fraction_of_peak"] * truth.max()
    ray_foreground = clean > cfg["foreground_fraction_of_peak"] * clean.max()
    sigma = cfg["noise_fraction_full_projection_rms"] * np.sqrt(np.mean(clean**2))
    noise = np.random.default_rng(cfg["noise_seed"]).normal(
        0, sigma, (cfg["noise_repeats"], len(clean))
    )
    initial = np.r_[
        (
            centres
            + np.random.default_rng(cfg["initialization_seed"]).normal(
                0, cfg["initialization_position_sd"], centres.shape
            )
        ).ravel(),
        density * cfg["initialization_density_fraction"],
    ]
    # Representation: exact fixed-basis convex solves, not nonconvex fits.
    extra = cfg["extra_component"]
    all_centres = np.vstack([centres, extra["centre"]])
    all_scales = np.vstack([scales, extra["scale"]])
    all_density = np.r_[density, extra["density"]]
    extra_basis, _ = ray_basis(all_centres, all_scales, angles, uv)
    extra_vb, _ = voxel_features(all_centres, all_scales, points)
    extra_clean = extra_basis @ all_density
    extra_truth = extra_vb @ all_density
    extra_fg = extra_truth > cfg["foreground_fraction_of_peak"] * extra_truth.max()
    representation = []
    for count in [k, k + 1]:
        weights = np.linalg.lstsq(extra_basis[:, :count], extra_clean, rcond=None)[0]
        reconstruction = extra_vb[:, :count] @ weights
        representation.append(
            dict(
                components=count,
                projection_relative_l2=float(
                    np.linalg.norm(extra_basis[:, :count] @ weights - extra_clean)
                    / np.linalg.norm(extra_clean)
                ),
                foreground_voxel_rmse=float(
                    np.sqrt(np.mean((reconstruction[extra_fg] - extra_truth[extra_fg]) ** 2))
                ),
            )
        )
    arrays = dict(
        truth=truth.reshape(cfg["voxel_shape"]),
        foreground=foreground.reshape(cfg["voxel_shape"]),
        extra_truth=extra_truth.reshape(cfg["voxel_shape"]),
    )
    optimization = []
    uncertainty = []
    for span in cfg["spans_deg"]:
        angle_mask = np.rad2deg(angles) < span - 1e-8
        mask = np.repeat(angle_mask, len(uv))
        local_angles = angles[angle_mask]
        for budget in cfg["optimization_budgets_nfev"]:
            fit, prior_sd = optimize(
                clean[mask], local_angles, uv, scales, initial, sigma, cfg, budget
            )
            pred, _, _ = features(fit.x, scales, angles, uv)
            local_vb, _ = voxel_features(fit.x[: 3 * k].reshape(k, 3), scales, points)
            reconstruction = local_vb @ fit.x[3 * k :]
            optimization.append(
                dict(
                    span=span,
                    budget_nfev=budget,
                    used_nfev=fit.nfev,
                    converged=bool(fit.success),
                    optimality=float(fit.optimality),
                    objective=float(2 * fit.cost),
                    voxel_rmse=float(
                        np.sqrt(np.mean((reconstruction[foreground] - truth[foreground]) ** 2))
                    ),
                    measured_relative_l2=float(
                        np.linalg.norm((pred - clean)[mask]) / np.linalg.norm(clean[mask])
                    ),
                    withheld_relative_l2=float(
                        np.linalg.norm((pred - clean)[~mask]) / np.linalg.norm(clean[~mask])
                    )
                    if (~mask).any()
                    else None,
                )
            )
        summaries = {name: [] for name in ["oracle_density", "learned_density", "learned_joint"]}
        converged = 0
        active = 0
        norm_differences = []
        for repeat in range(cfg["noise_repeats"]):
            y = clean + noise[repeat]
            # Oracle fixed geometry: exact Gaussian density posterior.
            b = basis[mask]
            cd = invert_precision(b.T @ b / sigma**2 + np.eye(k) / cfg["density_prior_sd"] ** 2)
            md = cd @ (b.T @ y[mask] / sigma**2)
            oracle_mean = vb @ md
            oracle_sd = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", vb, cd, vb), 0))
            summaries["oracle_density"].append(
                diagnostic(oracle_mean, oracle_sd, truth, foreground)
            )
            fit, prior_sd = optimize(
                y[mask],
                local_angles,
                uv,
                scales,
                initial,
                sigma,
                cfg,
                max(cfg["optimization_budgets_nfev"]),
            )
            converged += int(fit.success)
            active += int(np.any(fit.active_mask))
            centres_fit = fit.x[: 3 * k].reshape(k, 3)
            b, _ = ray_basis(centres_fit, scales, local_angles, uv)
            conditional_cov = invert_precision(
                b.T @ b / sigma**2 + np.eye(k) / cfg["density_prior_sd"] ** 2
            )
            density_mean = conditional_cov @ (b.T @ y[mask] / sigma**2)
            norm_differences.append(float(np.linalg.norm(density_mean - fit.x[3 * k :])))
            params = np.r_[centres_fit.ravel(), density_mean]
            _, jac, _ = features(params, scales, local_angles, uv)
            joint_cov = invert_precision(jac.T @ jac / sigma**2 + np.diag(1 / prior_sd**2))
            voxel_b, voxel_d = voxel_features(centres_fit, scales, points)
            voxel_j = np.concatenate(
                [(voxel_d * density_mean[None, :, None]).reshape(len(points), 3 * k), voxel_b],
                axis=1,
            )
            mean = voxel_b @ density_mean
            sd = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", voxel_b, conditional_cov, voxel_b), 0))
            joint_sd = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", voxel_j, joint_cov, voxel_j), 0))
            summaries["learned_density"].append(diagnostic(mean, sd, truth, foreground))
            summaries["learned_joint"].append(diagnostic(mean, joint_sd, truth, foreground))
            if np.min(joint_sd**2 - sd**2) < -1e-10:
                raise ValueError("joint covariance unexpectedly smaller")
            if repeat == 0:
                arrays[f"mean_{span}"] = mean.reshape(cfg["voxel_shape"])
                arrays[f"conditional_sd_{span}"] = sd.reshape(cfg["voxel_shape"])
                arrays[f"joint_sd_{span}"] = joint_sd.reshape(cfg["voxel_shape"])
        row = dict(
            span=span,
            converged_noise_fits=converged,
            bound_active_noise_fits=active,
            max_conditional_map_density_difference=max(norm_differences),
            methods={},
        )
        for method, values in summaries.items():
            row["methods"][method] = {
                metric: float(np.mean([v[metric] for v in values])) for metric in values[0]
            }
            row["methods"][method]["coverage_min_across_draws"] = min(
                v["coverage95"] for v in values
            )
            row["methods"][method]["coverage_max_across_draws"] = max(
                v["coverage95"] for v in values
            )
        uncertainty.append(row)
        print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "arrays.npz", **arrays)
    report = dict(
        kind=cfg["kind"],
        protocol=cfg,
        protocol_sha256=sha256(args.protocol),
        sigma=float(sigma),
        representation=representation,
        optimization=optimization,
        uncertainty=uncertainty,
        uncertainty_metric="mean of per-draw foreground voxel RMSE/coverage/width; one fixed truth, correlated voxels",
        joint_method="local Gauss-Newton/Laplace covariance with positions and densities; fixed scales and rotations",
        code_sha256=sha256(__file__),
        arrays_sha256=sha256(args.output / "arrays.npz"),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("representation", representation)
    print("optimization", optimization)


if __name__ == "__main__":
    main()
