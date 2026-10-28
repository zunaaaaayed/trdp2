"""Factorial test of Euler bounds and shared versus start-centred priors."""

from experiments.worm_uncertainty.layout import resource_path

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from experiments.worm_uncertainty.core.shape_gaussian import projection, voxel, uncertainty
from experiments.worm_uncertainty.studies.stress_3d import acquisition_mask
from experiments.worm_uncertainty.studies.diagnose_3d import diagnostic
from experiments.worm_uncertainty.data.prepare import sha256


def initial_parameters(centres, scales, density, rotations, seed, cfg):
    rng = np.random.default_rng(seed)
    return np.r_[
        (centres + rng.normal(0, cfg["initialization_position_sd"], centres.shape)).ravel(),
        (np.log(scales) + rng.normal(0, cfg["initialization_log_width_sd"], scales.shape)).ravel(),
        (rotations + rng.normal(0, cfg["initialization_rotation_sd"], scales.shape)).ravel(),
        density * cfg["initialization_density_fraction"],
    ]


def fit_control(y, angles, uv, initial, prior_reference, sigma, cfg, bounded):
    k = len(initial) // 10
    prior_mean = prior_reference.copy()
    prior_mean[9 * k :] = 0
    prior_sd = np.r_[
        np.full(3 * k, cfg["position_prior_sd"]),
        np.full(3 * k, cfg["log_width_prior_sd"]),
        np.full(3 * k, cfg["rotation_prior_sd"]),
        np.full(k, cfg["density_prior_sd"]),
    ]
    cache = {}

    def values(p):
        if "p" not in cache or not np.array_equal(cache["p"], p):
            cache["p"] = p.copy()
            cache["values"] = projection(p, angles, uv)
        return cache["values"]

    def residual(p):
        return np.r_[(values(p)[0] - y) / sigma, (p - prior_mean) / prior_sd]

    def jacobian(p):
        return np.vstack([values(p)[1] / sigma, np.diag(1 / prior_sd)])

    angle_bound = cfg["rotation_bound"] if bounded else np.inf
    lower = np.r_[
        np.full(3 * k, cfg["position_bounds"][0]),
        np.full(3 * k, np.log(cfg["width_bounds"][0])),
        np.full(3 * k, -angle_bound),
        np.full(k, cfg["density_bounds"][0]),
    ]
    upper = np.r_[
        np.full(3 * k, cfg["position_bounds"][1]),
        np.full(3 * k, np.log(cfg["width_bounds"][1])),
        np.full(3 * k, angle_bound),
        np.full(k, cfg["density_bounds"][1]),
    ]
    fit = least_squares(
        residual,
        initial,
        jac=jacobian,
        bounds=(lower, upper),
        max_nfev=cfg["budget_nfev"],
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    return fit, prior_sd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=resource_path("rotation_sensitivity_protocol_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parent", type=Path, default=Path("output/worm_uncertainty/shape_controls_v1")
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output exists")
    cfg = json.loads(args.protocol.read_text())
    if sha256(args.parent / "report.json") != cfg["parent_report_sha256"]:
        raise ValueError("parent changed")
    degrees = np.arange(cfg["angles_deg"]["count"]) * cfg["angles_deg"]["step"]
    angles = np.deg2rad(degrees)
    h, w = cfg["detector_shape"]
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * 2 / w - 1, (np.arange(h) + 0.5) * 2 / h - 1)
    uv = np.c_[uu.ravel(), vv.ravel()]
    coords = [(np.arange(n) + 0.5) * 2 / n - 1 for n in cfg["voxel_shape"]]
    points = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1).reshape(-1, 3)
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    rows = []
    draws = []
    for case in cfg["cases"]:
        phantom = next(p for p in cfg["phantoms"] if p["name"] == case["phantom"])
        extra = phantom["extra"]
        centres = np.vstack([phantom["centres"], extra["centre"]])
        scales = np.vstack([phantom["scales"], extra["scale"]])
        density = np.r_[phantom["densities"], extra["density"]]
        rotations = np.array(cfg["truth_rotations"])
        k = len(density)
        truth_parameters = np.r_[
            centres.ravel(), np.log(scales).ravel(), rotations.ravel(), density
        ]
        clean = projection(truth_parameters, angles, uv)[0]
        truth = voxel(truth_parameters, points)[0]
        sigma = cfg["noise_fraction_full_projection_rms"] * np.sqrt(np.mean(clean**2))
        noise = (
            np.random.default_rng(case["noise_seed"]).normal(
                size=(cfg["noise_repeats"], len(clean))
            )
            * sigma
        )
        mask = acquisition_mask(degrees, case["start"], case["span"])
        measured = np.repeat(mask, len(uv))
        fg = truth > cfg["foreground_fraction_of_peak"] * truth.max()
        withheld = ~measured & (clean > cfg["foreground_fraction_of_peak"] * clean.max())
        shared = initial_parameters(
            centres, scales, density, rotations, cfg["shared_prior_seed"], cfg
        )
        for variant in cfg["variants"]:
            records = []
            paired_differences = []
            selected = []
            for repeat in range(cfg["noise_repeats"]):
                y = clean + noise[repeat]
                pair = []
                means = []
                for seed in cfg["initialization_seeds"]:
                    initial = initial_parameters(centres, scales, density, rotations, seed, cfg)
                    reference = shared if variant["shared_prior"] else initial
                    fit, prior_sd = fit_control(
                        y[measured],
                        angles[mask],
                        uv,
                        initial,
                        reference,
                        sigma,
                        cfg,
                        variant["bounded"],
                    )
                    mean, jac = projection(fit.x, angles, uv)
                    vmean, vjac = voxel(fit.x, points)
                    ray_sd, v_sd = uncertainty(jac, vjac, measured, sigma, prior_sd, k)["rotations"]
                    record = dict(
                        case=case["name"],
                        variant=variant["name"],
                        noise_repeat=repeat,
                        initialization_seed=seed,
                        objective=float(2 * fit.cost),
                        converged=bool(fit.success),
                        nfev=fit.nfev,
                        optimality=float(fit.optimality),
                        active_indices=np.flatnonzero(fit.active_mask).tolist(),
                        parameters=fit.x.tolist(),
                        metrics=dict(
                            voxel=diagnostic(vmean, v_sd, truth, fg),
                            withheld_latent=diagnostic(mean, ray_sd, clean, withheld),
                            withheld_predictive=diagnostic(
                                mean, np.sqrt(ray_sd**2 + sigma**2), y, withheld
                            ),
                        ),
                    )
                    records.append(record)
                    draws.append(record)
                    pair.append(record)
                    means.append(vmean)
                paired_differences.append(
                    float(np.sqrt(np.mean((means[0][fg] - means[1][fg]) ** 2)))
                )
                if variant["shared_prior"]:
                    selected.append(min(pair, key=lambda r: r["objective"]))

            def aggregate(values):
                return {
                    domain: {
                        metric: float(np.mean([r["metrics"][domain][metric] for r in values]))
                        for metric in values[0]["metrics"][domain]
                    }
                    for domain in values[0]["metrics"]
                }

            row = dict(
                case=case["name"],
                variant=variant["name"],
                fits=len(records),
                converged=sum(r["converged"] for r in records),
                active_bounds=sum(bool(r["active_indices"]) for r in records),
                max_nfev=max(r["nfev"] for r in records),
                paired_mean_difference_rmse=float(np.mean(paired_differences)),
                paired_max_difference_rmse=max(paired_differences),
                metrics=aggregate(records),
                selected_metrics=aggregate(selected) if selected else None,
                selected_seeds=[r["initialization_seed"] for r in selected],
            )
            rows.append(row)
            (args.output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    (args.output / "draws.json").write_text(json.dumps(draws, indent=2) + "\n")
    report = dict(
        protocol=cfg,
        protocol_sha256=sha256(args.protocol),
        code_sha256=sha256(__file__),
        dependencies_sha256={
            name: sha256(resource_path(name))
            for name in ["shape_gaussian.py", "stress_3d.py", "diagnose_3d.py", "prepare.py"]
        },
        draws_sha256=sha256(args.output / "draws.json"),
        rows=rows,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
