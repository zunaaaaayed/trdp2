"""Frozen fresh-phantom validation of a scale-aware local numerical rule."""

import argparse
import json
from pathlib import Path

import numpy as np
from cholesky_gaussian import convert_euler, projection
from gaussian_l2 import relative_l2
from measured_initializer import perturb
from phantom_replication import acquisition_mask, observation
from prepare import sha256
from scaled_acceptance import assess
from worm_stationarity import continue_fit, diagnostics


def fresh_phantom(seed, elongated):
    rng = np.random.default_rng(seed)
    centres = np.column_stack([rng.uniform(-0.35, 0.35, (5, 2)), rng.uniform(-0.06, 0.06, 5)])
    widths = rng.uniform(0.07, 0.13, (5, 3))
    if elongated:
        widths *= np.array([0.1, 10.0, 1.0])
    rotations = rng.uniform(-0.2, 0.2, (5, 3))
    rotations[:, 2] = rng.uniform(-1.4, 1.4, 5)
    density = rng.uniform(1.3, 3.2, 5)
    return convert_euler(np.r_[centres.ravel(), np.log(widths).ravel(), rotations.ravel(), density])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("scaled_acceptance_protocol_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if not args.execute:
        print(json.dumps(design, indent=2))
        return
    if args.output.exists():
        raise ValueError("output exists")
    base = Path(__file__).with_name("worm_shape_protocol_v1.json")
    if sha256(base) != design["base_protocol_sha256"]:
        raise ValueError("base protocol mismatch")
    cfg = json.loads(base.read_text())
    degrees = np.arange(25) * 7.2
    angles = np.deg2rad(degrees)
    uv = np.array([[u, z] for z in np.linspace(-0.16, 0.16, 9) for u in np.linspace(-1, 1, 129)])
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(args.protocol.read_text())
    sources = [
        "validate_scaled_acceptance.py",
        "scaled_acceptance.py",
        "worm_stationarity.py",
        "staged_density.py",
        "cholesky_gaussian.py",
        "gaussian_l2.py",
        "measured_initializer.py",
        "phantom_replication.py",
        "shape_gaussian.py",
        "prepare.py",
        "rotation_sensitivity.py",
        "stress_3d.py",
    ]
    provenance = dict(
        protocol=design,
        base_configuration=cfg,
        source_sha256={n: sha256(Path(__file__).with_name(n)) for n in sources},
    )
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    prepared = []
    arrays = dict(angles=angles, uv=uv)
    for seed in design["phantom_seeds"]:
        for shape in design["shapes"]:
            truth = fresh_phantom(seed, shape == "elongated")
            clean = projection(truth, angles, uv)[0].reshape(25, len(uv))
            arrays[f"{seed}_{shape}_truth"] = truth
            arrays[f"{seed}_{shape}_clean"] = clean
            for offset in design["offsets_degrees"]:
                mask = acquisition_mask(degrees, offset)
                if mask.sum() != 17:
                    raise ValueError("17 acquired views required")
                for noise_seed in design["noise_seeds"]:
                    y, sigma = observation(clean, mask, noise_seed, design["relative_noise_sd"])
                    name = f"{seed}_{shape}_{offset}_{noise_seed}"
                    arrays[name + "_y"] = y
                    arrays[name + "_mask"] = mask
                    prepared.append(
                        (name, seed, shape, offset, noise_seed, truth, clean, mask, y, sigma)
                    )
    np.savez_compressed(args.output / "inputs.npz", **arrays)
    cases = []
    fits = []
    for name, seed, shape, offset, noise_seed, truth, clean, mask, y, sigma in prepared:
        local = dict(cfg, components=5, noise_sigma=sigma)
        initial = perturb(truth, design["start_seed"], local)
        stages = {}
        p = initial.copy()
        for stage in ["initial", "partial", "final", "polish", "oracle"]:
            if stage == "initial":
                old = diagnostics(p, y, angles[mask], uv, truth, local)
                fit = dict(
                    parameters=p.tolist(),
                    objective=old["objective"],
                    converged=old["projected_gradient_inf"] <= 1e-6,
                    nfev=0,
                    status=None,
                )
            else:
                if stage == "oracle":
                    p = truth.copy()
                schedule = dict(design["old_rule"], additional_nfev=design["budgets"][stage])
                fit = continue_fit(p, y, angles[mask], uv, truth, local, schedule)
                p = np.array(fit["parameters"])
            assessment = assess(p, y, angles[mask], uv, truth, local, design["thresholds"])
            fit.update(
                case=name,
                stage=stage,
                assessment=assessment,
                truth_density_error=relative_l2(p, truth),
            )
            stages[stage] = fit
            fits.append(fit)
        final = np.array(stages["final"]["parameters"])
        polished = np.array(stages["polish"]["parameters"])
        oracle = np.array(stages["oracle"]["parameters"])
        pred = projection(final, angles[mask], uv)[0]
        pred_polish = projection(polished, angles[mask], uv)[0]
        objective_gain = max(
            0.0, stages["final"]["objective"] - stages["polish"]["objective"]
        ) / max(1.0, stages["final"]["objective"])
        acquired_change = float(
            np.linalg.norm(pred_polish - pred) / max(np.linalg.norm(pred), 1e-30)
        )
        density_change = relative_l2(polished, final)
        limits = design["validation"]
        polish_stable = bool(
            objective_gain <= limits["max_relative_objective_gain"]
            and acquired_change <= limits["max_acquired_prediction_change"]
            and density_change <= limits["max_density_change"]
        )
        checkpoint_checks = {}
        for stage in ["initial", "partial", "final"]:
            snapshot = np.array(stages[stage]["parameters"])
            snapshot_prediction = projection(snapshot, angles[mask], uv)[0]
            gain = max(0.0, stages[stage]["objective"] - stages["polish"]["objective"]) / max(
                1.0, stages[stage]["objective"]
            )
            prediction_change = float(
                np.linalg.norm(pred_polish - snapshot_prediction)
                / max(np.linalg.norm(snapshot_prediction), 1e-30)
            )
            volume_change = relative_l2(polished, snapshot)
            stable = bool(
                gain <= limits["max_relative_objective_gain"]
                and prediction_change <= limits["max_acquired_prediction_change"]
                and volume_change <= limits["max_density_change"]
            )
            checkpoint_checks[stage] = dict(
                accepted=stages[stage]["assessment"]["accepted"],
                stable_after_polishing=stable,
                objective_gain=gain,
                acquired_prediction_change=prediction_change,
                density_change=volume_change,
            )
        accepted = stages["final"]["assessment"]["accepted"]
        row = dict(
            case=name,
            phantom_seed=seed,
            shape=shape,
            offset=offset,
            noise_seed=noise_seed,
            stages={
                stage: {
                    k: f[k]
                    for k in [
                        "assessment",
                        "truth_density_error",
                        "objective",
                        "converged",
                        "nfev",
                        "status",
                    ]
                }
                for stage, f in stages.items()
            },
            polish_relative_objective_gain=objective_gain,
            polish_acquired_prediction_change=acquired_change,
            polish_density_change=density_change,
            polish_stable=polish_stable,
            checkpoint_checks=checkpoint_checks,
            accepted_final_unstable=bool(accepted and not polish_stable),
            oracle_objective_relative_gap=abs(
                stages["polish"]["objective"] - stages["oracle"]["objective"]
            )
            / max(1.0, min(stages["polish"]["objective"], stages["oracle"]["objective"])),
            oracle_density_disagreement=relative_l2(polished, oracle),
        )
        cases.append(row)
        (args.output / "fits.json").write_text(json.dumps(fits, indent=2) + "\n")
        (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    completed=len(cases),
                    case=name,
                    accepted=accepted,
                    polish_stable=polish_stable,
                    density_error=stages["final"]["truth_density_error"],
                )
            ),
            flush=True,
        )
    accepted_initial = sum(c["stages"]["initial"]["assessment"]["accepted"] for c in cases)
    failed = sum(
        check["accepted"] and not check["stable_after_polishing"]
        for c in cases
        for check in c["checkpoint_checks"].values()
    )
    no_final = sum(
        not any(c["stages"][stage]["assessment"]["accepted"] for stage in ["final", "oracle"])
        for c in cases
    )
    decision = dict(
        initial_controls_accepted=accepted_initial,
        accepted_checkpoints_unstable=failed,
        cases_without_accepted_final_or_oracle=no_final,
        numerical_rule_validation_passed=bool(
            accepted_initial == 0 and failed == 0 and no_final == 0
        ),
        note="Local numerical validation only. No automatic posterior launch or worm gate replacement; independent-start disagreement and active bounds must be assessed separately.",
    )
    report = dict(
        **provenance,
        cases=cases,
        decision=decision,
        artifacts_sha256={
            n: sha256(args.output / n)
            for n in ["fits.json", "cases.json", "inputs.npz", "protocol.json", "provenance.json"]
        },
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
