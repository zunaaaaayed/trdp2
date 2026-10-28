"""Verify saved audit integrity and exact reuse without launching experiments."""

from experiments.worm_uncertainty.layout import verify_source

import argparse
import json
from pathlib import Path
import numpy as np
from experiments.worm_uncertainty.studies.audit_nonlinear_posterior import load_case
from experiments.worm_uncertainty.core.cholesky_gaussian import projection
from experiments.worm_uncertainty.core.posterior_local import gaussian_factor
from experiments.worm_uncertainty.data.prepare import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("output/worm_uncertainty/nonlinear_posterior_v1")
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/figures/worm_uncertainty/nonlinear_posterior_v1/summary.json"),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    for name, h in summary["source_artifacts_sha256"].items():
        if sha256(args.source / name) != h:
            raise ValueError("artifact hash mismatch: " + name)
    report = json.loads((args.source / "report.json").read_text())
    for name, h in report["provenance"]["sources"].items():
        verify_source(name, h)
    for name, h in report["provenance"]["parents"].items():
        if sha256(Path(name)) != h:
            raise ValueError("parent hash mismatch")
    parent = Path("output/worm_uncertainty/scaled_acceptance_v1")
    old = Path("output/worm_uncertainty/posterior_comparison_v1")
    oldr = json.loads((old / "report.json").read_text())
    cov = np.load(old / "covariances.npz")
    for row in report["rows"]:
        name = row["case"]
        target, p, _, _, _, _, _, _, _, index, _ = load_case(name, parent, old)
        _, jac = projection(p, target.angles, target.uv)
        for model, columns in [("density", np.arange(45, 50)), ("joint", np.arange(50))]:
            f, _ = gaussian_factor(jac[:, columns], target.sd[columns], target.sigma)
            np.testing.assert_allclose(
                cov[name + "_" + model][np.ix_(columns, columns)], f @ f.T, rtol=1e-9, atol=1e-12
            )
            draws = np.load(args.source / (name + "_" + model + ".npz"))
            z = np.random.default_rng(report["protocol"]["seed"] + index).normal(
                size=(1024, len(columns))
            )
            np.testing.assert_allclose(draws["parameters"][:1024, columns], p[columns] + z @ f.T)
            valid = draws["valid"]
            if not np.isneginf(draws["log_target"][~valid]).all():
                raise ValueError("invalid support accounting")
            np.testing.assert_allclose(
                draws["log_target"][valid], (draws["log_likelihood"] + draws["log_prior"])[valid]
            )
            oldrow = next(r for r in oldr["rows"] if r["case"] == name)
            if 1 - valid[:1024].mean() != oldrow["bounds"][model]["any_fraction"]:
                raise ValueError("reused bound accounting differs")
        if not row["methods"]["density"]["adequate"] or row["methods"]["joint"]["adequate"]:
            raise ValueError("saved decision does not match documented outcomes")
    reference = json.loads((args.source / "reference.json").read_text())
    for row in reference["rows"]:
        if row["passed"] or row["bounds_violations"] != 0:
            raise ValueError("reference decision mismatch")
        target = load_case(row["case"], parent, old)[0]
        draws = np.load(args.source / (row["case"] + "_reference.npz"))["parameters"]
        if ((draws < target.lower) | (draws > target.upper)).any():
            raise ValueError("reference outside support")
    print(
        "PASS: hashes, objective factors, covariances, original 1024 draws, support and failure accounting"
    )


if __name__ == "__main__":
    main()
