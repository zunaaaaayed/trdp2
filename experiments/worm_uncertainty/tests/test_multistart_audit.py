import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.worm_uncertainty.core.cholesky_gaussian import convert_euler
from experiments.worm_uncertainty.studies.audit_multistart import main
from experiments.worm_uncertainty.data.prepare import sha256


class MultistartAuditTests(unittest.TestCase):
    def audit(self, second_converged):
        p = convert_euler(
            np.r_[[0.1, -0.1, 0.02], np.log([0.2, 0.1, 0.15]), [0.1, 0.2, 0.3], 0.7]
        ).tolist()
        cfg = dict(
            angles_deg=dict(count=6, step=30),
            detector_shape=[2, 3],
            noise_repeats=1,
            cases=[dict(name="case", span=90, start=0)],
            stability_gate=dict(
                minimum_converged=2,
                relative_objective_gap=1e-5,
                relative_prediction_disagreement=1e-4,
            ),
        )
        records = [
            dict(
                case="case",
                noise_repeat=0,
                converged=True,
                objective=10.0,
                active_indices=[],
                parameters=p,
                initialization_seed=1,
                metrics=dict(voxel=dict(rmse=999.0)),
            ),
            dict(
                case="case",
                noise_repeat=0,
                converged=second_converged,
                objective=10.000001,
                active_indices=[],
                parameters=p,
                initialization_seed=2,
                metrics=dict(voxel=dict(rmse=0.0)),
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "draws.json").write_text(json.dumps(records))
            (path / "report.json").write_text(
                json.dumps(dict(protocol=cfg, draws_sha256=sha256(path / "draws.json")))
            )
            with (
                patch(
                    "sys.argv",
                    ["audit", "--input", str(path), "--output", str(path / "audit.json")],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if second_converged:
                    main()
                else:
                    with self.assertRaises(SystemExit) as caught:
                        main()
                    self.assertEqual(caught.exception.code, 2)
            return json.loads((path / "audit.json").read_text())

    def test_selection_ignores_truth_metrics(self):
        result = self.audit(True)
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["records"][0]["selected_seed"], 1)

    def test_single_converged_candidate_fails(self):
        self.assertFalse(self.audit(False)["all_passed"])


if __name__ == "__main__":
    unittest.main()
