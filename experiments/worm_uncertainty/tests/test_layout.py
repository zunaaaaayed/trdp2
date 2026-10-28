"""Resource and archival checks that guard reorganized experiment execution."""

import hashlib
import json
import unittest
from experiments.worm_uncertainty.layout import LAYOUT, REPOSITORY, resource_path, verify_source
from experiments.worm_uncertainty.core.native_parallel import REPO


class LayoutTests(unittest.TestCase):
    def test_all_resources_exist_and_protocols_are_unchanged(self):
        for name, record in LAYOUT["files"].items():
            path = resource_path(name)
            self.assertTrue(path.is_file(), name)
            if name.endswith(".json"):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256_before"]
                )
                json.loads(path.read_text())

    def test_native_repository_root_and_unknown_resource(self):
        self.assertEqual(REPO, REPOSITORY)
        self.assertTrue((REPO / "r2_gaussian").is_dir())
        with self.assertRaises(ValueError):
            resource_path("../../outside.py")

    def test_historical_source_verification_is_strict(self):
        name = "nonlinear_target.py"
        self.assertTrue(
            verify_source(name, LAYOUT["files"][name]["sha256_before"]).startswith("archived")
        )
        current = hashlib.sha256(resource_path(name).read_bytes()).hexdigest()
        self.assertEqual(verify_source(name, current), "current")
        with self.assertRaises(ValueError):
            verify_source(name, "0" * 64)
