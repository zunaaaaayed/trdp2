"""Locate organized resources and verify historical source without rewriting provenance."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
LAYOUT = json.loads((ROOT / "file_layout.json").read_text())


def resource_path(name):
    """Resolve a unique historical basename to its current code/protocol location."""
    if name not in LAYOUT["files"]:
        raise ValueError("unknown worm experiment resource: " + str(name))
    return ROOT / LAYOUT["files"][name]["path"]


def verify_source(name, expected):
    """Verify current bytes or the exact pre-organization Git snapshot.

    Historical verification does not claim current bytes are unchanged. Numerical
    tests and the documented import/path-only migration validate current behavior.
    """
    current = resource_path(name)
    if hashlib.sha256(current.read_bytes()).hexdigest() == expected:
        return "current"
    record = LAYOUT["files"][name]
    if record["sha256_before"] != expected:
        raise ValueError("source hash does not match current or archived version: " + name)
    revision = LAYOUT["before_commit"] + ":experiments/worm_uncertainty/" + name
    content = subprocess.check_output(["git", "show", revision], cwd=REPOSITORY)
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("archived source hash mismatch: " + name)
    return "archived " + LAYOUT["before_commit"][:7]
