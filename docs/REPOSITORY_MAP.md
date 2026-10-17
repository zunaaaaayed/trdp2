# Repository map and retention policy

The weekly summary and next-week plan are in [NEXT_WEEK.md](../NEXT_WEEK.md).
Detailed weekly results belong in the root README; each experiment README owns
its plans, methods, outputs and interpretations.

| Location | Role | Retention |
|---|---|---|
| `experiments/worm_uncertainty/` | Current noise, FBP/SART and uncertainty development | Active |
| `experiments/independent_volume_evaluation/` | Frozen chest TV results, deferred cohort and DICOM intake | Historical/deferred |
| `experiments/ordinary_tv_controls/` | Paired-control runner, tests and original selection protocol | Completed controls; reusable guarded tooling |
| `experiments/limited_angle_generalization/` | Source-bound manifests, generation, verification, baseline workflow | Preserve paths and hashes |
| `experiments/limited_angle_120/` | Completed historical pilot and diagnostics | Preserve as evidence |
| `experiments/baseline_chest_50views/` | Initial baseline environment and reproduction records | Preserve as evidence |
| `Reports/` | Previous report and Week 3 source, plots, PDFs and Overleaf bundle | Published deliverables; preserve |
| `train.py`, `initialize_pcd.py`, `r2_gaussian/` | Inherited reconstruction implementation and native dependencies | Frozen baseline; preserve attribution/licenses |
| `data_generator/`, `scripts/`, `test.py`, `environment.yml` | Inherited input/analysis tooling and setup | Still referenced and potentially needed; not assumed obsolete |
| `README_UPSTREAM.md`, `assets/` | Original usage documentation and linked illustrations | Attribution/reference material |
| `TRDP2_CODEX_HANDOFF.md` | Historical first-session requirements | Historical specification; current status is in NEXT_WEEK |
| `../worm_data/`, `../worm_data.zip` | Current supervisor dataset and verified original archive | Outside Git; preserve raw values and intake metadata |
| `data/`, `output/`, `logs/` | Local raw inputs, experiment products and operational logs | Ignored by Git; valuable research artifacts |
| `graphify-out/` | Local navigation graph | Regenerable index, not source of truth |

Old manifests and dated handoffs are historical snapshots, not duplicate current
instructions. Do not edit their dates/statuses to make them appear current. In
particular, keep the ordinary-TV selection protocol and its recorded hashes.

Python bytecode and LaTeX auxiliary files can be regenerated and are safe cleanup
targets when untracked. Compiled CUDA libraries, raw images, checkpoints, successful
or failed experiment logs, are preserved research evidence. Obsolete backup archives may be deleted after
member-by-member verification against retained originals, with a cleanup record.

Before deleting or moving a tracked file, check frozen manifests and report links.
The source inventory includes tracked Python, CUDA/C++, headers, YAML and submodule
identities. Cosmetic reorganization of those files can invalidate reproducibility.
No blanket `git clean -xfd` should be used on this research checkout: it would
remove the ignored datasets and completed results.

The 17 October storage cleanup is recorded in [the deletion inventory](maintenance/cleanup_20261017.json).
Unused broad upstream download ZIPs are removed; extracted chest inputs, accepted
reconstructions and the supervisor worm dataset are retained.
