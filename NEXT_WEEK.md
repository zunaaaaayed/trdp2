# Active handoff — worm uncertainty (17 October 2026)

Current work is on `experiment/worm-uncertainty`. See
[the worm protocol](experiments/worm_uncertainty/README.md) and
[supervisor clarification](docs/SUPERVISOR_DATASET.md). Parallel-beam synthetic
sinograms and single-worm deformation provenance are now confirmed. Projection
normalization and matching density volumes remain unresolved. No GPU run has
started. The dated Week 4 plan below is retained as history; its unresolved-axis
questions are superseded by the supervisor clarification.

# Week 4 handoff — independent-volume preparation

Prepared 8 October 2026 for the next reporting week, 10–16 October.
Research objective: reconstruction generalization across limited-angle acquisition
configurations and independent volumes, with settings frozen before evaluation.

## Current direction — updated 12 October 2026

The user has designated the dataset supplied by co-supervisor **Ms. Anna-Louise
Brun** as the dataset to use going forward. Its original archive is
`~/espaces/travail/trdp2/worm_data.zip`
with extracted TIFF files in `~/espaces/travail/trdp2/worm_data/`. See
[the dataset handoff](docs/SUPERVISOR_DATASET.md) for download integrity and intake.
The previous LIDC-IDRI expansion plan is deferred. Its source records, downloaded
subject and frozen splits remain historical preparation evidence. The existing
TV 0.05 selection is evidence for the supplied chest convention; do not assume it
transfers unchanged to the new data's measurement scale or geometry.

All 144 TIFFs are extracted and audited; each is a finite, nonnegative float64
array of shape (510, 100, 310). See the dataset README and `array_audit.json`.

Next actions: establish
projection/volume axes and acquisition geometry, identify any reference volumes
and independent subjects, document intensity units and provenance, then define a
new benchmark protocol. Validate tiny inputs before GPU reconstruction. Do not
interpret a sinogram filename as proof of a specific scanner model or angle range.

## Completed and frozen

- Original pilot is preserved. New start-0° and start-90° 120° baselines and all
  four ordinary-TV controls completed at 30,000 iterations.
- **TV 0.05 is the frozen ordinary-TV reference**, selected by the recorded rule.
  Mean PSNR over the two orientations: 29.652 (TV 0.025), 29.752 (0.05),
  29.615 (0.1) dB. This is one development volume, not population evidence.
- The accepted start90/TV0.1 result is the `__chani_20261008` attempt. Ignore the
  stale `running` state of its unfinished cheonech predecessor when aggregating.
- [Frozen metrics](experiments/independent_volume_evaluation/tv_reference_frozen.json)
  bind all six accepted results and the selection rule.
- [Week 3 report and Overleaf package](Reports/Week3/README.md) are complete.
  The earlier PDF named `TRDP2_Week1.pdf` is the user's Week 2 report; its filename
  is preserved to avoid breaking existing references.
- No experiment is queued by this handoff. Start no archived shell queue blindly.

## Previous milestone: LIDC-IDRI intake (deferred)

1. **Pilot provenance:** inherited `raw_metadata.py` identifies `0_chest` as
   LIDC-IDRI-0001. Verify that mapping against the supplied reference; it is still
   a candidate identity, not an established source match. Subject 0001 is excluded
   from the proposed cohort.
2. **Development pixel audit:** decode subject 0002 and validate rescale slope /
   intercept, pixel padding, physical slice ordering, orientation and spacing.
   Header checks already passed; pixel/preprocessing validation has not run.
3. **Freeze preprocessing:** record intensity units/mapping, physical resampling,
   crop/FOV and scanner length units. Test these on small phantoms and development
   data before any held-out reconstruction. Do not silently use per-volume min/max
   normalization or interpret normalized image values as calibrated attenuation.
4. **Second development intake:** download and audit subject 0003 using the frozen
   metadata URL and the same checks. Do not change validation/final assignments
   in response to reconstruction results.
5. **Release condition for GPU evaluation:** verified subject provenance, validated
   preprocessing, regenerated physical scanner geometry, tiny isolated GPU tests,
   new input/code hashes and a reviewed single-case manifest. Only then run a
   familiar 120° configuration with frozen TV 0.05.

The cohort is a convenience sample: 0002/0003 development, 0004/0005 validation,
0006/0007 final-test candidates. It supports a small transfer study. Raw data exist
only for 0002. No independent-volume reconstruction has run. The proposed split
remains conditional on geometry/identity eligibility checks, with any exclusions
recorded before inspecting reconstruction outcomes.

## Start a session

On a university machine with the shared checkout:

```bash
cd ~/espaces/travail/trdp2/r2_gaussian
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export MPLBACKEND=Agg
hostname
git status --short --branch
python -c 'import sys; print(sys.executable)'
python -B -m unittest discover -s experiments/limited_angle_generalization/tests -q
python -B -m unittest discover -s experiments/ordinary_tv_controls -p 'test_*.py' -q
python -B -m unittest discover -s experiments/independent_volume_evaluation -p 'test_*.py' -q
```

All three suites are CPU checks (43 tests at handoff). `-B` avoids new bytecode
cache files. Inspect GPU availability only when preparing GPU work. The current
research stack is Python 3.9.23, torch 2.1.2+cu118, CUDA 11.8, NumPy 1.24.4 and
Open3D 0.18; the known GPU model is RTX 4000 Ada. Use isolated backend smoke tests
on a changed host/environment before training. TIGRE forward calls must not share
live PyTorch CUDA tensors in one process.

[The intake README](experiments/independent_volume_evaluation/README.md) provides
an explicit read-only header audit command with a fresh output report name.
Completed training destinations deliberately reject reruns. The original baseline
runner also requires an exact source-file set and can reject v4 after additive
research scripts; do not disable or rewrite historical hash checks. The ordinary-TV
runner validates the unchanged frozen baseline files and records additive code.

## Data and migration

Current source data, derived projections, reconstructions, checkpoints and logs are
in ignored `data/` and `output/` directories on shared university storage. Confirm
those paths are available before moving to a computer without that shared mount.
A Git clone alone is insufficient. The 7 October transfer archive predates the
completed controls and first raw independent-volume download; create a new archive
and checksums when making the next actual transfer. Never overwrite the old archive.

Keep interrupted attempts for audit; their checkpoints are not exact continuation
points because RNG/camera-stack state is incomplete. Fresh retries must use a new
`--attempt` label and the original initializer. Pause/stop instructions apply to
our queue only; do not terminate unrelated GPU workloads.

## Scientific questions still open

- Does the frozen reference transfer under a consistent intensity/geometry scale?
- Does acquisition dependence persist across genuinely independent subjects?
- How much of the orientation effect comes from FDK initialization?
- Can a geometry descriptor predict operator sensitivity before choosing a penalty?
- Does a proposed geometry term outperform ordinary TV, an equally tuned uniform
  penalty and a wrong-orientation control while preserving detail?

Geometry regularization effectiveness and novelty remain unverified. Complete
source comparisons, including the outstanding SPARK full-text review, before
novelty claims. Keep the three reserved acquisition configurations untouched until
the next protocol is frozen.
