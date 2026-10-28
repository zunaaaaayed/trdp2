# Current continuation — 28 October 2026

Week 6 began 24 October. NEXT_WEEK.md retains the completed Weeks 4–5 summary
and then-next-week plan. Root README is cumulative; this file is the active handoff.

## Current state

Branch `experiment/worm-uncertainty`, base commit `8ba48b5`. Matched completion,
initialization adapter and bounded Langevin feasibility are new, uncommitted work.
No push or merge was authorized for this session. Historical results remain intact.
No applicable AGENTS.md was found; the old graph does not cover current work.

Read [frozen matched design](COMPLETION_MATCHED_V2.md) and
[results, figures and failures](COMPLETION_RESULTS_V2.md). These supersede the
preparation-only trainer plan, not the historical inference conclusions.

Completed: exact sparse grid/noise matching; angular connectivity and sensitivity
checks; CPU gradient smoke; both 20-epoch learned arms; familiar validation;
frozen-checkpoint unseen-mask evaluation; initialization-only adapter; one bounded
Gaussian comparison (11/12 fits before the budget stop); one independent bounded MALA feasibility. **81 tests pass**.
Style checks pass. No reserved outcomes were evaluated or training sweep launched.

Familiar error: interpolation 39.29%, mask 27.51%, geometry 27.49%. Geometry selected
using familiar validation only; all four files improve over interpolation. Unseen
mask average: interpolation 50.90%, selected model 50.91%; narrow 90° transfer fails
in several conditions. Do not claim a robust geometry-conditioning advantage.
Refinement withheld errors were 24.14% acquired-only, 14.96% interpolation and
18.24% learned; all gates fail and 4/4/3 starts are unmatched. The learned advantage
over interpolation is not retained. No additional fit was launched.
The MALA study fails mixing on both evaluation cases; stop sampler expansion.
No calibrated posterior or physical-density accuracy is established.

Original training outputs had a metadata/model-name collision at key `mask`.
The first integration attempt failed before fits. Original arrays and metrics are
preserved; masks reconstructed from the frozen protocol reproduce every metric.
New outputs use `acquired_mask`; regression tests cover the repair. The actual
integration output is `completion_refinement_v2_retry1`, not the failed v2 folder.

## Verification commands (no experiments restarted)

```bash
cd ~/espaces/travail/trdp2/r2_gaussian
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib
python -m experiments.worm_uncertainty.studies.verify_matched_completion
python -m experiments.worm_uncertainty.studies.verify_nonlinear_audit
python -m unittest discover -s experiments/worm_uncertainty/tests -t . -p 'test_*.py'
make check PYTHON=/tmp/trdp2-format-tools/bin/python
```

Normally under one minute, CPU only. The temporary formatter may require the
existing development-tools setup on another host; do not modify the CUDA stack.

## Next decision and budget

First review the initialization result and acceptance failures in the result note.
Do not extend the unchanged optimizer or sampler runs automatically. A new
completion-development protocol could test explicit span diversity with a newly
reserved acquisition evaluation; existing shift results are now known and cannot
be reused as untouched tests. Keep matched sampling and initialization-only use.
No further training or reserved evaluation has been launched. A proposed repeat
would retain a prospective cap of 20 GPU minutes per learned arm, although this
pilot took under 30 seconds per arm. Freeze its masks/selection and obtain scope
approval before launching a larger study. A complete 12-fit Gaussian comparison
needs approximately 11 CPU minutes at the measured runtime, exceeding the present
10-minute integration cap; it needs a new prospective budget, not a silent extension.

## Artifacts to transfer when changing computers

Git holds protocols, code, compact summaries and plots, not raw TIFFs/checkpoints.
Retain `../worm_data` and ignored `output/worm_uncertainty/` directories:
`scaled_acceptance_v1`, `posterior_comparison_v1`, `nonlinear_posterior_v1`,
`worm_shape_pilot_v1`, `completion_matched_smoke_v2`, `completion_matched_v2`,
`completion_artifact_repair_v2`, failed `completion_refinement_v2`,
`completion_refinement_v2_retry1`, `langevin_feasibility_v1`, and their logs.
The repair folder preserves exact executed pre-fix trainer/data-helper sources.
Verify hashes after transfer. Use tmux for future remote experiments; recheck GPU
availability and preserve TIGRE/PyTorch process isolation if native backends return.
