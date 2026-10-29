# Current continuation — 29 October 2026

Week 6 began24 October. NEXT_WEEK.md remains the completed Weeks 4–5 summary and
then-next-week plan. Root README is cumulative; this is the active handoff.

## Current state

Branch `experiment/worm-uncertainty`, starting HEAD `ebb4754`. The v2 pilot was
committed and pushed. The completed v3 acquisition-diversity milestone was authorized for commit and
push by the user on 29 October. Use Git status and HEAD for the current commit
identity; the saved experiment provenance retains its original dirty source state.
Preserve all local files and ignored outputs. No applicable AGENTS.md was found. The existing Graphify
index has no worm_uncertainty entries, so source and saved artifacts were inspected.

Read the [frozen design](ACQUISITION_DIVERSITY_V3.md) and
[results with plots and limits](ACQUISITION_DIVERSITY_RESULTS_V3.md).
All nine A/B/C runs finished in tmux despite the chat interruption.
93 CPU tests, formatting/lint across 121 files, v2/v3 artifact verification and
local documentation links pass. The one-time
development audit and both balanced Gaussian cases are complete. No run is pending.
Do not restart the old experiments or open reserved outcomes.

## Findings

- Broad training B reduced narrow selection error62.65%→41.99% versus restricted A.
  On frozen development audit: interpolation 73.13%, A 65.89%, B 41.05%, C 42.80%.
  B improves every audit deformation for every seed versus interpolation.
- Geometry conditioning has a modest benefit over C, but C is slightly better in
  severe72° extrapolation. Familiar acquisitions favor A. No uniform dominance.
- Larger capacity was skipped by the frozen rule: training loss had not plateaued.
- Three-model disagreement ranks error better than angular distance in all 12 audit
  files at 50% retention (absolute error 0.1222 vs0.1878). A post-audit tie sensitivity
  gives 0.1886 for geometry. Strong high-error/low-disagreement failures remain.
- The historical missing learned/421 fit is appended separately. The old selected
  fits and failures are unchanged; the original 11-fit report remains historical.
- New Gaussian comparison completed 2 files×3 arms×2 starts. All 12 fits fail numerical
  acceptance. Refined withheld errors, acquired/interpolation/learned: worm153
  25.96/26.31/27.13%; worm32719.06/25.38/25.22%. No accepted reconstruction benefit.
- Sampler work remains paused. No posterior-calibration claim, voxel truth, verified
  physical-density scale or independent-organism generalization.

## Exact verification commands

```bash
cd ~/espaces/travail/trdp2/r2_gaussian
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib
python -m experiments.worm_uncertainty.studies.verify_acquisition_diversity
python -m experiments.worm_uncertainty.studies.verify_matched_completion
python -m unittest discover -s experiments/worm_uncertainty/tests -t . -p 'test_*.py'
make check PYTHON=/tmp/trdp2-format-tools/bin/python
```

CPU only, usually under one minute. The last formatter environment was cleared
from `/tmp` during interruption and restored with pinned Ruff 0.14.1 in a separate
venv. The research CUDA/PyTorch stack was not changed.

## Optional exact-budget reproduction (not launched)

The commands below are also packaged in
[reproduce_diversity_v3.sh](../studies/reproduce_diversity_v3.sh). After activating
`trdp2-r2`, run it in tmux with a fresh output directory argument. The shell script
passes `bash -n`; it was not executed because that would repeat completed training.


This repeats the completed design; it is not a new tuning experiment. The original
protocol, frozen budget and input cache are required for the same comparison.
The symlinks reuse verified cached inputs and the already completed historical
repair; the listed stages only read these shared inputs. Do not rerun that historical fit unnecessarily. Choose a fresh
root; `mkdir` and all scientific stages refuse an existing destination.

```bash
# After the activation commands above:
mkdir output/worm_uncertainty/acquisition_diversity_v3_reproduction
ln -s ../acquisition_diversity_v3/profile \
  output/worm_uncertainty/acquisition_diversity_v3_reproduction/profile
ln -s ../acquisition_diversity_v3/historical_completion \
  output/worm_uncertainty/acquisition_diversity_v3_reproduction/historical_completion
cp output/worm_uncertainty/acquisition_diversity_v3/training_budget.json \
  output/worm_uncertainty/acquisition_diversity_v3_reproduction/training_budget.json

# Run these commands inside a new tmux session after checking GPU availability.
tmux new-session -s trdp2-diversity-reproduction
python -u -m experiments.worm_uncertainty.studies.acquisition_diversity train \
  --root output/worm_uncertainty/acquisition_diversity_v3_reproduction
python -m experiments.worm_uncertainty.studies.freeze_diversity_analysis \
  --root output/worm_uncertainty/acquisition_diversity_v3_reproduction
python -u -m experiments.worm_uncertainty.studies.acquisition_diversity audit \
  --root output/worm_uncertainty/acquisition_diversity_v3_reproduction
python -u -m experiments.worm_uncertainty.studies.diversity_refinement \
  --root output/worm_uncertainty/acquisition_diversity_v3_reproduction
python -m experiments.worm_uncertainty.studies.diversity_risk_sensitivity \
  --root output/worm_uncertainty/acquisition_diversity_v3_reproduction
python -m experiments.worm_uncertainty.studies.summarize_diversity \
  --root output/worm_uncertainty/acquisition_diversity_v3_reproduction
python -m experiments.worm_uncertainty.studies.verify_acquisition_diversity \
  --root output/worm_uncertainty/acquisition_diversity_v3_reproduction
python -m experiments.worm_uncertainty.plots.plot_acquisition_diversity \
  --root output/worm_uncertainty/acquisition_diversity_v3_reproduction \
  --output output/worm_uncertainty/acquisition_diversity_v3_reproduction/figures
python -m experiments.worm_uncertainty.plots.plot_diversity_risk \
  --root output/worm_uncertainty/acquisition_diversity_v3_reproduction \
  --output output/worm_uncertainty/acquisition_diversity_v3_reproduction/figures
```

Observed budget: about 25 minutes for GPU workflow phases and 8 minutes for the new
single-thread Gaussian fits, plus CPU verification/plots. Use the same per-arm and
per-fit caps. The capacity-freeze command stops if its prospective condition is
met; do not bypass it to inspect audit outcomes. Risk scripts require an eligible
ensemble. On changed hardware, numerical results may differ; record the environment.
For a genuinely new throughput protocol, use the runner's `profile` stage in a fresh
root and document any changed batch/update budget as a new comparison.

## Smallest next experiment

A single selection-case objective-path diagnostic using saved learned/interpolation
initialized and refined Gaussian states: decompose acquired residual and shared prior
penalties while tracking their withheld-error difference along paired feasible paths. This can locate
where useful initialization is lost without another network or sampler sweep.
The path would not be the optimizer's actual trajectory. Freeze its analysis before
running it. No new replacement research direction has been launched.

## Transfer and storage

Keep `../worm_data`, the ignored `output/worm_uncertainty/acquisition_diversity_v3`
run (about 718 MiB), and the historical parent runs listed in the v2 result notes.
Large checkpoints/arrays are excluded from Git. Protocols, compact JSON/CSV summaries
and plots are versioned under `experiments/worm_uncertainty` and
`docs/figures/worm_uncertainty/acquisition_diversity_v3`. Verify archives/hashes after
moving machines. Preserve TIGRE/PyTorch process isolation if native backends return.
