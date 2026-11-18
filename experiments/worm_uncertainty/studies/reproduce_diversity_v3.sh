#!/usr/bin/env bash
# Repeat the frozen v3 budget in a fresh output root, reusing verified input caches.
# Run inside tmux with trdp2-r2 activated. No reserved data or sampler is used.
set -euo pipefail
if [ "$#" -ne 1 ]; then
    echo "Usage: bash experiments/worm_uncertainty/studies/reproduce_diversity_v3.sh NEW_OUTPUT_ROOT" >&2
    exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
run_root="$1"
reference_root="$PWD/output/worm_uncertainty/acquisition_diversity_v3"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib
python -c 'import torch; assert torch.cuda.is_available(), "Activate trdp2-r2 and check host CUDA"'
python -m experiments.worm_uncertainty.studies.verify_acquisition_diversity
mkdir -- "$run_root"
ln -s -- "$reference_root/profile" "$run_root/profile"
ln -s -- "$reference_root/historical_completion" "$run_root/historical_completion"
cp -- "$reference_root/training_budget.json" "$run_root/training_budget.json"
python -u -m experiments.worm_uncertainty.studies.acquisition_diversity train --root "$run_root"
python -m experiments.worm_uncertainty.studies.freeze_diversity_analysis --root "$run_root"
python -u -m experiments.worm_uncertainty.studies.acquisition_diversity audit --root "$run_root"
python -u -m experiments.worm_uncertainty.studies.diversity_refinement --root "$run_root"
if [ -f "$run_root/audit/ensemble.npz" ]; then
    python -m experiments.worm_uncertainty.studies.diversity_risk_sensitivity --root "$run_root"
fi
python -m experiments.worm_uncertainty.studies.summarize_diversity --root "$run_root"
python -m experiments.worm_uncertainty.studies.verify_acquisition_diversity --root "$run_root"
python -m experiments.worm_uncertainty.plots.plot_acquisition_diversity --root "$run_root" --output "$run_root/figures"
if [ -f "$run_root/risk_sensitivity.json" ]; then
    python -m experiments.worm_uncertainty.plots.plot_diversity_risk --root "$run_root" --output "$run_root/figures"
fi
