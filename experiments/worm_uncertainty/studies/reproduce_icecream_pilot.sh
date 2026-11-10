#!/usr/bin/env bash
# Explicit reproduction of the final FP32-amended design; not an automatic retry.
# Usage: bash experiments/worm_uncertainty/studies/reproduce_icecream_pilot.sh NEW_ROOT
set -euo pipefail
if [[ $# != 1 || -e "$1" ]]; then
    echo 'Provide one fresh output root; existing outputs are never overwritten.' >&2
    exit 2
fi
pilot_root=$1
pilot_python=output/worm_uncertainty/alternative_reconstruction_sources/env/bin/python
pilot_original=output/worm_uncertainty/icecream_pilot_v1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib
"$pilot_python" -m experiments.worm_uncertainty.studies.icecream_pilot geometry --root "$pilot_root"
# Reuse the frozen budget, not a newly selected throughput budget.
cp "$pilot_original/training_freeze.json" "$pilot_root/training_freeze.json"
cp "$pilot_original/precision_amendment.json" "$pilot_root/precision_amendment.json"
"$pilot_python" -m experiments.worm_uncertainty.studies.audit_icecream_operator --root "$pilot_root"
for pilot_case in phantom_base_full phantom_base_120 phantom_base_90 phantom_independent_90 phantom_rotated_120; do
    "$pilot_python" -m experiments.worm_uncertainty.studies.icecream_pilot prepare --root "$pilot_root" --case "$pilot_case"
    "$pilot_python" -m experiments.worm_uncertainty.studies.icecream_pilot train --root "$pilot_root" --case "$pilot_case"
    "$pilot_python" -m experiments.worm_uncertainty.studies.icecream_pilot evaluate --root "$pilot_root" --case "$pilot_case"
done
# This skips the already diagnosed AMP failure; the historical record remains preserved.
"$pilot_python" -m experiments.worm_uncertainty.studies.run_icecream_pilot --root "$pilot_root" --attempt fp32
"$pilot_python" -m experiments.worm_uncertainty.studies.verify_icecream_pilot --root "$pilot_root"
