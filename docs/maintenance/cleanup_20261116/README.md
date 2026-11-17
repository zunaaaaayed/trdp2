# Storage cleanup — 16 November 2026

The user explicitly requested deletion of obsolete experiments after reporting,
followed by committing and pushing Week8 work. No experiment was running.

## Retained

- `../worm_data`: original144 synthetic worm deformation TIFFs, unchanged.
- Current Python3.11/PyTorch2.8 environment:
  `output/worm_uncertainty/alternative_reconstruction_sources/env`.
- Pinned official ICECREAM source required by adapter tests; other small method sources.
- All Week8 bridge, ablation, phantom and SART-duration run artifacts.
- V3 checkpoints, schedules, JSON records and profile input/noise caches needed by
  current bridge/training code. Large historical primary/audit prediction arrays
  are removed; the current checkpoint-based studies remain usable.
- Compact historical JSON/CSV metrics, protocols, logs, source snapshots, report
  figures, papers, TeX/PDF reports and Git history. Original chest dataset source
  under `data/synthetic_dataset` and independent-volume intake records remain.

## Removed

- Retired Gaussian environment `../tools/miniforge3/envs/trdp2-r2`:
  15,136,106,402 apparent bytes. Its Conda metadata/history and pip freeze are saved
  here. Reinstallation and compiled-backend setup are required for old GPU Gaussian
  experiments; historical activation commands are no longer executable as-is.
- Conda package cache `../tools/miniforge3/pkgs`:4,578,071,430 apparent bytes.
  Base Miniforge remains; packages can be downloaded again if needed.
- 1,172 obsolete array/checkpoint/point-cloud/video files:5,925,771,205 bytes.
  This includes retired chest/TV training outputs, generated chest acquisitions,
  old worm inference artifacts and ICECREAM volumes/checkpoints. These completed
  studies are represented by their retained metrics, figures and provenance.
- Regenerable Graphify, Ruff and experiment bytecode caches.

`removed_artifacts.json` records each deleted scientific artifact's path, byte size
and SHA256. `plan.json` records the exact cleanup scope, and `completion.json`
records execution status. Environment/cache byte totals can overlap because Conda
uses hardlinks; use the final project disk measurement for actual space reclaimed.

This was deletion, not an archival copy. Historical raw-array verification, volume
inspection and checkpoint resume will require rerunning/regenerating those studies.
Hash records preserve artifact identities, not their contents. No failed result is
reclassified as successful. Old protocols and source hashes remain historical.

## Current commands

From the repository root:

```bash
PY=output/worm_uncertainty/alternative_reconstruction_sources/env/bin/python
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  $PY -m experiments.worm_uncertainty.studies.verify_phantom_density
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  $PY -m experiments.worm_uncertainty.studies.verify_sart_duration
```

These current studies retain all required arrays/checkpoints. The independent
Week8 reproduction commands and frozen protocols remain in their results notes.
Personal quota was not established; the user explicitly waived that execution gate.

## Final measurement — 17 November2026

Cleanup completed. Project allocation is26,762,878,976 bytes (**24.9GiB**),
down from the initial rounded49GiB: approximately24GiB reclaimed. The destination
filesystem now reports75GiB free. Tools occupy567MiB, output6.8GiB (including
the retained6.3GiB active environment), and original worm data about17GiB.

Post-pruning verification:108 tests pass;432 phantom metric rows regenerate;
1512 SART-duration rows replay; formatting/lint and diff checks pass. The active
environment's dependency check reports no broken requirements.
