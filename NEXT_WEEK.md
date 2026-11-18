# Weeks 4–5 closeout — 21 October 2026

## Completed work: 12–21 October 2026

Weeks 4–5 are complete. The combined report is due **22 October**, with the
presentation on **23 October**.

- Shifted from TV tuning to uncertainty in synthetic parallel-beam worm CT.
  Audited 144 TIFF sinograms: deformations of one worm, without matching voxel
  truth. Physical projection scaling remains unresolved.
- Prepared controlled Gaussian noise, acquired/withheld masks, analytic projector
  checks, FBP/SART comparisons and native Gaussian slab baselines. Earlier noisy
  1,500-Gaussian withheld errors are 13.36%/17.79% at 120°/90°.
- The corresponding density-only posterior fails calibration: nominal 95%
  predictive intervals cover only 30.66%/24.17% of withheld foreground rays.
- Known-3D phantom controls separate omitted geometry uncertainty, missing
  structure and optimization failures. Acquired-data initialization plus fixed
  extra starts yields 40/40 passing stability checks and 95.45–96.22% voxel
  coverage on two fresh matched-model phantoms/two windows. This remains
  conditional synthetic evidence, with adaptive fallback on failed draws.
- The first small worm shape pilot fails stability. A final 16-fit controlled
  study improves initialization from 5/16 to 16/16 appreciably active components.
  At budget 450, acquired/withheld errors are 11.20%/19.84% for 16 components and
  8.37%/23.81% for 32. Only 2/16 starts converge; all four settings fail stability.
  These sparse-sampling results are not directly comparable to the earlier slab.
- Better acquired fit has not established missing-view generalization or
  calibrated worm uncertainty. Representation and optimization errors remain
  entangled; independent-specimen transfer has not been tested.
- Saved protocols, hashes, all candidates, plots and interpretations. All runs
  completed; 42 CPU tests and formatting/lint checks for 51 experiment Python
  files pass. Prior chest results and accepted historical artifacts are retained.

## Plan for the next reporting week — started 24 October 2026

1. Freeze and test staged density → centre → full-shape optimization under a
   fixed shared prior. Retain convergence, active-bound and multistart agreement
   checks; evaluate withheld views without selecting starts by their error.
2. Once fitting is stable, test reserved angular offsets/spans and deformations.
   Evaluate uncertainty coverage together with interval width on known targets,
   including representation mismatch and nonlinear propagation controls.
3. Resolve the simulator's projection scaling, rotation convention and matching
   deformed voxel-density references with Anna-Louise before density claims.
4. Keep the experiment README detailed and the root README cumulative. Treat
   independent-specimen generalization as requiring additional data.
