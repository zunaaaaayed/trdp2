# Combined Weeks 4–5 handoff — updated 17 October 2026

## Combined Weeks 4–5 summary (progress through 17 October)

Weeks 4 and 5 are combined because the intervening supervisor meeting was
postponed. This is a provisional summary; the reporting period remains open.

- Switched focus from TV tuning on chest CT to uncertainty in synthetic
  parallel-beam worm reconstruction. Downloaded, extracted and audited 144 TIFFs;
  confirmed they are sinograms of deformations of one worm.
- Prepared reproducible Gaussian noise and measured/withheld angle masks.
- Verified a CPU projector/FBP path with an analytic disk phantom; completed 21
  paired FBP and six accepted SART development reconstructions on one worm slice.
- Clean withheld-projection relative errors at 120°/90° were 80.72%/86.32% for
  FBP and 14.38%/19.49% for SART. These are development projection diagnostics;
  no true voxel-density error or calibrated uncertainty has been established.
- Completed 63 structured-phantom SART reconstructions. Bias increased as
  coverage shrank while noise variability decreased, demonstrating a limitation
  of variability-only uncertainty. Nine current CPU tests pass.
- Added a small exact Bayesian reference: uncertainty grows with missing views,
  but high coverage can require broad intervals. Twelve current tests pass.
- Passed native parallel-beam Gaussian analytic/gradient checks and completed
  an eight-slice worm pilot: projection discrepancy fell from 38.38% to 6.85%
  after 300 updates. Fifteen current CPU tests pass; no GPU job remains running.
- Completed six matched limited-angle Gaussian baselines and three conditional
  density posteriors. Noisy withheld errors are 13.36%/17.79% (120°/90°);
  posterior predictive coverage is only 30.66%/24.17% for nominal 95% intervals.
  This is a failed-calibration baseline; 21 current CPU tests pass.
- Added shared-scale plots and detailed results to the repository and experiment
  READMEs. Seven CPU tests pass. Retained prior reports and accepted chest results.
- Narrowed local storage to current data and reproducibility evidence; cleanup
  details are recorded separately under `docs/maintenance/`.

## Plan for the next reporting week

1. Validate the completed limited-angle Gaussian/posterior path on known 3D
   targets to separate representation and optimization error from uncertainty.
2. Check sensitivity to solver budget while keeping noise and optimization
   effects distinct.
3. Address conditional-posterior undercoverage through geometry uncertainty or
   an explicit model-error term, using separate calibration cases and evaluating
   interval coverage, width and regional quantitative measurements.
4. Resolve the worm simulator's projection scaling, rotation convention and
   availability of matching deformed density volumes with Anna-Louise.
5. Transfer the validated procedure to development worm slices and plan a
   deformation-separated evaluation. Independent-specimen claims remain out of
   scope for this single-worm dataset.
