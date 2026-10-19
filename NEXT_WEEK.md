# Combined Weeks 4–5 handoff — updated 19 October 2026

## Combined Weeks 4–5 summary (progress through 19 October)

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
- Completed a known-3D control: omitted basis functions leave residual error;
  matched-target fits converge quickly; adding centre uncertainty improves nominal
  95% voxel coverage from 41–45% to 95–97% across 30 paired noise draws. This
  supports a joint centre+density model, without proving worm calibration.
  Twenty-four current CPU tests pass.
- Completed 400 separate-phantom/offset/mismatch fits. Joint voxel coverage is
  89.72–96.72% for matched models but only 8.86–29.29% with a missing component;
  centre uncertainty alone does not account for missing structure. All fits
  terminate successfully; 26 CPU tests pass.
- Restored the omitted component in 200 paired fits: joint voxel coverage
  returns to 90.26–96.32%. Another 120 fits include learned widths and rotations;
  full shape covariance helps but 22 fits hit rotation bounds. The difficult
  case has only 85.47% voxel coverage; boundary and initialization sensitivity
  must be resolved before worm transfer. Thirty CPU tests pass.
- Diagnosed rotation bounds and start-centred priors in 240 fits; a 60-fit
  Cholesky follow-up still exposed local minima. A fixed shared prior and four
  starts with a 300-evaluation budget produced 30/30 passing stability checks
  across another 120 fits. Selected voxel coverage is 95.92–97.36%; individual
  starts can still fail. Thirty-four CPU tests pass.
- On 19 October, replaced truth-informed starts/prior references with acquired-
  data dictionary fitting. Two fresh phantoms and two windows passed 37/40 checks
  initially; 12 fixed extra starts on failed draws brought this to 40/40 across
  172 fits. Voxel coverage is 95.45–96.22%; this remains conditional synthetic
  calibration evidence. Thirty-six CPU tests pass.
- The first measured-only worm shape pilot (16 Gaussians) fails stability:
  0/4 fits converge within budget; withheld error is 25.22% and provisional
  predictive coverage 16.79%. Only five dictionary components start appreciably
  active. This motivates initializer/support and optimization/capacity controls;
  calibrated worm uncertainty has not succeeded. Thirty-eight CPU tests pass.
- Added shared-scale plots and detailed results to the repository and experiment
  READMEs. Retained prior reports and accepted chest results.
- Narrowed local storage to current data and reproducibility evidence; cleanup
  details are recorded separately under `docs/maintenance/`.

## Plan for the next reporting week

1. Validate the audited Cholesky/shared-prior/multistart procedure on fresh fixed
   cases; test nonlinear propagation and retain failed candidates.
2. Improve the worm initializer using acquired-data support or finer/multiple-
   width dictionaries; isolate solver-budget and component-capacity effects.
3. Transfer only as a diagnostic pilot with withheld-view evaluation and explicit
   stability checks; calibrated worm uncertainty remains unestablished.
4. Resolve the worm simulator's projection scaling, rotation convention and
   availability of matching deformed density volumes with Anna-Louise.
5. Transfer the validated procedure to development worm slices and plan a
   deformation-separated evaluation. Independent-specimen claims remain out of
   scope for this single-worm dataset.
