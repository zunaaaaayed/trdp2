# TRDP2 — Limited-angle CT reconstruction generalization

TRDP2 is [Md Zunayedul Islam](https://github.com/zunaaaaayed)'s research project
on Gaussian-based CT reconstruction across limited-angle acquisition configurations
and reconstruction uncertainty. This repository is maintained for TRDP2 research and
experiments; it is not the official R²-Gaussian project or an upstream contribution.

The reconstruction implementation originates from
[R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) by Ruyi Zha and collaborators.
TRDP2 adds acquisition-controlled experiments, reproducibility checks, guarded
execution, evaluation, and research protocols. Original authorship, commit history,
and license notices are retained.

## Research objective

Develop and evaluate uncertainty-aware reconstruction under missing angular
coverage and controlled noise. Settings should transfer across acquisition
orientations, spans and held-out deformations. The current synthetic dataset is
one worm in different shapes; independent-specimen generalization is a longer-term
objective requiring additional data. Each scan has its own reconstruction.

TV tuning and geometry-dependent regularization are historical research directions.
The latter's effectiveness and novelty remain unverified. Completed chest results
are preserved as baseline evidence.

Documentation policy: this README accumulates detailed weekly findings, results
and plots; experiment READMEs own their plans, findings, outputs and interpretations.
[NEXT_WEEK.md](NEXT_WEEK.md) contains only the current weekly summary and next-week plan.

## Current dataset direction

As of 12 October 2026, the dataset supplied by co-supervisor **Ms. Anna-Louise Brun**
is the selected data source for subsequent work. The raw `sino_selected.zip`
archive and its checksum are retained locally; see the
[dataset handoff](docs/SUPERVISOR_DATASET.md) for intake status. The earlier LIDC-IDRI
expansion is deferred. Existing pilot and TV results remain evidence for their
original data convention and do not establish performance on the new dataset.

## Weeks 4–5 combined — worm uncertainty (12–23 October 2026)

Weeks 4 and 5 are reported together because the intervening supervisor meeting
was postponed. Work began on **12 October 2026**. The report must be sent on
**22 October 2026**, ahead of the presentation on **23 October 2026**. This
reporting period remains open; the findings below reflect progress through
**17 October 2026**.

Branch: `experiment/worm-uncertainty`. The current objective is uncertainty under
missing angles and controlled noise. The synthetic parallel-beam TIFFs contain
(z, angle, detector) = (510, 100, 310), with 1 µm sampling and 100 angles over
[0°,180°). Files are deformations of **one worm**, not independent specimens.
Physical projection normalization and matching voxel density targets remain
unresolved; all current calculations use stored projection units.

Completed CPU development work:

- Reproducible Gaussian-noise preparation with hashes, angle masks and overwrite
  protection; immutable source TIFFs.
- Analytic off-centre disk check and one full-angle filtered-backprojection (FBP)
  slice. This validates the scikit-image path, not the Gaussian/TIGRE backend.
- 21 paired FBP runs on worm_004, z=255: spans 180°, 120°, 90°; noise standard
  deviations 0%, 1%, 5% of the development slice RMS (6210.1146 stored units).
  Noisy conditions each use seeds 0, 1, 2, shared across angular masks.
- Seven CPU tests pass, including withheld-view isolation for FBP and SART. Missing views contribute
  zero on the full angle grid, retaining the original angular quadrature weights.

Seed-zero projection relative L2 errors against clean data:

| Span | Noise sigma / development RMS | Measured-view error | Withheld-view error |
|---|---:|---:|---:|
| 180° | 0% | 4.28% | — |
| 120° | 0% | 21.75% | 80.72% |
| 90° | 0% | 27.62% | 86.32% |
| 180° | 1% | 4.44% | — |
| 120° | 1% | 21.78% | 80.72% |
| 90° | 1% | 27.64% | 86.33% |
| 180° | 5% | 7.36% | — |
| 120° | 5% | 22.55% | 80.74% |
| 90° | 5% | 28.24% | 86.34% |

These are projection discrepancies, **not voxel errors or uncertainty calibration**.
The large missing-view errors expose the weakness of zero-filled limited-angle
FBP. Values use the same numerical reprojection operator and one development
slice. The three-realization standard-deviation maps measure noise response only;
they do not capture missing-wedge ambiguity or provide posterior intervals.

Local outputs: `output/worm_uncertainty/paired_fbp_20261017/` contains arrays,
checksummed results and a reconstruction comparison with a shared intensity scale.
See [the detailed protocol](experiments/worm_uncertainty/README.md).

Reproduce from the repository root (choose a fresh output directory):

```bash
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib
python -B -m unittest discover -s experiments/worm_uncertainty -p 'test_*.py' -q
python -B experiments/worm_uncertainty/compare_noise.py \
  --source ../worm_data/worm_004_gt.tiff \
  --output output/worm_uncertainty/paired_fbp_repeat
```

### Reading the scales: what counts as good?

| Plot or score | Meaning of increasing values | How to judge it |
|---|---|---|
| Density / reconstruction brightness | More reconstructed material density | Neither bright nor dark is inherently better; compare structure and values with a valid reference |
| Absolute error, bias, RMSE, relative projection error | Larger discrepancy from reference | Lower is better; zero is ideal. No application-specific acceptable threshold is established |
| Noise standard deviation | Less repeatability across measurement-noise draws | Low values can coexist with systematic error; assess alongside actual error |
| Posterior standard deviation | Less certainty under the assumed model | Neither high nor low alone establishes quality; check calibration and interval width |
| Empirical coverage of nominal 95% intervals | More reference values inside intervals | Near 95% is the target; below suggests intervals miss too often, above can indicate conservatism |
| Interval width | Less precise estimates | Narrower is useful only when coverage remains adequate |

Image brightness is not a pass/fail score. We label units, common display ranges,
and saturation where applicable. Error thresholds need a scientific measurement
tolerance; none has been supplied yet. Foreground scores avoid dilution by empty
background. No figures below demonstrate biological measurement accuracy.

### Reconstruction plots and iterative baseline

![FBP across angular spans and noise strengths](docs/figures/worm_uncertainty/fbp_noise.png)

**FBP:** shrinking angular coverage causes directional streaks and loss of interior
structure; added noise introduces finer fluctuations. All nine panels share one
intensity scale. These are reconstructions, not ground-truth images.

![FBP versus nonnegative SART, fixed central crop](docs/figures/worm_uncertainty/sart_comparison.png)

**Iterative baseline:** SART uses only measured views, a zero initializer, 10 fixed
sweeps, relaxation 0.15, nonnegativity and circular support. The fixed crop
[105:205,105:205] and shared scale make interior differences visible. FBP has no
nonnegativity constraint, so this compares complete baselines, not an isolated
solver change. Neither method uses TV here.

| Span | Noise sigma / RMS | FBP withheld error | SART withheld error |
|---|---:|---:|---:|
| 120° | 0% | 80.72% | 14.38% |
| 120° | 5% | 80.74% | 14.39% |
| 90° | 0% | 86.32% | 19.49% |
| 90° | 5% | 86.34% | 19.48% |

Errors are relative L2 against clean withheld projections, seed zero. SART improves
this development diagnostic substantially; larger missing wedges still leave more
error. Similar clean/noisy scores do not prove noise robustness: only one slice
and one realization were compared. At full coverage, SART measured-view errors
are 3.08% clean and 3.67% at 5% noise, versus FBP 4.28% and 7.36%.
No voxel ground truth or calibrated uncertainty is available. SART and evaluation
Radon discretizations differ. Settings were fixed before this run, with no stopping
or fitting based on withheld values.

Accepted outputs: `output/worm_uncertainty/sart_20261017_v2/`. The superseded
attempt record documents the support warning; v2 enforces circular support after
each sweep. Seven tests pass, including withheld-value isolation for SART.
Figure and input provenance: [plot records](docs/figures/worm_uncertainty/provenance.json).

After the paired FBP command above, reproduce with:

```bash
python -B experiments/worm_uncertainty/compare_sart.py \
  --input output/worm_uncertainty/paired_fbp_repeat \
  --output output/worm_uncertainty/sart_repeat
```

Next: evaluate uncertainty against appropriate references. Resolve simulator scaling and
orientation, group related deformations before freezing splits, and separate
noise sensitivity from reconstruction ambiguity. No GPU training is queued; the bounded slab pilot below has completed.

### Known-target phantom: noise variability versus reconstruction error

A 64×64 structured phantom contains an elliptical body, distinct density regions,
small inclusions and a textured interior patch. We ran 63 fixed-budget SART
reconstructions: three clean controls and 20 noise realizations for each of
180°, 120° and 90° coverage. All masks share the same full-grid noise draws.
Gaussian sigma is 5% of full clean phantom sinogram RMS; seeds are 0–19. The
solver retains 10 sweeps, relaxation 0.15, zero initialization, nonnegativity and
circular support. Known target values are used only to evaluate the reconstructions.

![Known target, reconstruction, bias and noise variability](docs/figures/worm_uncertainty/phantom_reading_guide.png)

Truth and reconstructed means share a density scale; absolute bias and noise
standard deviation share a second scale. Density units are arbitrary. “Bias” here
is the error of the empirical mean over 20 noise draws, not an exact population
expectation. Noise standard deviation uses the sample estimator (ddof=1).

| Coverage | Clean foreground RMSE | Noisy foreground RMSE | Empirical bias RMSE | Mean noise SD | Squared bias / total MSE |
|---|---:|---:|---:|---:|---:|
| 180° | 0.0355 | 0.1119 | 0.0460 | 0.1033 | 16.9% |
| 120° | 0.0879 | 0.1272 | 0.0918 | 0.0889 | 52.1% |
| 90° | 0.1274 | 0.1510 | 0.1280 | 0.0808 | 71.8% |

**Finding:** shrinking coverage increases actual error and empirical bias while
noise variability decreases for this fixed-budget solver. At 90°, squared bias
accounts for 71.8% of foreground MSE; 18.6% of foreground pixels have mean error
larger than twice their noise SD (32.9% at boundaries). That threshold is a
descriptive diagnostic, not a confidence-interval coverage claim. A low-variability
map can therefore miss systematic reconstruction error.

This is one controlled development phantom with matched numerical projection
physics. Boundary/interior/texture masks are reference-derived diagnostics. The
noise trend can depend on the solver and fixed sweep count (fewer views also mean
fewer per-view updates); it is not a universal law. No posterior inference or
calibration has yet been implemented. The empirical MSE decomposition is checked
to numerical precision using population variance over the finite ensemble.

Outputs: `output/worm_uncertainty/phantom_variability_20261017_v2/`; tracked
[metrics and provenance](docs/figures/worm_uncertainty/phantom_variability.json).
Nine current CPU tests pass, including a test that detects constant biased
reconstructions with zero ensemble variance.

```bash
python -B experiments/worm_uncertainty/phantom_variability.py \
  --output output/worm_uncertainty/phantom_variability_repeat
```

Next: use the known-target setup to evaluate an uncertainty model that accounts
for poorly constrained reconstruction directions. Test calibration on separate
phantoms/acquisitions and sensitivity to prior assumptions before worm transfer.

### Exact Bayesian reference: uncertainty and calibration

We added a small **linear Gaussian reference**, using a 24×24 resized development
phantom and an explicit parallel-beam projection matrix. This establishes a
tractable uncertainty baseline before any Gaussian-splatting adaptation. The
model is y = Ax + noise, with known Gaussian noise sigma (5% of full phantom
sinogram RMS), and an independent zero-mean Gaussian prior of standard deviation
tau. The exact posterior is:

`C = inverse(AᵀA / sigma² + I / tau²)`, `mean = C Aᵀy / sigma²`.

The three fixed prior scales (0.3, 1, 3) are sensitivity probes; none is selected
as best from these development results. All geometries share 50 noise draws.
The matrix reproduces the direct projector within 1e-10, and removing views must
increase posterior marginal variance for fixed prior/noise settings.

![Exact posterior reference with interpretation labels](docs/figures/worm_uncertainty/bayes_posterior.png)

**Read this plot:** density brightness measures material value, while darker error
means closer to truth. Higher posterior SD means the model is less certain.
The error and SD columns have separate numeric ranges; do not compare colors
between them. Density panels use [-0.2, 1.2], with out-of-range display values
saturated; metrics retain all values. The independent Gaussian prior allows
negative density and is only a mathematical reference. Shown: prior SD 1 and the
first noise realization. This coarser model is not a controlled accuracy comparison
with the earlier SART experiment.

| Coverage | Foreground RMSE | Mean posterior SD | Nominal 95% empirical coverage | Mean interval width |
|---|---:|---:|---:|---:|
| 180° | 0.1514 | 0.1673 | 96.79% | 0.6557 |
| 120° | 0.2546 | 0.3462 | 98.80% | 1.3569 |
| 90° | 0.3013 | 0.4532 | 99.28% | 1.7764 |

**Finding:** posterior SD grows as angular coverage shrinks. At 90°, the mean
interval width is 1.776 density units, compared with the phantom's designed
0–1.2 density range. High coverage here comes with limited precision; it is not
proof of a useful quantitative reconstruction.

![Coverage versus nominal target](docs/figures/worm_uncertainty/bayes_coverage.png)

**Read this plot:** the dashed 95% line is the nominal target; proximity to it
matters, rather than maximizing coverage. On this fixed structured phantom,
coverage ranges from 95.36% to 99.81% across prior/geometry choices. A separate
model-consistency check draws 200 objects from the assumed prior and their noisy
measurements; coverage is 94.84–95.11%, consistent with the expected model behavior.
That check verifies behavior under the assumed model; it does not establish
calibration on worm data. Pixels are correlated, and these are not independent
subject-level statistics.

This reference captures uncertainty conditional on a simple prior and known noise.
It leaves prior misspecification, physical-model mismatch, positivity and unseen
shape transfer unresolved. Next: freeze a calibration protocol on separate
phantoms and test structural priors and measurement-model mismatch. The worm
simulator's scaling remains an independent question.

Local output: `output/worm_uncertainty/bayes_linear_20261017/`. See
[full metrics and provenance](docs/figures/worm_uncertainty/bayes_reference.json).
Twelve current tests pass, including a scalar posterior check, preservation of
prior variance in an unmeasured direction, and covariance growth after view removal.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/worm_uncertainty/bayes_linear.py \
  --output output/worm_uncertainty/bayes_linear_repeat
```

### Native 3D Gaussian worm slab — first integration pilot

The combined Weeks 4–5 work now includes a native GPU parallel-beam Gaussian
baseline. Analytic checks use three off-centre anisotropic 3D Gaussians at four
angles: projection relative L2 error is at most 0.0213%, voxel-query error is
0.1084%, and a density finite-difference gradient agrees within 0.0021%. Position,
scale, rotation and density gradients are finite. These checks validate the native
adapter against analytic Gaussian integrals, not the supervisor's simulator.

The worm pilot uses eight contiguous slices, z=252:260, all 100 clean angles,
1 µm sampling, 1,500 Gaussians and 300 Adam updates. Initialization samples positive
FBP values; positions, scales, rotations and densities are optimized with fixed
Gaussian count. No TV or posterior uncertainty is used. All fitting and
initialization data are measured full-angle projections.

![Worm Gaussian slab initialization, reconstruction and optimization](docs/figures/worm_uncertainty/gaussian_slab_readable.png)

**Read this plot:** the first two panels share a scale and spatial crop; brightness
represents an uncalibrated reconstruction value, not quality. FBP is an initializer,
not ground truth. Lower measured-view MSE is better data agreement, but the curve
visits different angles and does not measure voxel error. Each displayed slice
is local z=4 (source z=256). The 3D Gaussian model covers all eight slices.

| Full-angle projection relative L2 discrepancy | Value |
|---|---:|
| Initial Gaussian representation | 38.38% |
| After 300 updates | 6.85% |

The reduction verifies a functioning optimization path; it does not establish
convergence, missing-wedge recovery or density accuracy. Every angle was fitted,
so there is no held-out score. Values are normalized by this development slab's
projection RMS (6180.27545), with scene coordinate spacing 2/310. Physical
projection normalization remains unresolved.

The native parallel renderer assumes a square normalized field of view. The
adapter keeps a 310×310 detector canvas and applies loss only to rows 151:159,
which represent the observed slab. Unobserved rows never become artificial zero
measurements. Detector u=x*cos(theta)-y*sin(theta), v=z and symmetric half-pixel
centres are explicit assumptions awaiting simulator confirmation. No inherited
source or CUDA file was changed.

[Native correctness results](docs/figures/worm_uncertainty/native_parallel_checks.json)
and [slab result/provenance](docs/figures/worm_uncertainty/gaussian_slab.json).
Local complete arrays and Gaussian parameters are under
`output/worm_uncertainty/gaussian_slab_20261017/`. The parameter export is not an
exact-resume optimizer checkpoint. Fifteen current CPU tests pass.

On a GPU workstation, after environment activation, use fresh destinations:

```bash
python -B experiments/worm_uncertainty/native_parallel.py \
  --output output/worm_uncertainty/native_parallel_repeat
python -B experiments/worm_uncertainty/fit_gaussian_slab.py \
  --source ../worm_data/worm_004_gt.tiff \
  --output output/worm_uncertainty/gaussian_slab_repeat --execute
```

Next: establish fixed-budget limited-angle slab comparisons with measured-only
initialization and held-out projection evaluation, then validate a conditional
posterior over Gaussian densities on a known 3D phantom. Uncertainty in Gaussian
positions, shapes and model count will initially remain outside that approximation.

### Limited-angle Gaussian baselines and conditional density posterior

The fixed [development protocol](experiments/worm_uncertainty/slab_protocol_v1.json)
was recorded before execution: the same eight-slice slab, 1,500 Gaussians, 1,000
updates, seed zero and no densification. Angular spans are 180°, 120° and 90°,
all starting at 0°. Clean and 5% Gaussian-noise cases share settings; noisy cases
share the same full-grid realization (seed 1701). Initializers and optimization
use measured projections only. Missing views are zero-filled for FBP quadrature,
never supplied as measured loss targets. Final iteration is fixed without
held-out stopping or tuning. Equal update counts imply different numbers of
passes through measured views.

Normalization is frozen from the earlier full-angle development slab (RMS
6180.27545). Consequently, these are development withholding diagnostics, not
blind final-test results or independent-specimen generalization.

| Coverage | Added noise / frozen RMS | Measured-view error | Withheld-view error |
|---|---:|---:|---:|
| 180° | 0% | 4.70% | — |
| 120° | 0% | 4.76% | 13.46% |
| 90° | 0% | 4.49% | 16.83% |
| 180° | 5% | 4.92% | — |
| 120° | 5% | 4.87% | 13.36% |
| 90° | 5% | 4.50% | 17.79% |

Errors are relative L2 against clean projections; **lower is better, zero is ideal**.
The larger missing wedge has higher withheld error despite similar measured-view
agreement. One slab and one noise draw cannot establish robustness. These scores
are not voxel reconstruction errors, and the earlier single-slice SART results
have different sampling and budgets, so they are not a matched method comparison.

#### First conditional posterior: a calibration failure worth preserving

For each noisy fit we freeze positions, scales, rotations and Gaussian count,
then infer all 1,500 density amplitudes. Continuous analytic Gaussian ray
integrals define a linear matrix B. With known normalized noise sigma=0.05 and
a fixed independent zero-mean Gaussian density prior of SD=10:

`C = inverse(BᵀB / sigma² + I / 10²)`

`density_mean = C Bᵀy / sigma²`

Only measured rows contribute to these equations. Voxel and latent projection
variances use the full covariance, `b C bᵀ`; noisy predictive variance adds
`sigma²`. This is an exact linear Gaussian calculation **conditional on plug-in
geometry learned from the same measurements**. It omits geometry/model-selection
uncertainty. Analytic projections differ from the native rasterizer by at most
0.0875% on these fitted models (small truncation differences). The Gaussian prior
allows signed amplitudes: 28.3–30.4% of posterior mean amplitudes are negative;
no clipping is applied. It is an experimental reference, not a physically
constrained final density model.

![Limited-angle Gaussian reconstruction and conditional SD](docs/figures/worm_uncertainty/conditional_density_v1/slab_comparison.png)

**Read this plot:** the first three columns share density scales and the same
central slice/crop. Brightness indicates density, not correctness. The SD column
has its own scale; larger values mean less certainty under the conditional model.
Low SD is misleading when systematic error is omitted. There is no worm voxel
ground truth against which to validate these density maps.

| Coverage, 5% noise | Native withheld error | Posterior-mean withheld error | Latent 95% coverage | Predictive 95% coverage | Predictive interval width |
|---|---:|---:|---:|---:|---:|
| 120° | 13.36% | 13.23% | 7.90% | 30.66% | 0.2028 |
| 90° | 17.79% | 17.61% | 8.92% | 24.17% | 0.2132 |

Coverage is evaluated on withheld foreground rays where the clean normalized
projection exceeds 0.05, a fixed evaluation-only threshold. Latent intervals are
checked against clean projections; predictive intervals are checked against the
noisy withheld measurements. Width is in normalized projection units. The target
is approximately 95% coverage with useful widths, not maximum coverage or minimum
width alone. This is empirical coverage on one correlated slab, not a repeated
independent-subject calibration study.

![Coverage and width reveal overconfidence](docs/figures/worm_uncertainty/conditional_density_v1/coverage.png)

**Finding:** the posterior mean slightly improves withheld error, but nominal 95%
predictive coverage is only 30.66% at 120° and 24.17% at 90°. Latent coverage is
7.90% and 8.92%. The intervals are severely overconfident. Even full-angle measured
foreground latent coverage is only 13.78%, indicating unresolved model error
already before missing-wedge extrapolation. We do not tune intervals on these
withheld results or claim successful uncertainty calibration.

![Example held-out ray profile and predictive interval](docs/figures/worm_uncertainty/conditional_density_v1/withheld_profile.png)

The fixed 144° view, source z=256, is withheld from both limited-angle fits.
A useful mean should follow the clean reference, and predictive bands should
capture future noisy measurements at their nominal rate. Visible misses show
errors not represented by the density-only posterior. Their causes may include
representation capacity, incomplete optimization, plug-in geometry, prior choice
and unresolved simulator conventions; this experiment does not isolate them.

Results and figure provenance:
[record](docs/figures/worm_uncertainty/conditional_density_v1/results.json).
Local outputs: `output/worm_uncertainty/limited_gaussian_v1/` and
`output/worm_uncertainty/conditional_density_v1/`. Twenty-one current tests pass,
including measured-only initialization, rotated analytic ray quadrature,
posterior algebra and latent-versus-predictive interval checks.

Reproduce after the environment activation above, using fresh destinations:

```bash
python -B experiments/worm_uncertainty/limited_gaussian.py \
  --source ../worm_data/worm_004_gt.tiff \
  --output output/worm_uncertainty/limited_gaussian_repeat --execute
python -B experiments/worm_uncertainty/conditional_density.py \
  --input output/worm_uncertainty/limited_gaussian_repeat \
  --output output/worm_uncertainty/conditional_density_repeat --execute
python -B experiments/worm_uncertainty/plot_conditional_density.py \
  --baseline output/worm_uncertainty/limited_gaussian_repeat \
  --posterior output/worm_uncertainty/conditional_density_repeat \
  --output output/worm_uncertainty/conditional_figures_repeat
```

**Next:** use known 3D targets to separate representation/optimization error from
conditional uncertainty, then test geometry uncertainty or an explicit model-error
term on separate calibration cases. Keep these failed-calibration results as a
baseline. Fix protocol choices before evaluation on new deformations or angles.

### Known 3D phantom: separating three sources of error

A controlled CPU experiment now separates fixed-basis representation error,
optimization stopping, and omitted **Gaussian-centre uncertainty**. Here geometry
means the reconstructed Gaussian positions, **not uncertain scanner angles or
alignment**. Acquisition geometry is exact in this experiment.

The frozen `phantom_3d_protocol_v1.json` specifies three anisotropic 3D Gaussians,
36 parallel-beam views at 5° spacing, a 12×24 detector, and a 24³ evaluation grid.
Full/120°/90° acquisitions use 36/24/18 views. Thirty paired Gaussian-noise draws
have SD 5% of the full clean projection RMS; priors and initialization are fixed
before evaluation. These are synthetic density units, not calibrated worm units.
Foreground means truth above 5% of its peak, used only for evaluation.

**Representation control.** Add a fourth component to the truth, then solve
all-view clean least squares exactly with the original three fixed basis
functions. Projection relative error remains **11.283%**, with foreground voxel
RMSE **0.03369**. Restoring the fourth basis function reduces both to numerical
precision. This demonstrates an omitted-component error that further density
optimization cannot fix. It does not measure the best possible fit of three
freely moving/resizing Gaussians.

**Optimization control.** On the matched three-component target, identical starts
and priors are fitted with budgets of 5 versus 150 function evaluations. All long
fits stop successfully after 7 evaluations. Foreground voxel RMSE changes:

| Angular span | Budget 5 (stopped early) | Budget 150 (converged) |
|---|---:|---:|
| 180° | 1.003e-6 | 1.342e-7 |
| 120° | 5.789e-7 | 2.182e-7 |
| 90° | 6.422e-7 | 3.946e-7 |

Lower RMSE is better. Both budgets already yield tiny errors here; this is an easy
matched-model problem. Successful convergence and tiny residuals on this phantom
do not establish convergence of the 1,500-Gaussian worm fit. Remaining clean error
also includes the fixed prior's influence.

**Uncertainty control.** With noisy matched targets, compare an exact density
posterior at true centres, a conditional density posterior at fitted centres,
and a local joint centre+density covariance. The latter two use exactly the same
reconstruction mean; only uncertainty propagation changes. The joint calculation
retains cross-covariances and uses a Gauss–Newton/Laplace approximation and linear
voxel propagation, with fixed widths, rotations and component count.

| Span | Known-centre coverage | Fitted-centre density-only coverage | Joint coverage | Density-only / joint width |
|---|---:|---:|---:|---:|
| 180° | 96.91% | 44.85% | 95.25% | 0.000635 / 0.001780 |
| 120° | 93.96% | 43.41% | 95.05% | 0.000784 / 0.002325 |
| 90° | 96.68% | 40.98% | 96.63% | 0.000921 / 0.003115 |

Coverage is the mean across 30 draws of the foreground fraction inside nominal
95% **latent voxel-density** intervals. Width is mean interval width in synthetic
density units. These differ from the worm experiment's withheld-ray predictive
intervals. Fitted-centre mean voxel RMSE is 0.000502/0.000674/0.000847. All 90 noisy
fits converge without active bounds; conditional density re-solving changes MAP
densities by at most 2.13e-13.

![Coverage must be judged with interval width](docs/figures/worm_uncertainty/known_3d_diagnosis_v1/coverage_width.png)

The target is coverage near 95% with useful width. Density-only intervals are too
narrow after centre fitting; carrying centre uncertainty substantially repairs
coverage in this controlled setting. Coverage varies across individual draws and
voxels are correlated, so this table is not 30×foreground independent trials.

![Known truth, fitted density, error and uncertainty](docs/figures/worm_uncertainty/known_3d_diagnosis_v1/slices.png)

The figure fixes the first noise draw, 90° acquisition and central z slice. Truth
and reconstruction share density limits; both SD panels share limits. Error low
is good; SD low alone is not. Results and provenance are in the
[record](docs/figures/worm_uncertainty/known_3d_diagnosis_v1/results.json).

**Decision for the next uncertainty model:** test a joint centre-and-density local
posterior next, then add widths/rotations under controlled perturbations. Keep the
failed worm density-only baseline. Before transfer, evaluate on separately fixed
phantoms, starts and angular offsets, with withheld-ray coverage as well as voxel
coverage and widths. Include deliberate missing components to test when a
model-error term or richer representation is needed. Scanner-angle/alignment
uncertainty remains a separate, untested question. This matched analytic
synthesis/inference study is a diagnostic, not independent forward-model
validation, proof of worm calibration, or evidence of generalization across
independent volumes. No interval tuning used these reported outcomes.

## Week 3 — ordinary-TV controls (3–9 October 2026)

Two development baselines completed 30,000 iterations with 50 views, a 120° span,
TV weight 0.05, seed zero, and the same chest reference and evaluation projections:

| Acquisition start | Volume PSNR | Slice-averaged SSIM | RMSE | Boundary MAE |
|---|---:|---:|---:|---:|
| 0° | 30.913 dB | 0.9010 | 0.02847 | 0.03880 |
| 90° | 28.591 dB | 0.8624 | 0.03719 | 0.05654 |

These reproduce the completed pilot and demonstrate orientation sensitivity on
**one development volume**. They do not establish independent-volume generalization.
Metrics use the supplied unit-range reference without clipping predicted values.
See the protocol for the SSIM and boundary definitions.

As of 8 October 2026, all four ordinary-TV controls (weights 0.025 and 0.1 at
both orientations) have completed. **TV 0.05 remains the frozen reference** under
the predeclared selection rule. A six-subject LIDC-IDRI candidate cohort is recorded;
the first development series has been downloaded and passed its DICOM header audit.
Pixel preprocessing and independence from the pilot remain unverified. Reserved
acquisitions are untouched and no GPU training is queued. See the
[independent-volume preparation](experiments/independent_volume_evaluation/README.md)
for results, evidence and remaining gates.

### Complete TV comparison

| Start | TV | PSNR (dB) | SSIM | Boundary MAE |
|---|---:|---:|---:|---:|
| 0° | 0.025 | 30.766 | 0.8986 | 0.03833 |
| 0° | 0.05 | 30.913 | 0.9010 | 0.03880 |
| 0° | 0.1 | 30.782 | 0.8989 | 0.04060 |
| 90° | 0.025 | 28.538 | 0.8602 | 0.05611 |
| 90° | 0.05 | 28.591 | 0.8624 | 0.05654 |
| 90° | 0.1 | 28.448 | 0.8602 | 0.05823 |

[Week 3 TV tradeoff plot (PDF)](Reports/Week3/figures/tv_tradeoff.pdf)

Lower TV slightly reduces boundary errors but lowers SSIM; higher TV worsens both.
The recorded selection rule retained 0.05. The 2.32 dB orientation gap persists,
with one volume and one seed. No population-transfer claim follows. See the
[complete Week 3 report](Reports/Week3/README.md), including convergence and slice
plots. The LIDC-IDRI expansion was prepared but deferred before reconstruction.

## Week 2 — baseline to limited angle (26 September–2 October 2026)

Reproduced the original 50-view chest cone-beam pipeline, then regenerated noiseless
full-circle and 120° acquisitions with the same projection count, reference and
shared evaluation views. A rotated 120° acquisition exposed orientation sensitivity.
Each acquisition uses its own measurement-derived initializer, so the effect includes
initialization changes. Holding view count fixed changes angular spacing as well
as coverage; this was a feasibility experiment on one reference, not a cohort study.

Established the CUDA 11.8 research environment and addressed TIGRE context-reset
failures. The inherited backprojection patch preserved the saved initializer
exactly; later forward projection checks required isolated processes. Original
runs and their provenance remain available. The subsequent Week 3 benchmark uses
a different disjoint evaluation grid, so its projection scores must not be pooled
with the historical pilot.

See the [Week 2 report](Reports/TRDP2_Week1.pdf) (original filename retained),
[baseline records](experiments/baseline_chest_50views/README.md) and
[pilot protocol](experiments/limited_angle_120/README.md) for the recorded results.
No earlier weekly report is reconstructed from unverified numbers.

## Project guide

Start with the [next-week handoff](NEXT_WEEK.md) for current priorities and
[repository map](docs/REPOSITORY_MAP.md) for active versus historical material.

- [Historical workstation-transfer snapshot](experiments/ordinary_tv_controls/HANDOFF.md)
- [Generalization protocol and split rules](experiments/limited_angle_generalization/protocol.md)
- [Benchmark preparation and verification](experiments/limited_angle_generalization/README.md)
- [Guarded single-case workflow](experiments/limited_angle_generalization/workflow.md)
- [Ordinary-TV controls and selection rule](experiments/ordinary_tv_controls/README.md)
- [Independent-volume data research and intake plan](experiments/ordinary_tv_controls/independent_volumes.md)
- [Saved results and transfer inventory](experiments/ordinary_tv_controls/session_20261007/)
- [Historical limited-angle pilot](experiments/limited_angle_120/README.md)
- [Initial baseline reproduction](experiments/baseline_chest_50views/README.md)
- [Research handoff specification](TRDP2_CODEX_HANDOFF.md)

## Code readability before execution

The active worm Python scripts use consistent formatting and basic static checks.
See [the code style guide](docs/CODE_STYLE.md) for setup, previewing changes and
formatting on save. With the isolated formatting environment installed:

```bash
make format-diff PYTHON=.venv-style/bin/python  # preview formatting
make format PYTHON=.venv-style/bin/python       # apply formatting
make check PYTHON=.venv-style/bin/python        # check before experiments
```

GitHub also checks this style on relevant pushes and pull requests. These CPU-only
checks do not execute experiments; scientific correctness remains covered by the
experiment tests and diagnostics. Historical result hashes retain their original
source provenance after formatting.

## Setup and safe continuation

```bash
git clone --recursive https://github.com/zunaaaaayed/trdp2.git
cd trdp2
```

The verified university stack used Python 3.9.23, PyTorch 2.1.2+cu118, CUDA 11.8,
NumPy 1.24.4, Open3D 0.18, and an RTX 4000 Ada GPU. The local Conda environment is
named `trdp2-r2`. Native extensions and TIGRE must be checked on a new computer;
copied compiled binaries are not assumed portable. Follow the
[current handoff](NEXT_WEEK.md) before GPU execution.
The Python package remains named `r2_gaussian` to preserve the verified baseline.

After activating the research environment:

```bash
python -B -m unittest discover -s experiments/limited_angle_generalization/tests -v
python -B -m unittest discover -s experiments/ordinary_tv_controls -p 'test_*.py' -v
python -B -m unittest discover -s experiments/independent_volume_evaluation -p 'test_*.py' -v
```

Historical guarded GPU runners require an explicit `--execute`; the current CPU
diagnostic scripts run directly. Existing destinations are rejected;
exact checkpoint continuation is not verified. TIGRE forward projection is
isolated from live PyTorch CUDA tensors because its context reset caused a
segmentation fault in the original combined smoke process.

Data, checkpoints, logs, and compiled extensions are excluded from Git. Transfer
the current data and artifacts as well as this repository. The obsolete
7 October transfer archive is eligible for removal only after every member is
verified against the retained files; see the [cleanup record](docs/maintenance/cleanup_20261017.json).
Frozen source/data hashes must remain valid. Historical records
retain their original repository name and workstation paths for provenance.

Original installation options, data-generation instructions, and CLI reference
are preserved in [README_UPSTREAM.md](README_UPSTREAM.md); these are inherited
upstream documentation, not claims about TRDP2 results.

## Attribution and license

TRDP2 research extensions are maintained by Md Zunayedul Islam. The original
R²-Gaussian implementation and its dependencies retain their authorship and
applicable terms. See [LICENSE.md](LICENSE.md) and the license files in each
submodule. Renaming this project does not replace those licenses or transfer
ownership of the original implementation.

The inherited implementation builds on
[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF),
[NAF](https://github.com/Ruyi-Zha/naf_cbct), and
[TIGRE](https://github.com/CERN/TIGRE).

When using the original reconstruction method, cite its paper:

```bibtex
@inproceedings{r2_gaussian,
  title={R$^2$-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction},
  author={Ruyi Zha and Tao Jun Lin and Yuanhao Cai and Jiwen Cao and Yanhao Zhang and Hongdong Li},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```

For TRDP2 experiments, additionally reference this repository and the exact commit
and protocol used. No separate TRDP2 method publication is claimed here.
