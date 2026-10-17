# Worm uncertainty development

Combined Weeks 4–5 reporting period; development scripts started 17 October 2026. Scope: uncertainty under missing angles and controlled
additive noise, with held-out deformations of the same synthetic worm. No claim
of independent-specimen generalization. Ordinary TV remains a comparator, not
the new research objective. Old chest experiments and frozen weights are retained.

Supervisor clarification: TIFF axes are (z=510, angle=100, detector=310), parallel
beam, noiseless, 1 micrometre axial/detector sampling. We interpret 100 angles at
1.8-degree steps as [0,180), ending at 178.2 degrees. No flips were reported;
rotation centre and projector conventions still need validation.

Reported density units are mg/cm^3; underlying density converts to g/cm^3 by
1/1000. Whether stored projections are sums, averages or physical integrals is
unresolved. Do not apply that conversion to TIFFs without establishing the
projection convention. Request simulation JSON/operator convention and matching
voxelized density references. Full-angle reconstructions are not exact ground truth.

## Preparation contract

`prepare.py` reads a single axial sinogram without editing source files. It saves
clean and noisy full-angle arrays plus a measured-view mask. Reconstruction must
use only masked noisy rows. Clean withheld rows are evaluation-only; they must
never enter initialization, fitting, early stopping or hyperparameter selection
on final tests. Full and limited angle variants with the same source/z/scale/noise
seed share exactly the same noise realization. No clipping or normalization occurs.
The output directory must be new. Failed partial outputs must be retained or
explicitly archived; they are not successful preparations without a manifest.

Gaussian noise here is a synthetic stress test, not a calibrated scanner model.
Specify a fixed signal scale derived from development data, document its source,
and reuse it across deformations and angular spans. Do not compute per-test-file
scales. Suggested pilot fractions: 0, 0.01, 0.05; these are provisional, not a frozen
benchmark. Keep noise seeds distinct from optimizer seeds. Poisson count models
are deferred until measurement physics is specified.

## Plan and remaining questions

The preparation, analytic disk check, FBP comparison and SART comparator below
are complete. The structured phantom comparison is now complete (results below). Next assess
a principled uncertainty estimator against the known targets. Freeze settings before evaluation on held-out
deformations; related deformations must be grouped. Resolve simulator projection
normalization and obtain matching density volumes before physical-density claims.

Tests (research environment):

```bash
python -B -m unittest discover -s experiments/worm_uncertainty -p 'test_*.py' -v
```

No GPU job or training queue is launched by this preparation layer.

## CPU FBP diagnostic — 17 October 2026

`check_fbp.py` validates scikit-image against continuous off-centre disk chord
lengths, then reconstructs worm_004 slice 255 with all 100 clean views. This is a
separate classical CPU baseline, not validation of the Gaussian/TIGRE backend.
Disk forward relative L2 error: 0.01876; thresholded disk centroid error: 0.03014
pixels; interior density: 1.00035 for expected 1; linearity error: 4.44e-15.
The first centroid diagnostic used all positive FBP pixels and failed (0.8003
pixels), because weak positive ringing biased that statistic. The reported
geometry diagnostic instead uses the support above half the known disk density;
the acceptance tolerance was not relaxed.

Worm reprojection relative L2 residual: 0.04282, RMSE 265.91 stored units.
This is an in-sample consistency diagnostic with the same numerical projector,
not a voxel error or independent physical calibration. Reconstruction spans
-101.44 to 637.13 stored projection units per pixel; negative ringing is retained.
Assumed centre is detector index 155 with circular support. These assumptions
and angle handedness are not independently verified against the simulator.
Outputs: `output/worm_uncertainty/fbp_smoke_20261017/` (arrays, report, figure).

Reproduce with a fresh output directory:

```bash
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib
python -B experiments/worm_uncertainty/check_fbp.py \
  --source ../worm_data/worm_004_gt.tiff \
  --output output/worm_uncertainty/fbp_smoke_repeat
```

The paired coverage/noise comparison is completed below. Density-unit and
uncertainty-calibration claims await the appropriate references.

## Paired FBP pilot completed

`compare_noise.py` completed 21 reconstructions on worm_004 z=255: full/120/90
degree spans with clean data and 1%/5% Gaussian noise, three noisy realizations.
Noise realizations are shared across spans. Missing projections are zero-filled
on the full angle grid to preserve angular quadrature rather than renormalizing
the limited arc to pi. Six tests pass, including withheld-view isolation.

Outputs: `output/worm_uncertainty/paired_fbp_20261017/`. The main repository
README records the seed-zero metrics and exact reproduction commands. Saved
standard deviations are noise-response diagnostics, not posterior uncertainty.
The RMS scale uses the full clean development slice and must not be re-estimated
from withheld angles of a final evaluation case. No clipping is performed.

## Iterative comparator and README figures

`compare_sart.py` adds a fixed 10-sweep nonnegative SART comparator with circular
support. It verifies the paired FBP input checksum and uses measured noisy views
only. No withheld-data stopping or tuning. Six seed-zero cases completed; accepted
outputs are `output/worm_uncertainty/sart_20261017_v2/`. The earlier attempt is
retained with its support warning. The root README embeds shared-scale figures,
records measured/withheld projection results, and provides reproduction commands.
Small figure copies and provenance live in `docs/figures/worm_uncertainty/`.
No posterior uncertainty or quantitative density accuracy claim is supported.

## Findings and interpretations

All current worm results use z=255 of worm_004, angles starting at 0°. Noise
standard deviation is 0%, 1% or 5% of the full clean development slice RMS
(6210.1146 stored units). This provisional scale is reused across acquisition masks.
FBP has 21 runs (three seeds for noisy conditions); SART has six runs (seed zero,
clean and 5% noise). SART uses 10 fixed sweeps, relaxation 0.15, zero initialization,
nonnegativity and circular support after each sweep. It uses no TV.

| Coverage | Noise / RMS | FBP measured error | FBP withheld error | SART measured error | SART withheld error |
|---|---:|---:|---:|---:|---:|
| 180° | 0% | 4.28% | — | 3.08% | — |
| 180° | 5% | 7.36% | — | 3.67% | — |
| 120° | 0% | 21.75% | 80.72% | 2.84% | 14.38% |
| 120° | 5% | 22.55% | 80.74% | 3.48% | 14.39% |
| 90° | 0% | 27.62% | 86.32% | 2.90% | 19.49% |
| 90° | 5% | 28.24% | 86.34% | 3.52% | 19.48% |

All errors are relative L2 against clean projections, seed zero. Values use the
same Radon evaluation operator; SART's internal discretization differs. This is
projection consistency, not true density accuracy. SART improves the missing-view
diagnostic, while larger wedges leave more error. Nonnegativity contributes to
the difference; this is not an isolated solver ablation. Similar noisy/clean
held-out errors do not establish noise robustness from one realization.

![Paired FBP comparison](../../docs/figures/worm_uncertainty/fbp_noise.png)

Shared-scale FBP panels show directional artifacts as coverage shrinks and finer
noise texture as noise increases. Missing-wedge effects dominate this baseline.

![FBP and SART comparison](../../docs/figures/worm_uncertainty/sart_comparison.png)

The same central crop and intensity range reveal reduced streaks with SART,
with persistent distortion at 90°. There is no matching voxel ground truth.
Repeated-noise standard deviations measure noise sensitivity, not posterior
uncertainty or the full ambiguity of missing measurements.

## Outputs and reproduction

| Local directory under `output/worm_uncertainty/` | Contents / status |
|---|---|
| `preparation_smoke_20261017/` | Clean/noisy slice, masks, input/output hashes |
| `fbp_smoke_20261017/` | Disk checks, full-angle FBP arrays and diagnostic |
| `paired_fbp_20261017/` | 21 FBP results, noise-response maps, figure and metrics |
| `sart_20261017_v2/` | Six accepted SART results, figure and metrics |

Small figures and result snapshots are in
[`docs/figures/worm_uncertainty/`](../../docs/figures/worm_uncertainty/provenance.json).
Raw TIFFs remain in `../worm_data/` relative to repository root. The superseded
SART attempt's report is retained as a diagnostic record; it is excluded from results.

From repository root, choose fresh output destinations:

```bash
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib
python -B -m unittest discover -s experiments/worm_uncertainty -p 'test_*.py' -q
python -B experiments/worm_uncertainty/compare_noise.py \
  --source ../worm_data/worm_004_gt.tiff \
  --output output/worm_uncertainty/paired_fbp_repeat
python -B experiments/worm_uncertainty/compare_sart.py \
  --input output/worm_uncertainty/paired_fbp_repeat \
  --output output/worm_uncertainty/sart_repeat
```

Seven tests verify deterministic noise, angle masks, no clipping, source-array
preservation, and withheld-value isolation for both reconstruction paths. The
analytic disk also checks centroid, scale and linearity. These checks do not
verify the supervisor's simulator or a probabilistic uncertainty model.

## Known-target phantom: noise variability versus reconstruction error

A 64×64 structured phantom contains an elliptical body, distinct density regions,
small inclusions and a textured interior patch. We ran 63 fixed-budget SART
reconstructions: three clean controls and 20 noise realizations for each of
180°, 120° and 90° coverage. All masks share the same full-grid noise draws.
Gaussian sigma is 5% of full clean phantom sinogram RMS; seeds are 0–19. The
solver retains 10 sweeps, relaxation 0.15, zero initialization, nonnegativity and
circular support. Known target values are used only to evaluate the reconstructions.

![Known target, reconstruction, bias and noise variability](../../docs/figures/worm_uncertainty/phantom_reading_guide.png)

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
[metrics and provenance](../../docs/figures/worm_uncertainty/phantom_variability.json).
Nine current CPU tests pass, including a test that detects constant biased
reconstructions with zero ensemble variance.

```bash
python -B experiments/worm_uncertainty/phantom_variability.py \
  --output output/worm_uncertainty/phantom_variability_repeat
```

Next: use the known-target setup to evaluate an uncertainty model that accounts
for poorly constrained reconstruction directions. Test calibration on separate
phantoms/acquisitions and sensitivity to prior assumptions before worm transfer.

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

![Exact posterior reference with interpretation labels](../../docs/figures/worm_uncertainty/bayes_posterior.png)

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

![Coverage versus nominal target](../../docs/figures/worm_uncertainty/bayes_coverage.png)

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
[full metrics and provenance](../../docs/figures/worm_uncertainty/bayes_reference.json).
Twelve current tests pass, including a scalar posterior check, preservation of
prior variance in an unmeasured direction, and covariance growth after view removal.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/worm_uncertainty/bayes_linear.py \
  --output output/worm_uncertainty/bayes_linear_repeat
```

## Native 3D Gaussian worm slab — first integration pilot

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

![Worm Gaussian slab initialization, reconstruction and optimization](../../docs/figures/worm_uncertainty/gaussian_slab_readable.png)

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

[Native correctness results](../../docs/figures/worm_uncertainty/native_parallel_checks.json)
and [slab result/provenance](../../docs/figures/worm_uncertainty/gaussian_slab.json).
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

## Limited-angle Gaussian baselines and conditional density posterior

The fixed [development protocol](slab_protocol_v1.json)
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

![Limited-angle Gaussian reconstruction and conditional SD](../../docs/figures/worm_uncertainty/conditional_density_v1/slab_comparison.png)

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

![Coverage and width reveal overconfidence](../../docs/figures/worm_uncertainty/conditional_density_v1/coverage.png)

**Finding:** the posterior mean slightly improves withheld error, but nominal 95%
predictive coverage is only 30.66% at 120° and 24.17% at 90°. Latent coverage is
7.90% and 8.92%. The intervals are severely overconfident. Even full-angle measured
foreground latent coverage is only 13.78%, indicating unresolved model error
already before missing-wedge extrapolation. We do not tune intervals on these
withheld results or claim successful uncertainty calibration.

![Example held-out ray profile and predictive interval](../../docs/figures/worm_uncertainty/conditional_density_v1/withheld_profile.png)

The fixed 144° view, source z=256, is withheld from both limited-angle fits.
A useful mean should follow the clean reference, and predictive bands should
capture future noisy measurements at their nominal rate. Visible misses show
errors not represented by the density-only posterior. Their causes may include
representation capacity, incomplete optimization, plug-in geometry, prior choice
and unresolved simulator conventions; this experiment does not isolate them.

Results and figure provenance:
[record](../../docs/figures/worm_uncertainty/conditional_density_v1/results.json).
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

![Coverage must be judged with interval width](../../docs/figures/worm_uncertainty/known_3d_diagnosis_v1/coverage_width.png)

The target is coverage near 95% with useful width. Density-only intervals are too
narrow after centre fitting; carrying centre uncertainty substantially repairs
coverage in this controlled setting. Coverage varies across individual draws and
voxels are correlated, so this table is not 30×foreground independent trials.

![Known truth, fitted density, error and uncertainty](../../docs/figures/worm_uncertainty/known_3d_diagnosis_v1/slices.png)

The figure fixes the first noise draw, 90° acquisition and central z slice. Truth
and reconstruction share density limits; both SD panels share limits. Error low
is good; SD low alone is not. Results and provenance are in the
[record](../../docs/figures/worm_uncertainty/known_3d_diagnosis_v1/results.json).

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

Reproduce this CPU study after activating `trdp2-r2` (fresh destinations):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/worm_uncertainty/diagnose_3d.py \
  --output output/worm_uncertainty/known_3d_diagnosis_repeat
MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib python -B experiments/worm_uncertainty/plot_diagnose_3d.py \
  --input output/worm_uncertainty/known_3d_diagnosis_repeat \
  --output output/worm_uncertainty/known_3d_figures_repeat
python -B -m unittest discover -s experiments/worm_uncertainty -p 'test_*.py' -q
```

Twenty-four current CPU tests pass, including finite-difference checks of ray and
voxel centre derivatives and an exact representable-target density solve. The
experiment also checks covariance ordering and artifact integrity. GPU training
is not required for this diagnostic.

## Code readability and pre-run checks

All active Python files follow the shared [code style guide](../../docs/CODE_STYLE.md).
Run `make check PYTHON=.venv-style/bin/python` from the repository root before an
experiment; use `make format-diff` to preview and `make format` to apply formatting
with the same `PYTHON` argument. These checks do not launch GPU work. Formatting
preserves historical output reports and their original source hashes.
