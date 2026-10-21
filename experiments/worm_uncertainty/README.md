# Worm uncertainty development

Combined Weeks 4–5 work is **closed as of 21 October 2026**. Work began
12 October; these development scripts started 17 October. The report is due
22 October and the presentation is on 23 October. Scope: uncertainty under
missing angles and controlled additive noise, with held-out deformations of the
same synthetic worm as a planned evaluation target. No claim
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

Weeks 4–5 completed preparation, analytic projector checks, FBP/SART comparisons,
Gaussian slab baselines, density/geometry uncertainty controls on known phantoms,
and the final worm initialization/prior/budget/capacity study. The final study
completed all 16 fits, but all four settings failed stability. Only two starts
reported solver success. At 450 evaluations, the 16-component acquired/withheld
errors are 11.20%/19.84%; 32 components give 8.37%/23.81%. These are provisional
projection diagnostics; calibrated worm uncertainty has not been demonstrated.
The final test count is 42. Dated sections below preserve earlier intermediate
findings and plans; the final controls section records the latest evidence.

Next week, freeze a staged density → centre → full-shape fitting comparison with
a fixed shared prior. After stable optimization, evaluate reserved angular
configurations and deformations, and assess coverage with interval width on known
targets. Related deformations must be grouped; no independent-specimen claim
follows from this dataset. Simulator projection normalization and matching
voxelized deformed density targets remain unresolved. Physical-density accuracy
and representation error cannot yet be established from projection fits alone.

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

### Separate phantoms, angular offsets and missing structure

The next frozen CPU stress test uses two new manually specified 3D phantoms
(`separated` and `overlapping`), each with three anisotropic Gaussian components.
The acquisition windows are span/start = 180°/0°, 120°/0°, 120°/60°, 90°/0° and
90°/120°. The last wraps across 180°, selecting [120°,180°) and [0°,30°).
Each condition has ten paired noise draws and two initializations: **400 fits**
across 20 cases. These new synthetic targets are development stress tests, not
independent biological subjects or a blinded validation set.

The protocol, `phantom_stress_protocol_v1.json`, was fixed before execution.
Noise is 5% of each phantom/condition's full clean projection RMS. All angular
windows share the same noise realization within a draw. The priors, 150-evaluation
budget, density bounds, initialization perturbation and fixed widths are inherited
from the earlier control. Initialization centres are perturbed from truth and
widths/component identities are known: this remains an intentionally favourable
inverse problem. Acquisition geometry is exact.

For the mismatch condition, a fourth component is added to the truth while the
fitted model retains three movable components with fixed widths. The density-only
and joint centre+density approximations share the same fitted mean. Both use only
acquired measurements; withheld data and truth-based foreground masks are used
for evaluation. The foreground threshold is 5% of the respective truth peak.

| Condition | Density-only voxel coverage | Joint voxel coverage | Joint withheld latent coverage | Joint withheld predictive coverage |
|---|---:|---:|---:|---:|
| Matched models | 30.46–45.36% | 89.72–96.72% | 89.16–96.82% | 94.95–95.43% |
| Missing component | 1.99–11.40% | 8.86–29.29% | 9.05–22.58% | 43.87–65.32% |

Ranges are across case means, not confidence intervals. Each mean averages ten
noise draws and two starts; starts share those draws and are not independent
noise replicates. Coverage targets 95%. Latent intervals are checked against
clean voxel densities or clean withheld projections. Predictive intervals add
known measurement-noise variance and are checked against noisy withheld rays.
Full-angle cases have no withheld views.

![Coverage, width and error across phantoms and acquisition windows](../../docs/figures/worm_uncertainty/phantom_stress_v1/coverage_error.png)

Coverage should be near 95%, not merely high. Narrow intervals are useful only
when they cover the target reliably. Voxel RMSE should be low. Widths and errors
use synthetic density units, not calibrated worm units. Joint voxel interval
widths remain similar under missing-component error, while RMSE rises from
0.00056–0.00156 (matched) to 0.03977–0.04758 (mismatch): the covariance does not
capture the missing structure.

Angular orientation also matters. For the matched separated phantom at 90°,
changing the start from 0° to 120° raises voxel RMSE from 0.00100 to 0.00115 and
lowers joint voxel coverage from 92.73% to 89.72%, although predictive coverage
stays near 95%. Added measurement noise can therefore make predictive coverage
look satisfactory while latent density intervals still under-cover.

All 400 solver runs report successful termination within 15 function evaluations,
with no active parameter bounds. This reduces concern about a simple iteration
budget cutoff in these cases; it does not prove global optimization or remove
model error. Per-draw convergence, optimality, errors and interval metrics are
retained in `output/worm_uncertainty/phantom_stress_v1/draws.json`; the
[tracked result record](../../docs/figures/worm_uncertainty/phantom_stress_v1/results.json) contains all case
summaries, the frozen protocol, source hashes and the draw-file checksum.

**Decision:** joint centre+density uncertainty remains a useful baseline for
matched representations. It is insufficient for missing structure. Before worm
transfer, test a representation-repair control that restores the fourth component,
then test uncertain widths/rotations and an explicit residual model if systematic
error remains. A residual model must not silently absorb incorrect density or
scanner conventions. Fix those comparisons before evaluation; do not inflate
intervals to fit these reported outcomes. Ten draws and two selected phantoms
leave uncertainty about calibration across shapes and angular configurations.
These results provide no independent-volume or cone-beam generalization claim.

Reproduce after activating `trdp2-r2`, using fresh destinations:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/worm_uncertainty/stress_3d.py \
  --output output/worm_uncertainty/phantom_stress_repeat
MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib python -B experiments/worm_uncertainty/plot_stress_3d.py \
  --input output/worm_uncertainty/phantom_stress_repeat \
  --output output/worm_uncertainty/phantom_stress_figures_repeat
```

Twenty-six CPU tests pass. New checks cover wrapped angular masks, unchanged fits
and covariance after arbitrarily changing withheld measurements, and the joint
versus conditional variance ordering. Formatting and static checks pass for all
27 active scripts. No GPU experiment is required for this stage.

### Component restoration and uncertain Gaussian shape

A paired representation-repair control now restores the previously omitted fourth
component. It uses exactly the previous four-component truth, acquisition windows,
noise draws, prior settings and first-three-component initialization perturbations.
The new fourth component starts near its true centre with its true width: this is
an oracle capacity control, not an automatic component-discovery method. Both
phantoms, five windows, ten noise draws and two starts give **200 repair fits**.

| Four-component target | Joint voxel coverage | Voxel RMSE | Withheld predictive coverage |
|---|---:|---:|---:|
| Three-component fit (previous) | 8.86–29.29% | 0.03977–0.04758 | 43.87–65.32% |
| Restored four-component fit | 90.26–96.32% | 0.000656–0.001748 | 94.84–95.44% |

Ranges summarize case means, not confidence intervals. Nominal coverage is 95%;
RMSE and interval widths are in synthetic density units. All repair fits terminate
successfully within ten evaluations, without active bounds. This paired control
supports missing representational structure as a cause of the earlier failure.
It does not establish that arbitrary worm residuals have the same cause.

![Restoring the missing component reduces error and improves coverage](../../docs/figures/worm_uncertainty/shape_controls_v1/repair.png)

The second stage rotates each of the four true components and learns centres,
log widths, XYZ Euler rotations and densities jointly. Width bounds are 0.04–0.35;
Euler-angle bounds are ±1.4 radians. Gaussian priors have SD 0.35 on log widths and
0.5 radians on rotations, centred on the fixed initialization. Initial widths and
rotations are perturbed from truth (SD 0.1 and 0.15 respectively), so this is still
favourable initialization. The two phantoms use 180°/0°, 90°/0° and 90°/120° windows,
ten noise draws and two starts, giving **120 shape fits**. This stage has rotated
truth and should not be treated as a same-data comparison to the repair stage.

At each fitted mean, nested local Gaussian covariances include density alone,
then centres, then log widths, then rotations. Each covariance is the inverse of
the corresponding principal block of the joint Gauss–Newton precision: excluded
parameters are held fixed, not marginalized. The complete model retains all
cross-covariances. First-order propagation gives voxel and ray uncertainty.

| Parameters included in covariance | Voxel coverage across six cases |
|---|---:|
| Density only | 14.16–25.05% |
| Density + centres | 46.38–66.37% |
| Density + centres + log widths | 66.54–85.03% |
| All, including rotations | 85.47–96.58% |

![Nested shape uncertainty improves coverage but exposes a difficult case](../../docs/figures/worm_uncertainty/shape_controls_v1/shape_covariance.png)

Means are identical across the four covariance choices in each fit. Improvement
therefore comes from propagating omitted uncertainty, not improving the mean.
For example, the separated phantom at 90°/120° has voxel coverage
15.67% → 53.79% → 73.18% → 94.48%; mean interval width grows from 0.00112 to 0.01041.
Coverage must be assessed with width; wider alone is not a success criterion.

**A boundary failure remains.** All 120 shape fits report solver success within
42 evaluations, but 22 fits reach Euler rotation bounds: 4/20 for the separated
phantom at 90°/0° and 18/20 for the overlapping phantom at 90°/0°. The local
untruncated covariance ignores these constraints and is not a valid description
of a constrained posterior near the boundary. For the overlapping case, complete
voxel coverage is only **85.47%**, withheld latent coverage **82.09%**, and withheld
predictive coverage **92.64%**. Its two starts yield voxel coverage 91.58% and
79.37%, showing sensitivity to initialization (and the initialization-centred
prior). A successful stopping condition does not settle this issue.

No bounds or priors were changed after inspecting these results. The
[bound inspection](../../docs/figures/worm_uncertainty/shape_controls_v1/bound_diagnostics.json) identifies
which component/axis hits a limit. The [result record](../../docs/figures/worm_uncertainty/shape_controls_v1/results.json)
contains the frozen protocol, all case summaries and provenance. Local
`output/worm_uncertainty/shape_controls_v1/` retains per-fit parameters, convergence
and metrics, plus first-draw truth/reconstruction volumes.

**Next decision before worm transfer:** retain full shape covariance as the
controlled baseline, but test the difficult rotated case with an appropriate
rotation/covariance parameterization or boundary-aware posterior calculation.
Separate changes to parameterization, prior and initialization so any improvement
has an identifiable cause. Test nonlinear posterior propagation against the local
approximation and use new fixed validation cases. Automatic representation
selection and a possible residual model remain open; these toy controls do not
justify transferring a calibrated-uncertainty claim to worms.

Reproduce with the research environment activated and the preceding stress-test
artifacts available (fresh output paths):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/worm_uncertainty/run_shape_controls.py \
  --parent output/worm_uncertainty/phantom_stress_v1 \
  --output output/worm_uncertainty/shape_controls_repeat
MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib python -B experiments/worm_uncertainty/plot_shape_controls.py \
  --parent output/worm_uncertainty/phantom_stress_v1 \
  --input output/worm_uncertainty/shape_controls_repeat \
  --output output/worm_uncertainty/shape_figures_repeat
```

The frozen settings are in `shape_protocol_v1.json`. The runner checks parent
provenance and the exact restoration noise scale; output directories cannot be
reused. `progress.json` is diagnostic progress, not a resumable checkpoint.
Thirty CPU tests pass, including all shape derivatives for one and multiple
components, rotated-ray numerical quadrature, diagonal-model reduction and nested
covariance ordering. Static/format checks pass for 31 scripts. No GPU is needed.

### Rotation bounds, initialization sensitivity and a stability check

The follow-up separates two confounded choices in the previous shape fit: Euler
rotation bounds and a prior centred on each optimizer start. A fixed factorial
protocol tests bounded/unbounded rotations with start-specific/shared priors on
90°/0° and 90°/120° overlapping targets plus a new phantom at 90°/60°. Ten noise
draws and two starts give **240 fits**. The shared prior reference uses a separate
fixed seed, 907; it remains perturbed from synthetic truth. The bounded,
start-prior control reproduces the preceding experiment.

Removing rotation bounds improves the original difficult case's voxel coverage
from 85.47% to 91.12%, but creates failures elsewhere: two unbounded fits exhaust
the 150-evaluation budget. Sharing the prior makes objectives comparable across
starts, but does not eliminate local minima. These outcomes argue against simply
removing angle bounds and proceeding with a single fit.

An adaptive **60-fit Cholesky follow-up** represents each covariance as
`Sigma = L L^T`, with positive diagonals represented in log space. This removes
Euler coordinate limits and duplicate rotation representations. It also changes
the shape prior to Gaussian log-diagonal SD 0.35 and raw off-diagonal SD 0.1,
removes width bounds, and uses Jacobian-based optimizer scaling. Consequently,
improvement cannot be attributed to parameterization alone. All these choices
were fixed before that follow-up, without tuning on coverage.

Cholesky coordinates solve the original 90°/0° case stably, but two starts are
insufficient elsewhere. Some fits terminate at poor local minima, and two exhaust
the budget. The plot retains these failures rather than displaying only the
successful first case.

![Factorial controls and the two-start Cholesky follow-up](../../docs/figures/worm_uncertainty/rotation_sensitivity_v1/controls.png)

The final optimizer control keeps that Cholesky model and shared prior fixed,
uses starts 81, 193, 907 and 421, and raises the budget to 300 evaluations:
**120 fits**. For each noise draw, the audit selects the two lowest-objective
converged fits. Selection uses acquired measurements plus the fixed prior only.
The stability check requires:

- At least two converged candidates, both without active bounds.
- Relative objective difference at most `1e-5`.
- Relative acquired-prediction difference at most `1e-4`.

The thresholds were fixed before this optimizer follow-up. Truth, withheld rays
and their metrics do not participate in selection or in the check. The audit
returns exit code 2 if any draw fails, so downstream work can be conditional on
success. Passing checks do not establish global optimality or ensure that
unmeasured regions agree.

**All 30 draws pass.** Two of the 120 individual fits still exhaust the budget;
poor converged candidates also remain. The result supports an audited multistart
procedure, not a claim that initialization sensitivity has disappeared from
individual fits. Metrics below use the lowest-objective converged fit per draw.

| Case (span/start) | Stability checks | Voxel coverage | Withheld latent coverage | Predictive coverage | Voxel interval width | Voxel RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Overlap 90°/0° | 10/10 | 97.36% | 96.80% | 95.71% | 0.01341 | 0.003590 |
| Overlap 90°/120° | 10/10 | 97.32% | 96.92% | 95.67% | 0.01711 | 0.004252 |
| New phantom 90°/60° | 10/10 | 95.92% | 96.56% | 94.95% | 0.01049 | 0.003078 |

![Audited multistart coverage, widths and reconstruction error](../../docs/figures/worm_uncertainty/cholesky_multistart_v1/selected_results.png)

Coverage targets 95%; widths and voxel errors use synthetic density units. These
are means over ten paired noise draws per case, not confidence intervals. The new
phantom was fixed before the factorial test, then reused during the adaptive
Cholesky and optimizer follow-ups. It is therefore development evidence, not an
untouched final validation set. The local Gaussian posterior and linear
propagation remain approximations. Scanner geometry is exact, component count is
known, and starts/prior reference are informed by truth.

**Decision before worm transfer:** the concrete rotation-bound failure is avoided
with positive-definite covariance coordinates; optimizer sensitivity is handled
by a shared prior, multiple starts, measured-objective selection and an explicit
agreement check. Retain this procedure as a candidate and keep the failed controls.
Do not silently accept a single fit or choose a start using withheld performance.
A worm implementation still needs a measured-data-only initializer/prior, a
scalable covariance calculation, and fresh validation of interval calibration;
the toy-model results do not establish calibrated worm uncertainty.

Records: [factorial and two-start follow-up](../../docs/figures/worm_uncertainty/rotation_sensitivity_v1/results.json),
[audited four-start results](../../docs/figures/worm_uncertainty/cholesky_multistart_v1/results.json).
The frozen protocols are `rotation_sensitivity_protocol_v1.json`,
`cholesky_protocol_v1.json` and `multistart_protocol_v1.json`. Local outputs retain
all candidates, including failures, with objective values, parameters, convergence
and coverage metrics. Both start/budget changes in the final follow-up are explicit;
it does not isolate their individual contribution.

Reproduce after activating `trdp2-r2`. The parent shape-control output is required;
use fresh destinations:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/worm_uncertainty/rotation_sensitivity.py \
  --output output/worm_uncertainty/rotation_sensitivity_repeat
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/worm_uncertainty/cholesky_sensitivity.py \
  --output output/worm_uncertainty/cholesky_sensitivity_repeat
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/worm_uncertainty/cholesky_sensitivity.py \
  --protocol experiments/worm_uncertainty/multistart_protocol_v1.json \
  --output output/worm_uncertainty/cholesky_multistart_repeat
OPENBLAS_NUM_THREADS=1 python -B experiments/worm_uncertainty/audit_multistart.py \
  --input output/worm_uncertainty/cholesky_multistart_repeat \
  --output output/worm_uncertainty/cholesky_multistart_repeat/stability_gate_audit.json
MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib python -B experiments/worm_uncertainty/plot_multistart.py \
  --input output/worm_uncertainty/cholesky_multistart_repeat \
  --output output/worm_uncertainty/multistart_figures_repeat
```

Thirty-four CPU tests pass. They include exact reproduction of the original
bounded objective, Euler/Cholesky agreement for the same physical Gaussian,
Cholesky feature derivatives, and proof that audit selection ignores truth metrics
and returns a failing exit code when fewer than two candidates converge. No GPU
training or worm reconstruction is launched by this study.

### 19 October — initialization and priors from acquired data

The next prerequisite removes truth-derived starting positions, widths and prior
references. A fixed dictionary contains 405 isotropic Gaussians (9×9×5 centres,
width 0.14 in normalized coordinates). Greedy residual correlation selects four
atoms, with a nonnegative density solve after each selection. Only acquired noisy
projections and their angles enter this calculation. The resulting parameters
form one shared prior reference for all starts on that acquisition/noise draw.
Prior widths, the component count and the search grid remain fixed design choices.

Two new four-component phantoms (`fresh_spread`, `fresh_cluster`) are tested at
120°/30° and 90°/75°, with ten paired noise draws and four optimizer starts:
**160 initial fits**. Noise SD is a fixed **0.003** in synthetic projection units;
it is not derived from full-angle clean data or withheld measurements. The fresh
protocol was fixed before running these cases. Truth is used for synthesis and
scoring, while withheld views are used only for scoring.

The initial run passes **37/40** stability checks, despite all 160 fits reporting
successful termination without active bounds. The three failures are on the
clustered phantom: their relative objective gaps exceed `1e-5`, while acquired
prediction disagreement remains below `1e-4`. This again shows why solver success
alone is insufficient. The [initial record](../../docs/figures/worm_uncertainty/measured_init_v1/results.json)
retains these failures and all initial case metrics.

An explicitly adaptive fallback adds seeds 907, 202, 1123 and 151 only to failed
checks. It preserves the measured-data prior reference, 300-evaluation budget and
both agreement thresholds. The extra seed list was fixed before the fallback;
no truth or withheld metric selects which fits to run or accept. **Twelve extra
fits** bring the result to **40/40 passing checks**, with 172 total fits and no
active bounds. The largest final relative objective gap is 1.64e-11 and largest
acquired-prediction disagreement is 3.13e-7. All candidates remain available.

| Fresh case (span/start) | Initial / final checks | Voxel coverage | Withheld latent coverage | Predictive coverage | Voxel width | Voxel RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Spread 120°/30° | 10/10 → 10/10 | 95.45% | 95.85% | 95.30% | 0.00771 | 0.002210 |
| Spread 90°/75° | 10/10 → 10/10 | 95.80% | 95.04% | 94.81% | 0.01127 | 0.003434 |
| Cluster 120°/30° | 9/10 → 10/10 | 95.49% | 94.94% | 95.06% | 0.00901 | 0.002766 |
| Cluster 90°/75° | 8/10 → 10/10 | 96.22% | 96.27% | 94.78% | 0.01523 | 0.004670 |

Metrics average the selected lowest-objective converged fit over ten noise draws.
Coverage targets 95%; widths and voxel errors are in synthetic density units.
Width must be judged alongside coverage, and lower RMSE is better. Fallback
barely changes these aggregate metrics: it supplies corroborating converged fits
rather than tuning intervals against evaluation results.

![Measured-data-only initialization: coverage and interval widths](../../docs/figures/worm_uncertainty/measured_init_fallback_v1/coverage.png)

![Fresh phantom truth, selected reconstruction, absolute error and SD](../../docs/figures/worm_uncertainty/measured_init_fallback_v1/slices.png)

The slice fixes the spread phantom at 90°/75°, noise draw 0 and the central z
plane. Truth/reconstruction share density limits; error/SD share limits. Low error
is good; low SD alone is not proof of accuracy. Coverage elsewhere uses intervals
of approximately mean ±1.96 SD, not ±1 SD.

**Interpretation and remaining limits.** This supports acquired-data initialization
and adaptive multistart selection on fresh matched synthetic targets. The prior
reference is estimated from the same measurements, and local covariance conditions
on that estimate; it is not a full hierarchical posterior that propagates prior
estimation uncertainty. The four-component count is predeclared, the forward
model matches synthesis, and noise is known. The fallback was developed after
seeing the initial failures, so its final results are adaptive development evidence.
Two fresh phantoms do not establish general calibration, biological independence,
or performance under unknown noise and model mismatch.

The next step is a small worm-slab diagnostic using measured-only initialization,
explicit computational limits, and withheld-view error/coverage checks. Its richer
representation and scalable covariance calculation still need implementation and
validation. Retain synthetic checks and failed-run evidence; physical worm density
scaling and matching voxel ground truth remain unresolved.

[Final results and provenance](../../docs/figures/worm_uncertainty/measured_init_fallback_v1/results.json).
Protocols: `measured_init_protocol_v1.json` and `measured_fallback_protocol_v1.json`.
All 36 CPU tests pass, including exact recovery of a dictionary atom, deterministic
starts and withheld-data isolation. Formatting/static checks pass for 45 scripts.

Run this CPU study from the repository root:

```bash
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib
python -B experiments/worm_uncertainty/measured_init_study.py \
  --output output/worm_uncertainty/measured_init_repeat
python -B experiments/worm_uncertainty/audit_multistart.py \
  --input output/worm_uncertainty/measured_init_repeat \
  --output output/worm_uncertainty/measured_init_repeat/stability_gate_audit.json
```

The audit deliberately returns exit code 2 when a draw fails. Preserve its report;
do not loosen the thresholds or automatically continue to a worm experiment.
The recorded fallback is bound by hashes to the accepted original run and can be
reproduced with a fresh destination:

```bash
python -B experiments/worm_uncertainty/measured_init_fallback.py \
  --input output/worm_uncertainty/measured_init_v1 \
  --output output/worm_uncertainty/measured_init_fallback_repeat
python -B experiments/worm_uncertainty/audit_multistart.py \
  --input output/worm_uncertainty/measured_init_fallback_repeat \
  --output output/worm_uncertainty/measured_init_fallback_repeat/stability_gate_audit.json
python -B experiments/worm_uncertainty/plot_measured_init.py \
  --input output/worm_uncertainty/measured_init_fallback_repeat \
  --output output/worm_uncertainty/measured_init_figures_repeat
```

For a different parent run, record a new fallback policy with that parent's report
and audit hashes before execution. The code verifies regenerated acquired-data
prior references against the saved originals before adding any candidates. The
raw parent is preserved and only failed checks receive additional starts.

### 19 October — first measured-only worm shape pilot: stability failure

The measured-only pipeline has now been exercised on actual worm projections,
using a deliberately small CPU diagnostic. It fits 16 Gaussians (160 parameters)
to 17 acquired angles in a 120°/0° window, holding out eight other angles. The
25-angle grid takes every fourth source view; 62 detector samples take every
fifth pixel. Eight axial planes span source z=224…280 in steps of eight. Sampling
is by selecting original rays, not averaging detector bins. This differs from
the earlier contiguous eight-slice, 1,500-Gaussian experiment and is not a direct
performance comparison.

The normalization is the acquired sampled projection RMS, **6094.0105 stored
units**. Synthetic Gaussian noise SD 0.05 is added after normalization, with seed
20261019. Starts and the shared prior reference come only from noisy acquired
samples; withheld projections are reserved for evaluation. The dictionary uses
fixed widths (0.16, 0.16, 0.12), with 11×11×3 candidate centres. Four starts have a
fixed budget of 150 function evaluations each.

**The experiment fails the optimizer stability check.** All four fits exhaust
that budget and retain density parameters at their lower bound. With no converged
candidate, the displayed fit is the lowest-objective unfinished candidate, seed
421. Its maps and covariance are provisional diagnostics; they are not a valid
calibrated posterior or an accepted reconstruction.

| Diagnostic | Result |
|---|---:|
| Converged fits | 0/4 |
| Acquired clean-projection relative error | 20.51% |
| Withheld clean-projection relative error | 25.22% |
| Density-only latent / predictive coverage | 1.52% / 15.91% |
| Full-shape latent / predictive coverage | 5.93% / 16.79% |
| Full-shape predictive mean interval width | 0.20486 normalized projection units |

Coverage targets 95% on withheld foreground rays (clean normalized projection
above 0.05). These very low values are failure diagnostics at an unfinished fit,
not evidence that the full-shape posterior is calibrated. Clean withheld
projections test latent intervals; noisy withheld projections test predictive
intervals. The latter add the known injected noise variance.

![Provisional coverage and widths after the failed stability check](../../docs/figures/worm_uncertainty/worm_shape_pilot_v1/coverage.png)

![Withheld ray profile and provisional worm density maps](../../docs/figures/worm_uncertainty/worm_shape_pilot_v1/profile_and_maps.png)

The 144° ray profile at source z=256 is fixed and withheld. Density-map brightness
shows relative reconstructed values, not accuracy; SD is also in uncalibrated
relative density units. There is no matching voxel truth for this worm. Confidence
in fine internal structure or quantitative biological density is not established.

A concrete initialization issue is visible: the nonnegative dictionary solve
assigns appreciable density to only **5 of 16** components; the other 11 are
raised to the initializer floor of 0.01. This motivates a more suitable measured-
data support/grid and width dictionary. It does not isolate the cause of final
error: incomplete optimization, boundary effects, insufficient or poorly allocated
representation, and simulation conventions remain possible contributors.
The local precision condition number is approximately 5.91e9, adding a numerical
conditioning concern to the already failed convergence check.

The 160×160 covariance occupies 204,800 bytes; query uncertainty is evaluated in
chunks. This demonstrates a small dense calculation, not scalability to the
previous 1,500-Gaussian model. The fitting/evaluation portion took about 66 seconds
on this workstation, excluding source hashing and initialization; runtime is not
a controlled benchmark.

**Assessment:** the synthetic work has produced a working implementation and
useful failure checks, but calibrated worm reconstruction has not succeeded.
Next, compare acquired-data initializers with finer, multiple-width dictionaries
or measured-only reconstruction support, then separate solver-budget effects from
component-capacity effects under a fixed evaluation protocol. Keep withheld
metrics for evaluation rather than using them to select starts or inflate
intervals. The failed small pilot is preserved as a reference.

[Results and provenance](../../docs/figures/worm_uncertainty/worm_shape_pilot_v1/results.json).
The protocol is `worm_shape_protocol_v1.json`; local outputs are in
`output/worm_uncertainty/worm_shape_pilot_v1/`. Thirty-eight CPU tests pass,
including sampled physical coordinates, acquired-only normalization and
anisotropic dictionary consistency. No full GPU training was launched.

Reproduce the CPU pilot with a fresh output path after activating `trdp2-r2`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B experiments/worm_uncertainty/worm_shape_pilot.py \
  --source ../worm_data/worm_004_gt.tiff \
  --output output/worm_uncertainty/worm_shape_repeat --execute
MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib python -B experiments/worm_uncertainty/plot_worm_shape_pilot.py \
  --input output/worm_uncertainty/worm_shape_repeat \
  --output output/worm_uncertainty/worm_shape_figures_repeat
```

Without `--execute`, the pilot prints the protocol. A completed diagnostic process
can still fail scientific acceptance: inspect `report.json` → `gate.passed` and
`fits.json`. The runner preserves maps for diagnosis when no fit converges; it
does not automatically increase budgets, modify priors, or restart training.

### 21 October — paired worm initialization, budget and capacity controls

The failed 19 October pilot is preserved. The new CPU study reuses its exact
saved normalized projections, injected noise, eight sampled axial planes and
120° acquisition mask, verified by SHA-256. This is a development diagnostic on
one worm, one window and one noise draw; it is not independent validation.

The replacement initializer uses a fixed full-field 21×21×3 centre grid and
three transverse widths (0.04, 0.08, 0.16), retaining axial width 0.12. Greedy
nonnegative fitting sees only acquired noisy rays; minimum centre spacing is
0.04. The 3,969-atom dictionary occupies about 268 MB, with derivative
intermediates computed in chunks of 64 atoms. This is a small CPU reference,
not a demonstrated scalable alternative to the native GPU reconstruction.

The frozen protocol compares new starts with the old prior (16 components,
150 evaluations); new starts with a matching acquired-data prior (16/150);
the same latter starts/prior with a larger budget (16/450); and more components
(32/450). All arms retain four seeds, data, noise variance, solver tolerances
and acceptance thresholds. The component-count comparison also changes the
prior dimension. Grid spacing, width choices and separation change together,
so this does not isolate which dictionary design choice helps.

For the old-prior arm, deterministic nearest-centre assignment matches component
labels. Seed 0 gives a paired unperturbed prior comparison; label reordering
means the other seeds' perturbations differ between old- and new-prior arms.
The 16-component 150-versus-450 budget comparison uses identical initial
parameters and priors for every seed. Start selection uses acquired-data
objectives only. If no candidate converges, the lowest unfinished objective is
retained solely for provisional diagnosis. A passing gate additionally requires
two converged starts, objective/prediction agreement, and no active bounds.

No new uncertainty intervals are computed in this study: optimization must be
resolved before interpreting local covariance as a posterior. Lower clean
projection error is better, but it does not establish voxel accuracy or
calibration. Matching voxel truth and physical projection scaling remain
unresolved.

Reproduce the controls on a CPU workstation with the preserved parent outputs
available. Use fresh output directories; the runner intentionally refuses to
replace an existing run. Full GPU training is not required.

```bash
cd ~/espaces/travail/trdp2/r2_gaussian
source ~/espaces/travail/trdp2/tools/miniforge3/etc/profile.d/conda.sh
conda activate trdp2-r2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/cache-mdzislam/matplotlib
python -B experiments/worm_uncertainty/worm_fit_controls.py \
  --parent output/worm_uncertainty/worm_shape_pilot_v1 \
  --output output/worm_uncertainty/worm_fit_controls_repeat --execute
python -B experiments/worm_uncertainty/plot_worm_fit_controls.py \
  --input output/worm_uncertainty/worm_fit_controls_repeat \
  --output output/worm_uncertainty/worm_fit_controls_figures_repeat
```

Without `--execute`, the runner prints its protocol and writes nothing. During
execution, `fits.json` is checkpointed after each start; `report.json` appears
only after all 16 fits finish. A completed process is not scientific acceptance:
inspect each row's `gate.passed`. Output archives are local/gitignored; the
tracked compact results and figures provide the reviewable evidence.

The new initializer assigns appreciable starting density to **16/16** components
(the old initializer: **5/16**). Its acquired noisy-data relative residual falls
from **37.15% to 23.02%**. With 32 components, **31/32** are appreciably active and
the realized residual is **21.38%**, including the density floor. “Active” here
means starting density strictly above 0.01; it is not a solver bound diagnostic.

Increasing the 16-component budget from 150 to 450 produces two converged starts,
but their relative objective gap is **5.19%** and acquired-prediction disagreement
is **5.48%**. Acceptance thresholds remain **0.001%** and **0.01%**, respectively.
The other two starts are unfinished. Thus neither solver success for one start
nor improvement in projection error establishes a stable reconstruction.

| Setting | Solver budget | Converged starts | Acquired clean error ↓ | Withheld clean error ↓ | Stability |
|---|---:|---:|---:|---:|---|
| Preserved old initializer/prior, 16 Gaussians | 150 | 0/4 | 20.51% | 25.22% | Fail |
| New initializer, old prior, 16 Gaussians | 150 | 0/4 | 11.34% | 20.93% | Fail |
| New initializer/prior, 16 Gaussians | 150 | 0/4 | 11.34% | 23.12% | Fail |
| New initializer/prior, 16 Gaussians | 450 | 2/4 | 11.20% | 19.84% | Fail |
| New initializer/prior, 32 Gaussians | 450 | 0/4 | 8.37% | 23.81% | Fail |

Errors are relative L2 over all rays in the indicated clean evaluation set,
not foreground-only coverage measures. Each row uses its predeclared acquired-
objective selection rule; none is selected by withheld error. All four new
settings fail acceptance, and only 2 of the 16 new starts report solver success.
The four 32-component objectives range from approximately 32,119 to 87,962,
with every start exhausting its 450-evaluation budget.

**Interpretation:** better initialization substantially improves the provisional
fit. More iterations permit some solver termination but leave distinct solutions.
Increasing component count reduces acquired error while worsening withheld error
relative to 16 components at the same budget. This is a warning against equating
training fit with missing-view generalization, not proof of a capacity optimum:
none of these settings passes stability, and the capacity comparison also changes
the prior dimension. Representation error cannot yet be separated cleanly from
incomplete optimization. These sparse-sampling results are not directly comparable
to the earlier 1,500-Gaussian contiguous-slab baseline.

![Projection error and failed optimization acceptance checks](../../docs/figures/worm_uncertainty/worm_fit_controls_v1/fit_controls.png)

![Fixed withheld profile, with clean reference](../../docs/figures/worm_uncertainty/worm_fit_controls_v1/withheld_profile.png)

The profile uses the same preselected 144° view and source z=256 as the parent
pilot. Closeness to the black reference is better; projection intensity itself
is not a quality score. No error bars or calibration claims are inferred from
this single noise realization.

**Next:** freeze a staged density → centre → full-shape optimization comparison
under a fixed shared prior, retaining the same data and withheld-only evaluation.
Evaluate both optimization agreement and missing-view error before adding new
posterior intervals. Physical projection scaling, matching deformed voxel truth,
and deformation-separated evaluation remain outstanding. The Weeks 4–5 evidence includes these failed worm checks alongside the
controlled-phantom results; calibrated worm uncertainty remains unresolved.

[Compact results and provenance](../../docs/figures/worm_uncertainty/worm_fit_controls_v1/results.json).
Protocol: `experiments/worm_uncertainty/worm_fit_controls_protocol_v1.json`;
local full outputs: `output/worm_uncertainty/worm_fit_controls_v1/`.
Checks: **42 CPU tests passed**, all **51 experiment Python files** pass the pinned
formatter/linter, saved parent/artifact hashes are verified, and the default dry
run writes no output. All 16 fits finished; no GPU experiment was launched.
