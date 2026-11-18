# ICECREAM reconstruction pilot — frozen 10 November 2026

## Question and status

Does per-acquisition self-supervised reconstruction improve volume and missing-ray
errors over matched FBP and nonnegative SART? The previous v3 results motivate an
alternative representation; they do not rule out all Gaussian methods. The starting
checkout was clean at `36128bc`, including the Week 6 report. Saved v3 verification
passed without rerunning training. Samplers remain paused. No reserved file is used.

The machine-readable [protocol](../protocols/icecream_pilot_v1.json) was written
before new reconstruction outcomes. `output/worm_uncertainty/icecream_pilot_v1/`
contains immutable stage directories and dirty-source snapshots. No push is authorized.

## Method fidelity and necessary adapters

- ICECREAM official MIT source: `swing-research/icecream`, pinned
  `39c891a55af3e1cd96ef0d585668b0c2af4c177d`. The original default 16,313,248-parameter
  `unet3d_bf` and **unmodified `EquivariantTrainer.compute_loss`** are called directly.
  Both cross-pair data losses and rotated cross-pair double-network equivariance
  losses backpropagate through all applications. Equivariance weight is 2, Adam
  learning rate 1e-4. Inference averages `f(A(f(y0)))` and `f(A(f(y1)))`.
- Necessary CT adapter: arrays `(z,y,x)` become computational `(x,y,z)` without
  resampling. The Fourier wedge is extruded along physical z, permitting arbitrary
  modular acquisition starts. Upstream's 2D-mask broadcast uses a different array
  axis; a permutation-aware test verifies the supported symmetric case away from
  floating-point spherical boundaries. An explicit axial DC line is retained.
  Upstream's subsequent real-mask symmetrization, spherical support, doubled FFT
  size and central-half loss window are retained. These are idealized Fourier
  operations, not the exact discrete CT projection operator.
- Rotation descriptors enumerate the actual cubic isometries used by upstream's
  rotation application. Their transformed wedges must differ by >0.5 relative norm;
  duplicates are removed. This avoids relying on the upstream generator's unrecorded
  permutation-loop axes. Reflections are retained, as in the official defaults.
  Rotation counts depend on acquisition geometry; all physical spacings are equal.
- Fixed projection normalization 6180.275452553537 is inherited from development.
  A fixed volume gain of 64 is applied and inverted. Per-tomogram empirical
  standardization is disabled to retain intensity scale and avoid random pair-wise
  affine factors. Patch coordinates use a fixed central 128×128 region for worms,
  all 64 contiguous axial planes, and the whole support for phantoms. These are
  declared data/training adaptations, not exact reproduction of the cryo-ET study.
- Inference streams crops and accumulates in FP32 instead of materializing all
  crops in FP16. The original two-pass expression and central-half overlap are
  preserved. Identity, scale and odd-dimension coverage checks pass.
- Full-angle positive control uses the same network's N2N loss with equivariance
  weight zero because no wedge-changing rotations exist. It is labeled explicitly.

The paper's loss and inference are in [Kishore et al., equations 4–7](https://journals.iucr.org/d/issues/2026/04/00/zi5009/).
Its cropping approximation and approximate rotation-distribution assumption remain
limitations here. This is per-scan learning from acquired data, not zero-shot transfer
and not the earlier supervised sinogram CNN.

## Acquisition, pairs and baselines

Three pilot phantoms have independently generated smooth blobs, an elongated sharp
ellipsoid, a shell and a box, with different poses. The first is evaluated at full,
120° and 90° coverage. Two additional fixed cases change the object and acquisition.
Five runs total; generation seeds and rotations are in the protocol. Implementation
checks use a separate shifted anisotropic Gaussian and independent mixed object.

Worm cases are the first three training-role files in manifest order: 004 at 120°,
012 at 90°, and 013 at start36°/span120°. They have prior intake and supervised
development exposure. Numeric grouping does not establish independent organisms.
Slab z224:288 uses every original angle/detector sample: 64×100×310, 1 µm spacing,
angles0:1.8:178.2°, symmetric detector centres `j-154.5`. This is a new, denser
experiment; old 25×62 scores are not matched baselines.

For each case, two independent Gaussian full-grid fields with SD0.05 are added to
the same permitted clean acquired signal, then masked. Training receives only the
two masked noisy arrays and their FBP volumes. Clean withheld data and phantom truth
are saved separately under `evaluation_only/`; the trainer never opens them.
All methods receive the same pair. FBP and SART use its average, whose noise SD is
0.05/sqrt(2). No second-observation advantage is attributed to ICECREAM alone.

The new CPU projector is pixel-driven linear detector splatting with its exact
matrix transpose. Pixel centres, mass/path-length factor and circular support are
explicit. FBP uses the original 1.8° quadrature with missing angles zeroed; it does
not artificially multiply intensity by pi/acquired-span. SART uses five sequential
angle sweeps, relaxation0.5, zero initialization and positivity. Its data operator
matches evaluation exactly. Neither is asserted to reproduce the unknown simulator.
TV is omitted because the old SART baseline did not contain a verified matching
voxel-TV algorithm. The historical CNN is omitted because its input grid differs.

## Pre-training checks and frozen budget

Geometry passed: analytic projection error0.955%, adjoint1.34e-16,
180° reversal1.12e-15, FBP scale0.98734 and centroid error0.135px.
Ten targeted CPU tests passed, including reserved exclusion, no withheld leakage,
independent noise/average variance, SART data isolation, physical axes, Fourier
orientation, patch merging and nonzero finite gradients through the official loss.

Training-only profiling on the 120° phantom observation selected 48³ patches,
batch1, mixed precision, **2,000 updates per phantom and 4,000 per worm**.
Profiled update time was0.0750s; AMP versus FP32 loss difference2.58e-6 and gradient
relative difference2.61% passed prospective2%/5% tolerances. Peak allocated1.21GiB,
reserved1.49GiB, free16.29GiB. The 32³ alternative also passed. No evaluation target
or reconstruction score selected these settings.

Every run selects the fixed final update. Per-case training caps270s phantom/900s
worm; inference caps90s/300s. The workflow imposes pooled30min phantom and105min
worm caps, CPU preparation/evaluation30min, and stops on failure rather than retrying.
These are tighter than the total180GPU-minute authorization. Every failed stage and
checkpoint remains. The main study uses one training seed. Extra seeds and unrolling
are conditional; no open-ended extensions are allowed.

The separate Fourier audit measures substantial model mismatch: central-region
CT-FBP versus the actual configured Fourier operator differs by28–43% on limited
smooth/mixed controls; cropping contributes12–27% on mixed controls. The full-angle
smooth difference is3.68%. This is a quantitative limitation of a short-patch ideal
wedge applied to finite-angle CT, not an adjoint/coordinate failure. An exploratory
un-symmetrized larger-grid check reduced smooth120° mismatch, but no settings were
changed to optimize it. The actual upstream symmetrization is preserved and audited.
Run the bounded pilot to assess this adapted method; do not claim exact CT consistency.

## Environment and source inspection

The isolated environment is under ignored `output/worm_uncertainty/alternative_reconstruction_sources/env`:
Python3.11, PyTorch2.8.0+cu126, NumPy2.2.6, SciPy1.15.3. It meets official package
requirements; `pip check` passes. The existing Python3.9/PyTorch2.1 CUDA11.8
environment and native backends are unchanged. TIGRE is not imported.
`nvidia-smi` must run outside the sandbox: its missing device nodes initially gave
a misleading driver failure. On liet, the external check shows RTX4000Ada20GiB,
driver620.70.06, no compute workload and about18.4GiB free before this pilot.

Comparator source inspection only (neither launched):

- DeepDeWedge `072075692a44a8f17394214369e6e762abe52bc3`, BSD2-Clause;
  original environment recommends Python3.10/PyTorch2.2/CUDA11.8 with Lightning1.8.
  Its masked additional-corruption loss and iterative input replacement differ
  from ICECREAM. [Paper](https://doi.org/10.1038/s41467-024-51438-y),
  [official code](https://github.com/MLI-lab/DeepDeWedge).
- Efficient-unrolling `285c20600c6348fa1a4bf99754b0ace8fd72f877`:
  no LICENSE file or grant found in the pinned tree. Inspect only; do not vendor
  or redistribute its code. Its broad environment includes ASTRA/DeepInverse.
  The paper's specific contributions are domain partitioning with known training
  context/estimated inference context, and diagonal–circulant normal-operator
  approximation. A small exact-operator supervised phantom unroll would be an
  independently implemented inspiration/control, not reproduction of those methods.
  [Paper](https://arxiv.org/html/2601.02141v1),
  [official code](https://github.com/romainvo/efficient-unrolling).

## Evaluation and decisions

Report each case separately: volume L2/RMSE/foreground error for phantoms,
withheld clean-projection L2, acquired clean/noisy residuals, negative voxel fraction,
timing and memory. There are no exact worm voxel targets. Fixed central planes,
sinograms, Fourier views and errors use common scales. Acquired FBP gradient-energy
ratios diagnose orientation anisotropy but cannot prove distributional invariance.
Each acquisition is trained independently. A visual change, FSC or agreement is
not calibrated uncertainty, true density or independent-organism generalization.

Results will be appended in a separate `ICECREAM_RESULTS_V1.md`; this protocol
and original numerical failures remain historical evidence.

## Numerical amendment after the first worm training failure

The original mixed-precision worm004 run stopped after six accepted updates with
nonfinite gradients. Its checkpoint, loss history and failure report are retained.
The same diagnostic crop had finite unscaled AMP and FP32 gradients; a fresh
16-update FP32 smoke also passed. This is consistent with loss-scaling overflow,
not proof of the exact cause. No worm withheld outcomes were inspected to choose
the repair.

A separate `precision_amendment.json` froze one FP32 attempt for each of the three
already declared worm cases. Patch size, seed, observations, normalization, model,
4000 updates, final-checkpoint rule and time caps were unchanged. New artifacts use
`runs_fp32` and `evaluation_fp32`; original outputs remain immutable. This numerical
amendment is not an outcome-selected hyperparameter experiment. Failed-run and
smoke time count toward the total budget. No automatic further retries are allowed.

The common circular reconstruction support is imposed identically during final
evaluation of all methods. Raw network output and its outside-support squared norm
are retained. Negative values inside support are not clipped for ICECREAM or FBP.
