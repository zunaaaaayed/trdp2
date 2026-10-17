# Worm uncertainty development

Started 17 October 2026. Scope: uncertainty under missing angles and controlled
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

## Immediate next steps

1. CPU tests and one development preparation smoke check (no performance claims).
2. Analytic phantom checks of axes, rotation centre, scaling and projector response.
3. Full-angle reconstruction of a few development slices in stored units, with
   paired reference/reconstruction views and fixed display scales.
4. Cross full/limited coverage with clean/noisy measurements. Evaluate clean
   held-out projection error; use voxel error only with matching volume references.
5. Establish deformation grouping before freezing splits. Begin ensemble
   disagreement diagnostics; seed variation is not calibrated posterior uncertainty.
   Test error ranking, interval coverage/width and quantitative-region estimates.

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

Next: paired full/limited-angle clean/noisy FBP controls with shared noise,
consistent plot scales and held-out clean projection metrics. Defer density-unit
claims and uncertainty calibration claims pending the appropriate references.

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
