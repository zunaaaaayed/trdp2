# Matched-input restoration pilot v2 — frozen 28 October 2026

This amends the preparation-only v1, retained as history. Both learned arms and
all initializers/refinement receive exactly the inherited sparse measured rays.
The machine-readable design is `../protocols/completion_matched_protocol_v2.json`.
Roles and the16/4/4-file first subset are inherited unchanged from manifest v1.
Only16 training and4 validation files are opened in this phase; reserved outcomes
remain unavailable. Grouping by numeric-ID proximity remains provisional.

## Sampling and noise

Input and target:25 angles x62 detector samples for each of8 slices. Exact indices
come from the saved worm-shape protocol: angles0,4,...,96 of the100-angle TIFF,
detectors2,7,...,307 of310, z=[224,232,240,248,256,264,272,280]. Angles are0:7.2:172.8
degrees, converted to radians for the Gaussian operator. Masks include start and
exclude end modulo180 with1e-10 degree tolerance. This is point subsampling, not
bin averaging. No other measured rays enter learning or Gaussian initialization.

Inherited detector centre convention: u=(pixel_index+0.5-155)*2/310,
v=(z-mean(selected_z))*2/310. The62 u coordinates reverse exactly under u→-u.
This verifies internal consistency, not the unresolved simulator convention or
physical projection units. Sparse normalization is fixed to6180.275452553537
from the historical worm_004 development slab; do not reuse historical per-case
normalization. Therefore old worm errors are contextual, not paired comparisons.

Generate an underlying100x310 independent Gaussian noise field (SD0.05 normalized)
per file/z/epoch with the frozen SeedSequence rule, then select the sparse indices.
All masks and arms share those selected noise values. Missing clean targets never
enter the measured-input constructor. Save float64 measured values; neural inputs
use float32 and every completed array restores the exact float64 acquired samples.
The same restored arrays supply Gaussian refinement. Input arrays/indices/masks,
source hashes and seed rules are recorded in the run artifacts.

## Context and architecture

Keep the original5809-parameter CNN:7→16→16→16→1,3x3 kernels, dilation1/2/4/8,
ReLU after the first3 layers. Nominal angular receptive field31, radius15 rows.
On25 rows this covers the intended missing regions even at finite boundaries.
A Boolean layer-by-layer path check tests all missing rows for every familiar and
shift mask. A gradient check at declared initial weights tests centre-detector
outputs at all missing rows for sensitivity to measured values, holding mask and
geometry fixed. These establish connectivity and tested sensitivity, not useful
learned dependence. The same check detects the inadequate100-row/90° design.

Zero padding, no ordinary angular circular padding. Interpolation uses angular
extension with detector reversal: p(theta+180,u)=p(theta,-u). No wrap is introduced
in the CNN. No extra neighbors or larger network is added.

Geometry arm channels: measured values, mask, sin(theta),cos(theta),sin(start),
cos(start),span/180. Mask-only arm has identical nominal architecture/initial
weights, with the final5 channels zero; their first-layer weights have no data
signal, so effective conditioning differs despite matched nominal capacity.

## Training, selection and decision (fixed before outcomes)

Train starts0/36/72/108 crossed with spans120/144; exact sparse counts are recorded
by context checks. One seed2026102803, Adam0.001, batch8, maximum20 epochs and1200
seconds per arm. Time cap includes training and familiar validation after loading;
stop new updates15 seconds before it to allow the final validation/checkpoint.
Record actual time, completed epochs/updates and PyTorch peak memory. Both arms
share initialization seed, batch ordering, noise and validation observations.
CPU minibatch finite-gradient smoke and isolation/context tests precede CUDA use.

Loss: missing-ray clean normalized MSE. Checkpoint: lowest mean over validation
files of mean across the8 familiar masks of missing-ray relative L2, pooling the8
slices within each file/mask. First exact tie wins. Also report MSE, each file/mask,
measured preservation and inference cost. Per-slice samples are not independent
organisms. Neither withheld final data nor unseen masks select checkpoints.

Choose the learned arm by the same familiar criterion (mask arm wins exact tie).
Proceed to one Gaussian integration case only if mean relative L2 improves by≥5%
over interpolation, at least3/4 validation files improve, and mean MSE is no worse.
These are pragmatic pilot progression criteria, not significance/calibration tests.
If the rule fails, retain results and stop integration/architecture expansion.

Save checkpoint hashes and selection.json before evaluating validation-only
shifts: starts18/90 at span120; start0 at span90; starts18/90 at span90. No retuning
after these results. This measures acquisition shift on validation deformations;
reserved-deformation transfer remains a later frozen assessment. Retain zero-fill
and interpolation and common image/error scales for all arms.

## Conditional Gaussian integration

If the gate passes, use worm_153_gt (first validation file), start0/span120 and
all8 planes. Three arms: acquired-only dictionary, interpolation dictionary,
selected learned completion dictionary.16 Gaussians,4 starts0/81/193/421,
450 evaluations, same acquired arrays/operator/noise0.05/bounds/density prior.
Each arm uses the **acquired-only** geometry-prior reference. Align component
labels by centre-distance Hungarian matching before using shared seed perturbations;
noise in coordinates is paired but distinct initial scenes are not identical starts.
Predicted rays enter initialization only. Save all fits/failures; choose by the
existing acquired-objective/converged-candidate rule. Record scale-aware local
adequacy as diagnostic, not retroactive certification of the worm model.

One baseline case/start provides runtime estimate before the remaining11 fits;
maximum600 CPU seconds for this small integration feasibility, checked between
fits. No automatic budget extension. The full reserved evaluation grid is not run
until its adapter, procedure and compute budget are recorded separately.

These experiments concern deterministic restoration and initialization. They
establish neither bounded-posterior inference nor physical-density accuracy.
