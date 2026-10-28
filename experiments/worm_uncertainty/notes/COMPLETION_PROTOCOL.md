# Projection-completion pilot v1 — frozen preparation, 28 October 2026

Question: does supervised completion improve withheld projection prediction after
Gaussian refinement using only actual measurements? Restoration, inference and
robustness are separate outcomes. No neural training has run under this protocol.

## Data and separation

Use the supplied synthetic parallel-beam sinograms (510 axial planes, 100 angles
0:1.8:178.2 degrees, 310 detector pixels, 1 micrometre sampling). These are 144
poses of one organism. Complete clean projections supply supervised targets;
there are no corresponding original deformed voxel targets. No physical density
or independent-specimen transfer claim is supported.

`../protocols/completion_manifest_v1.json` freezes every filename, intake SHA-256, population
role, sampling and scale. Numerically sort IDs, group consecutive sets of twelve,
and allocate blocks 0,1,3,5,6,8,10,11 to training; 2,7 to validation; 4,9 to reserved
assessment (96/24/24 files). This keeps nearby IDs together but is only a proxy:
ID ordering is not verified deformation genealogy. Strong relatives may still
cross blocks. The entire set represents the same underlying organism. Obtain
simulation grouping information before stronger generalization claims.

The previous intake decoded all files and made overview images/statistics. Thus
"reserved" means prospectively reserved for experiment outcomes, not never-seen
raw data. The historical development file worm_004 stays in training. No test
reconstruction outcome informed this split. This session opens only its slice256.

For the bounded first training pilot use the first two numeric IDs in each
training block (16 files), first two in each validation block (4 files), and first
two in each reserved block (4 files) for the eventual frozen assessment. All other
files retain their roles, never recycle final cases after seeing poor results.
All eight z indices [224,232,240,248,256,264,272,280], crops and neighboring slices
from each file inherit its population role. IDs are selected by manifest order,
not images, foreground quality or reconstruction error.

## Acquisition and noise splits

Train and validate on starts [0,36,72,108] degrees crossed with spans [120,144].
Use start-inclusive/end-exclusive masks modulo180; e.g.120 gives67 of100 views.
Validation draws use the same acquisition set, different deformations/noise.
Final assessment reports four separate groups:

1. Reserved deformation, familiar start0/span120.
2. Reserved deformation, unseen starts18 and90/span120.
3. Reserved deformation, familiar start0/unseen span90.
4. Reserved deformation plus unseen starts18 and90/span90.

Also evaluate unseen masks on the four validation files after model selection,
without changing the model; this distinguishes acquisition-only from combined
shift. These results are evaluation-only, not another tuning set.

Normalization is **6180.275452553537**, frozen RMS from the old worm_004 full-angle
development slab in `../protocols/slab_protocol_v1.json`. It is a model intensity scale, not a
physical calibration. Never derive a per-test normalization from clean targets.
Initial noise is independent Gaussian SD0.05 in this normalized scale. Generate a
full100x310 noise field then mask; sharing a seed across masks shares noise on
common rays. Training seed identities use SeedSequence([2026102803,file numeric
ID,z,epoch]); validation [2026102804,ID,z,0]; final [2026102805,ID,z,replicate],
replicates0,1,2. Noise mismatch (0.025/0.10), forward mismatch and neighbors are
later separately frozen experiments, not additions to the first comparison.

## Model and comparisons

Deterministic four-layer 2D convolutional model: channels7→16→16→16→1,
3x3 kernels, dilations1,2,4,8, ReLU after the first three layers, linear output.
Capacity 5,809 trainable parameters including biases. Zero padding at boundaries
(no incorrect ordinary angular circular padding). Inputs: noisy measured values,
binary mask, sin(theta), cos(theta), sin(start), cos(start), span/180. Missing
entries are zero only alongside the explicit mask. Record architecture/capacity
when instantiated; the data infrastructure does not yet instantiate this model.

Train with Adam lr0.001, batch8, at most20 epochs or20 GPU minutes per arm,
seed2026102803, MSE on missing clean projections. Select the checkpoint with lowest
mean per-deformation missing-projection relative L2 over the fixed validation
masks/slices. Include unweighted normalized MSE so background-heavy relative
scores cannot silently dominate selection. The selection criterion remains the
relative L2 criterion, not whichever metric looks better. Save all epoch scores,
fixed final checkpoint on budget exit, elapsed time and peak allocation. Initial
pilot has one training seed; no claim of training-seed robustness.

Primary completion arms: zero-fill (mask always retained), detector-reversal-aware
angular linear interpolation, learned values+mask, learned values+mask+geometry.
For the 2-channel arm preserve7-channel capacity by zeroing geometry channels.
For a mask-conditioning ablation, feed the same interpolation-filled input to
both models and retain the mask in the data interface; one network receives mask,
the other has its mask channel zeroed. Both retain identical geometry channels,
capacity and budgets. This avoids confusing unknown zeros with physical zeros.
This fourth trained arm is optional after the primary pilot; no large sweep now.
Keep measured values unchanged in the completion used for initialization.

The parallel-beam relation is p(theta+180,u)=p(theta,-u), not ordinary wraparound.
The implemented interpolation comparator reverses detector coordinates on both
angular extensions; its convention is tested against our analytic projector.
The simulator's detector-centre convention still needs confirmation before
claiming this comparator is physically exact for the worm data.

## Gaussian initialization-only integration

Compare the existing acquired-only multiscale dictionary initializer with the
same dictionary fitted to each completed full angular array. Completion rays may
enter the initializer only. Both arms then refine against identical acquired
noisy rays, sigma, 16 components, bounds, prior, optimizer schedule and budget450.
Use the existing eight axial planes/25-angle/62-detector sparse sampling for both
arms; the completion model can use the supplied full sinogram but save this
information/cost difference explicitly. A stricter matched-input ablation must
also restrict completion inputs to the same25/62 sampling before attributing a
benefit to learning instead of extra acquired rays. This extra ablation needs a
separate frozen data resampling protocol and is a prerequisite for that claim.

Crucially the geometry-prior reference for **both** refinement arms is the
acquired-only baseline initializer, never each arm's own initialization. Otherwise
an alleged initialization comparison changes the prior too. Starts0,81,193,421,
perturbations and selection by acquired objective remain matched. Preserve all
failed numerical/multistart outcomes; no withheld start selection. Completion
is not an independent likelihood observation, even when its prediction is good.
The integration adapter is not implemented yet; add a test proving identical
refinement observations, prior reference, bounds and budgets before launching it.

## Assessment and launch boundary

Save completion error before refinement, acquired/withheld error after refinement,
failed-fit fraction, wall time and capacity. Plot input/target/completion/residual
with shared intensity ranges and lower-error-is-better captions. Report each
population/acquisition group and paired per-file differences; slices are not
independent specimens. Full-angle FBP may be a visualization reference, never
voxel truth. Do not add Bayesian coverage claims to this deterministic experiment.

Zero-shot application only in v1; no adaptation to reserved scans. Any later
acquired-only adaptation gets a new protocol, fixed steps and separate results.
No reserved targets are exposed by the development loader. No final evaluation,
training sweep or GPU run is authorized automatically by this preparation.
Before launch: implement the small trainer and initialization adapter, pass matched
objective/input tests, confirm available GPU, then review the bounded training
commands. Budget estimate: <=20 GPU minutes per primary learned arm (two arms),
plus CPU preparation and matched refinement; benchmark refinement cost on one
validation case before authorizing the complete evaluation grid.
