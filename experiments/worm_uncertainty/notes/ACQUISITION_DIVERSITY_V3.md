# Acquisition diversity development experiment v3 — frozen before training

This continues the local checkout at `ebb4754`, with a clean starting tree. That
commit contains the completed v2 pilot; the earlier continuation note's uncommitted
status was stale. Historical metrics reproduce exactly from saved predictions.
The new work is authorized for bounded execution, without a commit or push.
No new sampler, importance correction or posterior-calibration run is planned.

## Question and analysis units

Does broader acquisition training improve missing projections beyond interpolation,
and does that improvement survive conversion to and refinement of Gaussian scenes?
A secondary conditional analysis asks whether three-model disagreement ranks
completion error better than angular distance to acquired rows. It is an empirical
error-risk score, not a posterior standard deviation or confidence interval.

The original 96 training /24 validation /24 reserved roles remain fixed. All 96
training files are used, with the same eight slices at 224:8:280. The two existing
12-file validation groups remain whole: numeric group 2 selects checkpoints and
development decisions; group 7 is opened for a one-time audit after settings freeze.
Both have prior exposure. Numeric blocks do not establish biological independence.
Reserved TIFFs are forbidden to all new development loaders.

## Acquired information and mask identity

The grid remains25 angles at 0:7.2:172.8 degrees and 62 point-sampled detector positions
(indices2:5:307); no detector binning. The original pixel-centre coordinates and
fixed scale6180.275452553537 remain. Noise SD 0.05 is applied to a full100×310 field
before subsampling. Training has 16 reproducible noise banks per slice and seed,
cycled by completed passes over the 768 deformation/slice examples. The same banks,
example order, initial weights and effective update/batch budgets are paired across
A/B/C. Selection and audit use distinct fixed noise seeds. Neural inputs are FP32;
completed sinograms restore the canonical FP64 noisy acquired values exactly.

Actual Boolean masks, serialized as25 uint8 entries, define SHA256 identities.
The [JSON protocol](../protocols/acquisition_diversity_v3.json) publishes every mask,
nominal start/span, acquired count and missing-wedge width. Training B/C use
counts 13/15/17/20 (nominal acquired spans90/108/120/144 degrees), balanced by count.
Start indices0,2,5,7,10,12,15,17,20,22 cover most of 180 degrees. The familiar A
family remains starts0/36/72/108 and acquired spans120/144 degrees.

Selection uses 8 familiar masks plus10 new masks at starts indexed1/6/11/16/21,
counts 13/15. Audit uses 8 familiar,6 new in-distribution masks,8 held-orientation
masks,6 masks at held counts 16/18, and 2 extrapolation masks with 10 measured angles
(72-degree acquisition). Orientation indices3/4/13/14 are absent from training;
counts 16/18 are wholly absent from training. Nonfamiliar selection/audit/train masks
are pairwise disjoint. Familiar masks intentionally recur to measure deformation
transfer at familiar acquisitions. Aliased nominal masks are not counted twice.
Historical v2 unseen masks already informed development and are not pristine tests.

## Arms, throughput and budget

A: original 5,809-parameter geometry CNN, restricted masks.
B: same architecture/channels, broad masks.
C: same nominal architecture, broad masks, five geometry channels zeroed as in v2
(720 first-layer weights receive no data gradient). B versus A is the acquisition
diversity comparison; B versus C is the explicit geometry comparison.

Three paired seeds:2026102901/2026102902/2026102903. Adam0.001, constant learning
rate. The loss averages squared missing-ray errors within each sample, dividing by
that sample's missing-ray count, then averages samples. No additional losses,
neighbouring slices, augmentations, full-resolution rays or architecture changes.

A training-only profile chooses the smallest batch within 95% of the best measured
throughput among16/32/64/128/256, subject to 4 GiB free-memory headroom. Batches contain
distinct genuine examples, not duplicates to occupy memory. The common update count
is at most4,000, rounded to 200, chosen from measured throughput to target at most240
seconds before validation overhead. It is written to `training_budget.json` before
primary training. FP32 is retained; AMP is unnecessary if the profile meets budget.
No per-update file I/O: only selected slices/noise are cached, with zero loader
workers. Full float64 TIFF volumes never enter GPU memory.

Validation occurs every 200 updates. Checkpoints minimize mean per-deformation
relative L2 over the 18 selection masks: first pool the 8 slices within each mask,
then average masks, then files equally. First checkpoint wins exact ties. Record
training loss, selection MSE/error, update count, throughput, allocated/reserved
memory and utilization. Per-seed cap390 s, arm cap1,200 s, three-arm cap3,600 s; no
automatic extension. A partial run is explicitly marked and cannot pass the gate.
OOM retries reduce microbatch size with gradient accumulation preserving effective
batch size, and are logged. The profile ceiling is600 s; analysis ceiling 600 s;
optional capacity ceiling 1,200 s; total GPU ceiling 6,000 s.

## Frozen decisions and audit

Primary narrow selection group: five new masks with 13 acquired angles. Every seed
of an advancing arm must satisfy all conditions: ≥5% relative improvement over
interpolation, positive paired improvement on≥8/12 files, and≤2 percentage points
absolute degradation on familiar masks. Select the passing arm with lowest seed-mean
narrow error (alphabetical tie). These are engineering gates, not significance tests.

An optional capacity experiment is considered only if B passes but narrow error
remains>35%, final-quarter training loss improves<1%, and training/selection per-ray
MSE differs<10%. If eligible, its architecture and budget must be frozen separately
before opening audit outcomes. Otherwise document the skip. Checkpoints, capacity
decision, risk and plotting rules are hashed in `analysis_freeze.json` before audit.

Report per-file/mask/seed L2 and MSE, means/medians/worst files, paired win fractions,
retained counts and orientation groups. No raywise significance or organism-population
intervals. Boundary missing rows are≤2 sparse angular steps from a measured row;
centre rows are≥3. Distance uses the circular angular geometry, not circular CNN
padding. Detector reversal at 180 degrees remains explicit in interpolation.

If eligible, use the selected arm's3 seeds as an ensemble with sample SD (ddof1).
Within each audit deformation compare absolute-error retention curves at 10/25/50/
75/100% with random ranking and angular-distance ranking. Report Spearman association
and high-error/low-disagreement failures. Do not fit calibration to audit targets.

Fixed figures: first audit deformation, first new90-degree audit mask, z256.
Worst-case panels use highest seed-mean relative L2 file/mask, z256, with common
scales. Fourier views show the detector-axis transform on polar frequency coordinates;
they are not a2D sinogram FFT mislabeled as an object's Fourier transform. FBP from
clean projections is a surrogate reconstruction, never simulator voxel truth.
Gaussian density panels are model reconstructions in unresolved normalized units.

## Gaussian comparison and historical repair

The old missing fit is learned/seed 421. A separate append-only 180-second run restores
this fit with the exact saved initializer, prior, observations, bounds and 450-evaluation
budget. The original 11-fit report remains intact. Separate finiteness, feasibility,
stationarity, optimizer termination and multistart agreement; thresholds are unchanged.

New integration is conditional on the completion gate. Predeclare the first and
last selection files, first narrow mask (start 7.2/span 90),16 components, seeds 0/81
for acquired-only/interpolation/selected-learned initialization. All arms share the
acquired-only geometry reference, measurements, noise, bounds, objective and 450-
evaluation cap; predictions enter initialization only. The selected learned input
is the 3-seed mean under a frozen rule. Use acquired-only accepted-candidate selection;
if none qualify, mark the lowest finite feasible acquired-objective candidate as
provisional. Start intersection is explicit; failed fits are retained. Up to 12 fits
plus the historical repair must stay within 1,800 CPU seconds, one thread, with 100 s
per-fit caps and a preflight based on the observed historical runtime. Report errors
before conversion, after Gaussian initialization and after measured-only refinement.

The historical learned fit already had successful optimizer termination but failed
two scale-aware components (gradient1.19e-6>1e-6; correction0.001236>0.001).
The other arms had no terminated fits and failed all stationarity components.
Completing the start count does not repair those scientific acceptance failures.

## Scope

One synthetic organism, provisional deformation genealogy, correlated slices,
unresolved path-length scaling and absent deformed voxel truth remain limitations.
The separate phantom posterior studies retain oracle priors and failed inference
checks. Better sinogram prediction does not imply accurate3D morphology, physical
density, generalization to another organism or calibrated uncertainty.

## Frozen throughput result and launch

The training-only profile completed in 102.72 s, including91.05 s reading/hashing the
96 permitted training TIFFs,2.99 s generating16 sparse noise banks, and 0.093 s initial
GPU transfer. Batch256 had the best measured throughput (1,774 examples/s), with
0.145 s per update and 340 MiB peak reserved VRAM in that profile. All tested batches
passed finite-gradient and batch-independence checks. The common primary budget
is **1,000 updates ×256 examples**, validation every 200 updates, three seeds per
arm. This is333.3 effective passes through768 deformation/slice examples, with
noise banks and masks varying under the frozen schedule. FP32 is sufficient.

The CPU dependency check confirms nonzero sensitivity of all missing-row centre-
detector predictions for all 72 unique masks at declared initial weights. This
extends the v2 context check to the new inventories; it does not establish accuracy.
The single historical repair completed in 52.55 s. All 4 starts per old arm now exist,
but the selected candidates, errors and failed gates are unchanged. The appended
report also separates initialization-projection error from completion/refinement.

Primary training uses the source archive saved in its own provenance directory.
An audit-output-only fix made before audit execution restores measured FP64 entries
after averaging seed predictions (averaging identical values can otherwise round)
and sets measured-entry disagreement to zero. It changes no primary training or
checkpoint calculation; the later audit archives its actual source separately.

## Post-audit reporting additions (29 October)

The frozen primary risk comparison used stable ordering for equal scores. After
seeing the audit, we added a clearly labeled sensitivity check that averages over
all orders within tied angular-distance scores. The primary results are preserved;
no checkpoint, mask, model or gate was changed. A diagnostic panel selects the
largest error among rays below their file's 25th disagreement percentile. This is
an outcome-selected failure illustration, not the prospectively fixed example.

An analytic companion reports clean-target MSE plus0.05² for an independent future
noisy ray under the declared Gaussian noise model. This is not measured noisy
coverage. The main restoration metrics and risk curves use latent clean targets.

Current reproduction code now writes and checks profile cache hashes automatically.
The executed initial profile had those same hashes saved as a separate pretraining
manifest. Executed source ZIPs remain unchanged, and verification checks each run
against its actual archive rather than pretending current source bytes are identical.
