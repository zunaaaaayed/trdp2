# Missing-wedge papers: implications for TRDP2

Review date: 27 October 2026. This is a literature assessment and a proposed
experiment plan, not new reconstruction or calibration evidence.

We read all three supplied PDFs, including methods, figures, tables and
limitations. The UsiNet supplement and official training notebooks were also
checked. The older graphify index contains historical chest-CT context but none
of these papers or the current worm work; current experiment records, not that
index, establish the project status below.

## Main conclusion

The papers provide useful directions, but none verifies calibrated uncertainty
for our nonlinear 3D Gaussian worm model. The highest-priority action is a
well-solved density subproblem followed by controlled geometry optimization.
Sinogram completion is a plausible subsequent initialization experiment. A large
GAN or unfolded-network training run is not justified by these papers alone.

| Paper | Useful insight | Applicability to our current data | Priority |
|---|---|---|---|
| Ding et al. (2019), supervised two-domain reconstruction | Projection completion and image cleanup address different errors | We have complete projection targets for completion; we lack voxel targets for the image-cleanup stage | Later, small completion-only pilot |
| Yao et al. (2024), UsiNet | Missing-angle patterns can be learned without complete training sinograms under population/orientation assumptions | We have no voxel truth, but diversity of visible features/orientations across our deformations is unverified | Conditional follow-up; do not copy blindly |
| Le et al. (2024), unfolded proximal networks | Operator structure, adjoint matching and controlled steps matter for stable optimization | Directly informative for fixed-basis density optimization; guarantees do not cover freely moving Gaussian geometry | Immediate optimization control |

## Evidence we need to explain

The final worm controls reuse one sparse development acquisition and one noise
realization. With 16 components and budget 450, acquired/withheld relative errors
are 11.20%/19.84%; with 32 components they are 8.37%/23.81%. Only two of the 16 new
fits report solver success, and all four settings fail multistart stability.
These errors are not directly comparable to the earlier contiguous 1,500-Gaussian
slab, which used different sampling.

Our earlier density-only posterior has predictive coverage of 30.66%/24.17% at
120°/90°, far below its nominal 95% target. Matched small phantoms support joint
Gaussian-centre/shape uncertainty, while missing-component controls show that
local covariance does not explain absent structure. None of the reviewed papers
removes those scientific concerns by itself.

Sources: [final worm controls](../figures/worm_uncertainty/worm_fit_controls_v1/results.json),
[conditional density results](../figures/worm_uncertainty/conditional_density_v1/results.json),
and [experiment history](../../experiments/worm_uncertainty/README.md).

## 1. Ding et al.: supervised sinogram completion and image cleanup

**Reference.** G. Ding, Y. Liu, R. Zhang and H. L. Xin, *A joint deep learning
model to recover information and reduce artifacts in missing-wedge sinograms for
electron tomography and beyond*. Scientific Reports 9, 12803 (2019).
[Publisher and DOI](https://doi.org/10.1038/s41598-019-49267-x),
[primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6728317/).
Local copy: [articlemissingwedge.pdf](../../papers/articlemissingwedge.pdf), 13 pages.

The pipeline is missing-angle sinogram → supervised RRDB-based GAN completion →
weighted backprojection → supervised U-net GAN artifact removal. The stages are
trained separately; the main paper does not establish an end-to-end physical
projection loss (PDF pp.3–4, Figs.3–4). Training uses 50,000 images and 5,000
validation images drawn from geometric, natural and medical image libraries.
Completed sinogram targets come from known images; the second stage also needs
original-image targets (pp.3–5 and 7–9).

The strongest positive insight is the domain ablation (p.11, Fig.13): image-only
cleanup blurs some boundaries and recovers intensities poorly; sinogram-only
completion leaves residual reconstruction artifacts. This supports evaluating
projection accuracy, shape and density separately. A visually sharper image is
not enough for quantitative imaging.

The main missing-wedge demonstration removes 45°, equivalent to roughly 135°
acquired coverage in our 180° parallel-beam convention. It does not establish
performance at our harder 120° and 90° settings. Table 4 (p.9) reports joint-model
PSNR 27.46 dB and SSIM 0.9532, versus missing-angle SART 18.55 dB/0.3124. However,
complete-angle SART has higher PSNR (30.17 dB) and lower RMSE than the joint model
(pp.9–10, Tables 4–5). The measures disagree, reinforcing the need for density
error alongside perceptual/structural scores. Experimental examples on nanorods
and cathode material are qualitative (p.11, Fig.14), without matching density truth
or uncertainty calibration.

**What we can use.** Our full-angle clean projections already provide supervised
sinogram-completion labels after retrospective angular masking. Missing voxel
truth does not prevent this first stage. It does prevent directly reproducing
their supervised voxel/image cleanup on the worm. A full-angle FBP reconstruction
would be a surrogate target, not exact voxel truth.

**What we should not assume.** GAN realism does not establish true missing-view
structure. The main paper does not demonstrate exact preservation of acquired
rays, a final reprojection constraint, or posterior coverage. Source-image noise
augmentation is not a measured projection-noise robustness study. Random image
splits also do not establish independence of all medical sources. This is a
limitation in reported grouping, not evidence of proven leakage.

Code and data are listed as available on reasonable request (p.12); no official
public implementation was verified. The supplementary PDF could not be obtained
during this review, so detailed supplementary losses and sampling are not claimed
as independently checked.

## 2. UsiNet: unsupervised sinogram inpainting

**Reference.** L. Yao, Z. Lyu, J. Li and Q. Chen, *No ground truth needed:
unsupervised sinogram inpainting for nanoparticle electron tomography (UsiNet)
to correct missing wedges*. npj Computational Materials 10, 28 (2024).
[Publisher and DOI](https://doi.org/10.1038/s41524-024-01204-x),
[official code](https://github.com/chenlabUIUC/UsiNet),
[supplement](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41524-024-01204-x/MediaObjects/41524_2024_1204_MOESM1_ESM.pdf).
Local copy: [missingwedge2.pdf](../../papers/missingwedge2.pdf), 14 pages.

The U-Net predicts a complete sinogram from an incomplete one. A second random
missing band is applied to that prediction, and the same network is trained to
recover its first prediction. That first prediction supplies a detached
pseudo-target, refreshed over training (pp.3–4, Fig.2; p.12, Methods).

Crucially, the paper explicitly requires similar features to appear unmasked in
other particles. Collectively, the particles need sufficiently varied out-of-plane
orientations; otherwise some features remain systematically absent (p.3,
Overview). Angular mirror-extension and shifting relocate the masks using the
parallel-beam relation S(theta+180°,s)=S(theta,-s) (p.12). They rearrange existing
information rather than acquire new directions.

The official experimental notebook uses prediction → random remasking → fitting
back to the prediction. It does not explicitly restore the measured pixels or
add a measured-data/reprojection term in that loop. Consequently, small
self-training loss alone cannot certify missing-angle accuracy: a constant
predictor is also a mathematical fixed point. This observation does not claim
that the published runs collapsed. It motivates independent evaluation and
measured-ray preservation in any adaptation we build.
[Example notebook](https://github.com/chenlabUIUC/UsiNet/blob/main/experimental_data/trainingExample.ipynb)
(checked 27 October 2026; experimental notebook Git blob
`42a33de4c177f62762317ccf7438c75a8399b316`, simulated notebook blob
`4a80aa1e6a7eb2c39b6f8df4f80d2d386760f0b1`; these are file-blob IDs,
not repository commit IDs).

The paper tests acquired spans ±60° and ±45° (120° and 90°), varied polygon and
polyhedron shapes, and Poisson noise (Fig.4). This makes its geometry range more
relevant than Ding et al.'s main 45°-missing example. Supplementary Notes 2–3
report failures with very small particle populations or ±35° coverage, recommend
at least 20 particles for their tested setting and generally at most 50% missing
contents. These are empirical limits for that design, not universal thresholds.

The 126 experimental nanoparticles serve as both training and testing data
(p.8). The official repository separately documents simulated train/validation/
test particle splits. Three-dimensional data generate many correlated slices;
these must not be counted as independent objects. Neighboring-slice input reduces
visible inter-slice waviness under noise, with little change to aggregate metrics
(pp.6–7). Ground-truth-tuned binarization in simulated evaluation (p.13) is not
available for our density measurements.

**Positive insight.** A sinogram-domain model can use cross-slice/shape redundancy
without requiring voxel labels. Our clean full-angle projections additionally
allow a simpler supervised completion test before attempting self-training.
Small neighboring-slice inputs could later be tested for axial consistency.

**Transfer limits.** Our 144 files are deformations of one worm, not 144 distinct
randomly oriented particles. The necessary diversity of visible internal features
has not been established. Randomly shifting masks on a single object does not
supply its missing structure. The paper validates mostly nanoparticle morphology;
it does not establish mass-density accuracy, calibrated uncertainty or reliable
Gaussian optimization. Intensity rescaling/clipping in the example workflow
should not be copied into quantitative-density experiments without a separate
scaling analysis.

## 3. Unfolded proximal networks: structured denoising and optimization

**Reference.** H. T. V. Le, A. Repetti and N. Pustelnik, *Unfolded proximal
neural networks for robust image Gaussian denoising*, arXiv:2308.03139v2
(21 August 2024). [Record](https://arxiv.org/abs/2308.03139),
[reviewed version](https://arxiv.org/pdf/2308.03139v2).
Local copy: [unfolded-PNNs.pdf](../../papers/unfolded-PNNs.pdf), 30 pages including
annex and references. PNN means **proximal neural network**, not probabilistic
neural network.

The networks unfold dual forward–backward and Chambolle–Pock iterations into
finite learned layers. Fixed proximal/projector operations are combined with
learned linear analysis/synthesis operators; inertial updates create particular
skip connections (Sections 2–3, pp.3–10). Their normalized-operator strategy ties
an operator to its adjoint within a layer and constrains step sizes. This is a
useful principle for controlling numerical sensitivity.

However, the convergence theorems require a fixed operator across iterations,
matched adjoints, a convex problem and appropriate steps. The PnP theorem also
requires a linear measurement operator and infinitely many inner layers
(Corollary 1; Theorem 4.1, p.20). Trained finite-depth networks may vary operators
by layer, so the theorem cannot simply be applied to them. It certainly does not
establish global convergence when our Gaussian centres and shapes move.

Reported sensitivity values are Jacobian-norm estimates on sampled test images,
not certified global Lipschitz constants (Eq.24, pp.16–17). Figure 6 gives means
about 1.04 and 1.02 for two normalized noninertial/strongly-convex variants, but
3.80 for the inertial DDiFB variant. Acceleration is not automatically more stable.

The training uses paired clean/noisy RGB images from BSDS/ImageNet; the main
variable-noise setting uses 50,000 training images and noise SD in [0,0.1] in its
image scale (p.12). Denoising and deblurring are tested, not missing-angle CT.
The paper is not uniformly best in PSNR: Table 3 reports 33.6 dB for DScCP-LNO,
34.7 dB for DRUnet and 34.0 dB for BM3D. Its benefits include smaller models and
better empirical robustness in some settings.

**What transfers now.** A fixed-geometry density fit is linear in amplitudes and
can be solved as a convex constrained problem with our existing quadratic prior.
Norm-controlled steps and an adjoint test provide useful checks if an iterative
solver is used. There is no need to train a PNN to obtain that reference.

**What does not transfer.** Freely moving centres/shapes make the reconstruction
nonconvex. Better convergence cannot restore missing components or establish
correct missing-view structure. The paper provides point estimates, not posterior
covariances or 95% coverage guarantees. A denoiser would also change residual
noise statistics and require a new likelihood/calibration assessment.

## A concrete next comparison

### First: a fixed-geometry density reference with verified optimality

Keep the current measured data, normalization, positive bounds, density prior
and shared geometry prior. For each chosen geometry snapshot, solve

\[
\min_{0.001\le a_k\le100}
\frac{\|B_Aa-y_A\|^2}{2\sigma^2}+\frac{\|a\|^2}{2\tau^2},
\qquad \sigma=0.05,\quad\tau=10.
\]

This is bounded ridge least squares. It can be passed to `scipy.optimize.lsq_linear`
with the augmented matrix `[B_A/sigma; I/tau]` and target `[y_A/sigma; 0]`. The
finite prior gives Hessian `B_A.T @ B_A / sigma**2 + I/tau**2`, bounded below by
`0.01 I`; hence the conditional problem has a unique minimizer even when views
are missing. This is a mathematical consequence of the existing quadratic
prior, not a result already measured on our data or a new result from the PNN paper.

Solve the numerical problem to a verified optimality/KKT tolerance; the objective
is exact for the fixed geometry, while the floating-point solve is approximate.
Compare identical geometry snapshots and the same summed objective. Record data
and prior terms, solver optimality, active bounds, projected-gradient/KKT residual,
conditioning and acquired error. The optimized density objective should not exceed
the original candidate's at that same geometry. Different geometry snapshots are
different conditional problems; agreement of their densities is not expected.

Then compare direct joint optimization with density → centre → shape stages,
re-solving the density block after accepted geometry steps under the same fixed
objective/prior. Count stage and total evaluations. Keep all multistart candidates,
active-bound and agreement checks. Monotone loss reduction alone does not establish
global convergence or missing-view accuracy. Withheld values must not influence
stage acceptance, stopping or candidate selection.

### Then: a small projection-completion initialization study

If the preceding fitting is stable, compare measured-only initialization with a
simple angle-interpolation completion and, later, a small supervised completion
model. Retain ordinary Gaussian prediction as another reference where practical.
A separately labelled **oracle completion initialization on development cases**
can test whether better missing-view information would help at all; it is not a
deployable method, guaranteed upper bound or an independent test result.

For learned completion, our complete training sinograms provide labels after
retrospective masking. Split deformation groups before slicing, augmentation or
noise generation. Use development/validation cases to choose masks, architecture,
stopping and any completion strength; lock those choices before reserved-case
scoring. Complete projections from reserved cases, including their withheld rays,
must remain excluded from training targets, normalization, prior construction,
oracle initialization and model selection. Adjacent slices and related
deformation windows must stay together or
receive suitable guards. This still evaluates within-one-worm transfer, not
independent biological generalization.

For the first adaptation, preserve acquired values and use completed projections
only to build an initializer. Refine the Gaussian model using **acquired rays
only** with the same measured-data-derived prior. Treating generated rays as
independent observations would double-count prior predictions and can make
uncertainty even more overconfident. Keeping the prior fixed when initializers
change avoids silently changing the inference problem.

Score withheld clean-ray error, convergence and acquired fit; on known phantoms,
also score voxel/edge/interior error, integrated density, coverage and interval
width. Worm voxel-density error remains unavailable without matched truth. Repeat
noise draws when assessing variability or calibration, and keep the full
preprocessing/initialization pipeline inside those repeats. Noise variability
or a sharper reconstruction alone is not calibrated uncertainty.

## Scope of this review

No reconstruction code or protocols were changed, no network was trained, and no
GPU experiment was launched. The proposed density and completion comparisons
have not been run. Paper results are not directly comparable to our error tables.
Original PDFs and completed experiments are preserved.

| Local PDF | Pages | SHA-256 |
|---|---:|---|
| `articlemissingwedge.pdf` | 13 | `bbaedbb7f7806d60e61ecff5f22bc0534e75ccf74c320f403fdec84eade0452e` |
| `missingwedge2.pdf` | 14 | `880a884d6b68a15d70f64c0445719181c51435824758f40550b5d7c48072d938` |
| `unfolded-PNNs.pdf` | 30 | `56a26ff77169654d0b5887bfe6a1c3875a1d120cabe5a45f0070e4070fba3e82` |
