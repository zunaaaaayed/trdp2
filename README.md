# TRDP2 — Limited-angle CT reconstruction generalization

TRDP2 is [Md Zunayedul Islam](https://github.com/zunaaaaayed)'s research project
on Gaussian-based CT reconstruction across limited-angle acquisition configurations
and independent volumes. This repository is maintained for TRDP2 research and
experiments; it is not the official R²-Gaussian project or an upstream contribution.

The reconstruction implementation originates from
[R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian) by Ruyi Zha and collaborators.
TRDP2 adds acquisition-controlled experiments, reproducibility checks, guarded
execution, evaluation, and research protocols. Original authorship, commit history,
and license notices are retained.

## Research objective

Develop and evaluate a reconstruction rule whose settings can be selected on
development data and then frozen across acquisition orientations, angular spans,
and independent volumes. Each scan has its own reconstruction optimization.

Geometry-dependent regularization is a **research hypothesis**. Neither its
benefit nor its novelty has been established. Ordinary-TV controls, boundary
preservation, independent-volume evaluation, and relevant prior work are essential
to assessing that hypothesis.

## Current dataset direction

As of 12 October 2026, the dataset supplied by co-supervisor **Ms. Anna-Louise Brun**
is the selected data source for subsequent work. The raw `sino_selected.zip`
archive and its checksum are retained locally; see the
[dataset handoff](docs/SUPERVISOR_DATASET.md) for intake status. The earlier LIDC-IDRI
expansion is deferred. Existing pilot and TV results remain evidence for their
original data convention and do not establish performance on the new dataset.

## Active work: worm uncertainty — 17 October 2026

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
- Six CPU tests pass, including withheld-view isolation. Missing views contribute
  zero on the full angle grid, retaining the original angular quadrature weights.

Seed-zero projection relative L2 errors against clean data:

| Span | Noise sigma / development RMS | Measured views | Withheld views |
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

Accepted outputs: `output/worm_uncertainty/sart_20261017_v2/`. The first attempt is
retained; v2 explicitly enforces circular support after each sweep to remove a
projector warning. Seven tests pass, including withheld-value isolation for SART.
Figure and input provenance: [plot records](docs/figures/worm_uncertainty/provenance.json).

After the paired FBP command above, reproduce with:

```bash
python -B experiments/worm_uncertainty/compare_sart.py \
  --input output/worm_uncertainty/paired_fbp_repeat \
  --output output/worm_uncertainty/sart_repeat
```

Next: evaluate uncertainty against appropriate references. Resolve simulator scaling and
orientation, group related deformations before freezing splits, and separate
noise sensitivity from reconstruction ambiguity. No GPU training is queued.

## Historical chest results and status

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

Execution requires an explicit `--execute`. Existing destinations are rejected;
exact checkpoint continuation is not verified. TIGRE forward projection is
isolated from live PyTorch CUDA tensors because its context reset caused a
segmentation fault in the original combined smoke process.

Data, checkpoints, logs, and compiled extensions are excluded from Git. Transfer
the current data and artifacts as well as this repository. The historical
7 October archive predates the completed TV controls and is not a current backup.
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
