# Reconstruction bridge preparation — 16 November 2026

Status: storage-blocked preparation, not a completed reconstruction experiment.
Starting branch experiment/worm-uncertainty, clean HEAD 26cf9d6.
The existing graph index has no worm_uncertainty entries; inspected source instead.
No repository AGENTS.md was found. Historical data and results remain intact.

## Storage gate

`quota -s` returned no information. `quota -v -s -f .` failed to contact RPC
inside the sandbox; outside it reported no matching quota-enabled mountpoint.
Neither establishes personal byte or file-count headroom. `df -h .` reports
99G total, 49G used, 51G available; this is filesystem information only.
No CREMI experiment is authorized by this preflight. Require independently
verified personal headroom of at least 1 GiB plus proposed writes, and verified
file-count capacity. New run artifacts, including temporary files, are capped
at 250 MiB. No run directory, cache, backup or new dependency environment created.
Only small source/document edits and synthetic CPU checks are performed.

Cleanup candidates (apparent bytes from du -sb; allocated usage may differ):

| Exact path under repository | Bytes | Purpose and disposition |
|---|---:|---|
| output/worm_uncertainty/alternative_reconstruction_sources/env | 6538932647 | Isolated ICECREAM environment; potentially recreatable, but currently used for checks. Dependency record is protocols/icecream_environment_20261110.txt; upstream SHA 39c891a55af3e1cd96ef0d585668b0c2af4c177d is preserved in code/protocol/results. Check all active jobs outside the sandbox before any future removal. |
| output/worm_uncertainty/icecream_pilot_v1 | 1447233268 | Historical inputs, checkpoints, failures, arrays and provenance. Retain; relocation only after approved destination and verified checksums. |

No external storage destination is verified or approved in this session. Request
an institution-approved project/archive allocation outside this personal quota;
verify its byte/inode quota and access policy, then checksum every copied artifact
before separately authorizing removal. Do not substitute /tmp for archival storage.
Nothing was deleted, moved, uploaded or reinstalled.

## Prospective decisions retained from the brief

Primary geometry: V3 25 angles 0:7.2:172.8 degrees, original detector indices
2:5:307 (62 point samples), eight axial planes 224:8:280. These are separate
slices, not a contiguous volume. Fixed normalization 6180.275452553537 and V3
full-field-before-subsampling noise SD 0.05. No dense-grid checkpoint transfer.
V3 Arm B best.pt files exist locally; full hash/architecture verification remains
pending. No new mask outcomes were inspected. All reserved files remain sealed.

Stage 2 budget: 600 seconds after geometry checks, acquired-only FBP/SART,
interpolation+FBP and saved B+FBP. Thresholds: at least 0.02 lower mean withheld
relative L2, majority of deformations improving, at most 0.01 worse acquired
relative L2. Compare interpolation and SART independently; SART competitiveness
requires within 0.02 withheld and 0.01 acquired. Aggregate slices/masks within
file before averaging files. Report raw completion and reprojection separately.

The actual case list, new Boolean-mask hashes, primary checkpoint rule, noise
identities, reconstruction grid and sparse-grid geometry tolerances are NOT yet
frozen. Consequently this document must not be used to launch or score Stage 2.
Complete those items and checkpoint provenance before any real-data outcome access.

## Implemented bounded work

core/differentiable_fbp.py reproduces the existing ramp filter, FFT padding,
angular weights and discrete transpose in PyTorch. Its dense backprojection
matrix is intended for small slice grids only. It is not a full-volume backend.
completion_loss supplies the parameterized missing-ray MSE plus optional L1 FBP
surrogate with identical acquired noise on both branches. Lambda zero bypasses
the auxiliary path. No lambda or training configuration has been selected.

The synthetic unit test declares float64 reference tolerance rtol 1e-11,
atol 1e-12, bitwise acquired copying, zero measured-prediction gradients and
finite nonzero missing-prediction gradients. This is an implementation smoke;
it does not validate physical sparse-grid geometry, generalization or density.

DuDoNet PDF and publisher HTML requests returned HTTP 403 in this session.
Its Figure 2 and full reconstruction-loss method still require primary-source
inspection before claiming a literature-complete design. User-specified surrogate
was implemented directly; no image-refinement CNN or metal protocol was added.
Reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Lin_DuDoNet_Dual_Domain_Network_for_CT_Metal_Artifact_Reduction_CVPR_2019_paper.pdf

## Status and resume

| Stage | Status |
|---|---|
| Workspace/storage inspection | Completed; personal quota unknown |
| Frozen comparison / complete geometry verification | Partial; sparse-grid calibration and manifest pending |
| No-training bridge | Blocked by quota; not executed |
| Conditional training | Gated off pending bridge results |
| Disagreement analysis | Gated off pending primary comparison |
| Differentiable FBP synthetic check | Implemented; see test output |

Next read-only command on CREMI:

```bash
cd ~/espaces/travail/trdp2/r2_gaussian
quota -v -s -A
```

If this still omits the work allocation, obtain byte and file-count quota from
CREMI support for the exact destination. There is deliberately no experiment
resume command yet: the comparison manifest and runner are unfinished. Do not
use an unverified alternate mount to bypass the storage gate.

Reproduce the bounded synthetic check without bytecode writes:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
output/worm_uncertainty/alternative_reconstruction_sources/env/bin/python \
-m unittest experiments.worm_uncertainty.tests.test_differentiable_fbp
```
