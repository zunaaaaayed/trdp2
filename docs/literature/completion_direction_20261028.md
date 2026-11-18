# Completion direction — primary-method follow-up, 28 October 2026

This supplements the older optimization-first review; those optimization controls
are now complete. None of these papers establishes calibrated Gaussian uncertainty
or novelty for our proposed combination. Primary methods and limitations inform
preparation; we are not reproducing their architectures in this session.

- **DeepDeWedge** uses independently noisy FBP pairs, random 3D rotations and
  additional Fourier wedges; its masked loss and iterative input updates train a
  3D U-Net. This motivates explicit corruption masks. Our noiseless complete
  projection targets instead permit a simpler supervised control. Rotation-based
  information sharing can miss hidden structures; restoration is not posterior
  calibration. [Methods, equations2–3](https://pmc.ncbi.nlm.nih.gov/articles/PMC11420219/).
- **ICE-TIDE** fits an implicit density together with shifts, in-plane rotations
  and local acquisition deformations. Its simplified model omits CTF and uses
  a simple noise model; the discussion identifies these limitations. Different
  worm files do not demonstrate motion during a scan. Test deliberate acquisition
  mismatch before adding those nuisance parameters. This is not the unfolded
  proximal-network paper. [Methods II and discussion IV](https://arxiv.org/html/2403.02182v2).
- **ICECREAM** combines data fidelity with equivariant self-supervision and uses
  two network applications during inference. Its discussion explicitly limits
  rotational-invariance assumptions and distinguishes FSC consistency from
  reconstruction resolution. Transfer/warm-start comparisons are useful, but our
  elongated-worm orientation distribution is unverified. [Methods2.4–2.7,
  discussion4 and appendixA](https://doi.org/10.1107/S2059798326001622).
- **UsiNet** completes sinograms using predictions as pseudo-targets under further
  masking. It requires sufficient diversity of visible structures. The available
  synthetic clean worm projections allow supervision without assuming that all
  hidden structures appear elsewhere. [Primary paper](https://doi.org/10.1038/s41524-024-01204-x).
- **Ding et al.** separate projection completion from image cleanup. We retain
  the decomposition, start small, and omit voxel-supervised cleanup because voxel
  truth is absent. Deterministic restoration supplies no calibrated posterior.
  [Primary paper](https://doi.org/10.1038/s41598-019-49267-x).

The supplied UsiNet/Ding PDFs, their detailed limitations and the unfolded-proximal
paper remain documented in [the earlier review](missing_wedge_papers_20261027.md).
Publisher browser access failed for several links; DeepDeWedge was read through
PMC, ICE-TIDE through arXiv, and ICECREAM through the publisher's indexed full text.
No cryo-ET-specific CTF or noise mechanism is imported into the synthetic worm
likelihood. The concrete design is the
[completion protocol](../../experiments/worm_uncertainty/notes/COMPLETION_PROTOCOL.md).
