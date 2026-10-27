# Papers reviewed for the worm CT project

Reviewed 27 October 2026. Full notes, evidence locations and proposed controlled
experiments are in the [literature assessment](../docs/literature/missing_wedge_papers_20261027.md).
The review does not constitute a new experiment or a validated method improvement.

| Local file | Paper | Main relevance |
|---|---|---|
| [articlemissingwedge.pdf](articlemissingwedge.pdf) | Ding et al., *A joint deep learning model to recover information and reduce artifacts in missing-wedge sinograms for electron tomography and beyond*, Scientific Reports (2019), [DOI](https://doi.org/10.1038/s41598-019-49267-x) | Supervised projection completion followed by image cleanup |
| [missingwedge2.pdf](missingwedge2.pdf) | Yao et al., *No ground truth needed: unsupervised sinogram inpainting for nanoparticle electron tomography (UsiNet) to correct missing wedges*, npj Computational Materials (2024), [DOI](https://doi.org/10.1038/s41524-024-01204-x) | Self-training completion and its population/orientation assumptions |
| [unfolded-PNNs.pdf](unfolded-PNNs.pdf) | Le et al., *Unfolded proximal neural networks for robust image Gaussian denoising*, [arXiv:2308.03139v2](https://arxiv.org/abs/2308.03139) | Structured optimization, adjoint matching and limits of denoising guarantees |

The immediate priority is a bounded fixed-geometry density reference followed by
controlled geometry stages. A learned completion model is a later initialization
comparison using our full-angle training sinograms; generated rays must remain
separate from measured evidence. None of these papers supplies calibrated
uncertainty for our Gaussian worm reconstruction.

The supplied PDFs are retained under their original filenames. Publication rights
remain with the respective authors/publishers; the notes are our own synthesis.
