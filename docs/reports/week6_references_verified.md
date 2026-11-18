# Week 6 report: primary-source citation checks

Checked on 29 October 2026. The report's numerical results come from saved local
experiments; no cited paper is presented as the source of those measurements.
Python libraries and our own earlier reports are excluded from the bibliography.

| Key | Primary source | Supports |
|---|---|---|
| kak | [Kak and Slaney, author-hosted book](https://www.slaney.org/pct/) | Limited-angle CT, Fourier/backprojection context and parallel-beam detector reversal; IEEE Press1988 edition verified. |
| r2 | [Official NeurIPS2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4f550cb7b30b59553e50cd08a9dbf068-Abstract-Conference.html) | Gaussian CT representation and originating codebase; six authors/title/year/DOI10.52202/079017-1427 verified. Our analytic parallel-beam studies extend that representation. |
| laplace | [Laplace Redux, original paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/a7c9585703d275249f30a088cebba0ad-Paper.pdf), sections1–2 | MAP/local Gaussian approximation and generalized Gauss–Newton curvature. Our bounded Gaussian likelihood/prior target is an explicit specialization, not an exact posterior supplied by the paper. |
| psis | [Vehtari et al., JMLR25(72),2024](https://jmlr.org/papers/volume25/19-556/19-556.pdf) | Ordinary self-normalized importance ratios, weight concentration and ESS diagnostics. The report explicitly states that our audit did not implement Pareto smoothing. |
| mala | [Roberts and Tweedie, original Bernoulli paper](https://projecteuclid.org/journals/bernoulli/volume-2/issue-4/Exponential-convergence-of-Langevin-distributions-and-their-discrete-approximations/bj/1178291835.pdf) | Metropolis-adjusted Langevin sampling;1996,2(4):341–363, DOI10.2307/3318418 verified. Our bounded affine-coordinate implementation is a feasibility experiment. |
| rhat | [Vehtari et al., primary manuscript](https://arxiv.org/abs/1903.08008) | Rank normalization/folding and local ESS convergence diagnostics; published2021 in Bayesian Analysis16(2):667–718, DOI10.1214/20-BA1221. |
| ding | [Ding et al., original Scientific Reports paper](https://www.nature.com/articles/s41598-019-49267-x) | Supervised sinogram completion and separation from image-domain cleanup;2019,9:12803, four authors verified. The report identifies our masked MSE and tiny deterministic model as adaptations, not their GAN architecture. |
| ddw | [Wiedemann and Heckel, original paper](https://www.nature.com/articles/s41467-024-51438-y) | Learned missing-wedge restoration and the risk of untrustworthy hidden structures;2024, Nature Communications15:8255. Our method is explicitly supervised, unlike DeepDeWedge. |
| ensembles | [Lakshminarayanan et al., NeurIPS2017 paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf) | Independently trained ensembles and predictive variation. Our sample SD from three MSE-trained point predictors is explicitly a heuristic risk score; we do not claim their learned-variance model or calibration results. |

Direct Nature fetches redirected to an inaccessible identity service; the original
publisher pages' indexed full method text and publication metadata were inspected.
All nine references have inline citations in the self-contained TeX bibliography.
No quotation or novelty claim is used.
