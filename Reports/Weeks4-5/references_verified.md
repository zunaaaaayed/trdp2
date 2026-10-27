# External reference verification — Weeks 4–5 report

The report uses numbered inline `\cite{...}` citations and a self-contained
bibliography. Published sources were checked against original proceedings,
publisher records, author-hosted material or the originating project. Some full
publisher pages restrict automated access; the source/record checked is specified
below. The bibliography contains external methods, concepts and software only.
Experiment results are our measurements. The dataset provider is acknowledged
in the dataset section without a bibliography entry.

| Key | Source checked | What the citation supports |
|---|---|---|
| `kak` | [Kak and Slaney, author-hosted 1988 book](https://www.slaney.org/pct/) | Tomographic reconstruction and missing-angle information; authors/title/year verified |
| `r2` | [Official NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4f550cb7b30b59553e50cd08a9dbf068-Abstract-Conference.html) | Original Gaussian CT method, authors and DOI 10.52202/079017-1427 |
| `skimage` | [PeerJ publisher record](https://peerj.com/articles/453/) | Library identity and 2014 article, DOI 10.7717/peerj.453; cached publisher text checked, direct full-page fetch later returned 403 |
| `sart` | [Original paper scan](https://citeseerx.ist.psu.edu/document?doi=b581ca8aa1dfef284ae6babe432e039e04c1cd3c&repid=rep1&type=pdf) and [publisher search record](https://journals.sagepub.com/doi/10.1177/016173468400600107) | Andersen/Kak, title, 1984, volume 6, pages 81–94; publisher DOI 10.1177/016173468400600107 (original Elsevier DOI 10.1016/0161-7346(84)90008-7 is also encountered) |
| `scipy` | [Nature Methods publisher record](https://www.nature.com/articles/s41592-019-0686-2) | SciPy numerical algorithms, Virtanen et al., 2020, 17:261–272, DOI 10.1038/s41592-019-0686-2; publisher search/full-text extract checked |
| `pytorch` | [Official NeurIPS 2019 proceedings](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) | PyTorch paper identity and framework role; installed version comes from local environment inspection |
| `laplace` | [Official NeurIPS 2021 paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/a7c9585703d275249f30a088cebba0ad-Paper.pdf), Section 2 | Local Gaussian posterior/curvature construction; our Gauss–Newton and first-order approximations are identified separately, not represented as exact nonlinear inference |
| `gneiting` | [Author-hosted original JRSS B paper](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf), summary and introduction | Calibration and sharpness/concentration; 69(2):243–268, DOI 10.1111/j.1467-9868.2007.00587.x |
| `numpy` | [Nature publisher record](https://doi.org/10.1038/s41586-020-2649-2) | Harris et al., 2020, Nature 585:357–362; array programming library |
| `matplotlib` | [IEEE publisher record](https://doi.org/10.1109/MCSE.2007.55) | Hunter, 2007, Computing in Science & Engineering 9(3):90–95; plotting software |
| `rasmussen` | [Author-hosted Chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf), Section 2.1, and [Appendix A](https://gaussianprocess.org/gpml/chapters/RWA.pdf) | Rasmussen/Williams, MIT Press 2006; Gaussian linear-model posterior and conditioning identities |
| `lawson` | [SIAM book record](https://doi.org/10.1137/1.9781611971217) and [SciPy 1.10.1 NNLS documentation](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.optimize.nnls.html) | Lawson/Hanson, Solving Least Squares Problems, SIAM 1995 reprint; nonnegative least squares |
| `golub` | [Johns Hopkins University Press book record](https://www.press.jhu.edu/books/title/10678/matrix-computations) | Golub/Van Loan, Matrix Computations, fourth edition, 2013; matrix factorization reference |

Compilation support was checked against [Overleaf's XeLaTeX guide](https://www.overleaf.com/learn/latex/XeLaTeX),
which demonstrates Times New Roman by name. The local PDF embeds real Times New
Roman fonts. Font binaries are not redistributed in the Overleaf ZIP.
