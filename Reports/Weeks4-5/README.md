# Combined Weeks 4–5 report

**TRDP2 Weeks 4–5 | 12–23 October 2026**

Report title: **Uncertainty-Aware 3D Gaussian Reconstruction for Limited-Angle CT**.
The report focuses on the worm data, the methods we implemented and the
experiments completed during this combined period.

- `TRDP2_Weeks4-5.pdf`: compiled 15-page report with 12 figures and 11 tables.
- `TRDP2_Weeks4-5.tex`: editable main document.
- `TRDP2_Weeks4-5_Overleaf.zip`: self-contained source upload for Overleaf.
- `figures/`: copies of the selected original experiment plots; hashes retained.
- `metrics_source.json`: 14 frozen result records and figure provenance.
- `resources_source.json`: software, CPU and allocated-storage snapshot.
- `references_verified.md`: source-by-source bibliography verification.
- `verify_report.py`: standard-library checks for principal tables, storage,
  citation keys and figure hashes. Run with `python3 verify_report.py`.

## Overleaf

Upload the ZIP as a new project, choose `TRDP2_Weeks4-5.tex` as the main document,
and select **XeLaTeX**. LuaLaTeX is also supported. Do not use pdfLaTeX.
The document explicitly selects **Times New Roman** with `fontspec` and contains:

```tex
\IfFileExists{inconsolata.sty}{\usepackage{inconsolata}}{}
```

The Times font is available by name in Overleaf's documented XeLaTeX setup.
The conditional Inconsolata package affects code text if installed; the report's
main text uses Times New Roman. References are inline-numbered and self-contained;
no BibTeX, shell escape, raw dataset, Python or network access is needed to compile.

On a local computer with Times New Roman installed, compile twice:

```bash
xelatex -interaction=nonstopmode -halt-on-error TRDP2_Weeks4-5.tex
xelatex -interaction=nonstopmode -halt-on-error TRDP2_Weeks4-5.tex
```

For this workstation's local validation, Times font files were extracted in
`/tmp/trdp2-report-fonts/` and supplied without installing them system-wide:

```bash
xelatex -interaction=nonstopmode -halt-on-error -jobname=TRDP2_Weeks4-5 \
  '\def\LocalTimesPath{/tmp/trdp2-report-fonts/}\input{TRDP2_Weeks4-5.tex}'
```

This temporary path is not needed on Overleaf. Font binaries and the installer
are excluded from the report package; the PDF contains embedded font subsets.

## Evidence and reading

The report includes dataset preparation, controlled noise, CPU/GPU and total
CREMI storage, classical and
native Gaussian baselines, failed conditional density uncertainty, known-phantom
representation/optimization/geometry controls, acquired-data initialization, and
final worm transfer failures. Error is judged downward; coverage is judged
against its nominal 95% target together with width. Worm voxel truth, physical
scaling and independent-specimen generalization remain unresolved.

The 16-versus-32-component controls use different sampling from the earlier
1,500-Gaussian slab and are not a matched accuracy ranking. The report
retains the scientific stability failures. Future staged optimization is a proposal,
not another completed experiment.

The bibliography cites external work only. Short method explanations cover
Gaussian conditioning, the local Laplace/Gauss–Newton approximation, nonnegative
least squares and Cholesky parameterization.

Validation: repeated XeLaTeX compilation, resolved citations/cross-references,
no overfull boxes or missing glyphs in the final log, actual Times New Roman
embedding checked with `pdffonts`, numerical table/provenance checks, and rendered
page inspection. The 42 research tests were already passing at evidence closure;
report generation did not rerun or modify the experiments.
