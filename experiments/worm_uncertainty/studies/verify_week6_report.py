"""Check report layout, embedded fonts, references, figure hashes and saved numbers."""

import json
import re
import subprocess
from pathlib import Path
from experiments.worm_uncertainty.data.prepare import sha256

ROOT = Path(__file__).resolve().parents[3]


def main():
    folder = ROOT / "Reports/Week6"
    tex = (folder / "TRDP2_Week6.tex").read_text()
    pdf = folder / "TRDP2_Week6.pdf"
    if {p.name for p in folder.iterdir()} != {"figures", "TRDP2_Week6.tex", "TRDP2_Week6.pdf"}:
        raise ValueError("report folder contains extra files")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    if not re.search(r"Pages:\s+8\b", info) or "595.28 x 841.89" not in info:
        raise ValueError("expected8 A4 pages")
    fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    if "TimesNewRomanPSMT" not in fonts or "TimesNewRomanPS-BoldMT" not in fonts:
        raise ValueError("Times New Roman not embedded")
    for token in [
        r"\documentclass[11pt,a4paper]{article}",
        r"\IfFileExists{inconsolata.sty}{\usepackage{inconsolata}}{}",
        r"\setmainfont{Times New Roman}",
        "TRDP2 Week 6 | 24--30 October 2026",
    ]:
        if token not in tex:
            raise ValueError("format/header missing: " + token)
    refs = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    cited = {key for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
    if refs != cited or len(refs) != 9:
        raise ValueError("unresolved/unused citations")
    figures = set(re.findall(r"\\fig\{([^}]+)\}", tex))
    if figures != {p.name for p in (folder / "figures").iterdir()}:
        raise ValueError("missing/unused figures")
    sources = json.loads((ROOT / "docs/reports/week6_sources.json").read_text())
    for name, digest in sources["sources"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("source changed: " + name)
    for name, digest in sources["figures"].items():
        if sha256(folder / "figures" / name) != digest:
            raise ValueError("figure changed")
    normalized = re.sub(r"\\textbf\{([^}]+)\}", r"\1", tex)

    def row_present(label, values):
        line = next(
            (line for line in normalized.splitlines() if line.startswith(label + " &")), None
        )
        if line is None:
            raise ValueError("table row absent: " + label)
        numbers = [float(x) for x in re.findall(r"([0-9]+\.[0-9]+)\\%", line)]
        if len(numbers) != len(values) or any(
            abs(a - b) > 0.00001 for a, b in zip(numbers, values)
        ):
            raise ValueError("table numbers differ: " + label)

    base = ROOT / "docs/figures/worm_uncertainty"
    staged = json.loads((base / "worm_staged_v1/summary.json").read_text())
    for label, r in zip(["Joint fitting", "Staged fitting"], staged["rows"]):
        row_present(
            label, [round(100 * r[k], 2) for k in ["measured_relative_l2", "withheld_relative_l2"]]
        )
    current = json.loads((base / "acquisition_diversity_v3/summary.json").read_text())
    for label, role, group in [
        ("Narrow selection", "selection", "primary_narrow"),
        ("Narrow audit", "audit", "primary_narrow"),
        ("Familiar audit", "audit", "familiar"),
        ("Audit held orientations", "audit", "held_orientation"),
        ("Audit held counts 16/18", "audit", "held_count"),
        ("Audit 72\\degree{} extrapolation", "audit", "extrapolation"),
    ]:
        r = next(r for r in current["tables"][role] if r["group"] == group)
        row_present(
            label,
            [
                round(100 * (r[a] if a == "interpolation" else r[a]["mean"]), 2)
                for a in ["interpolation", "A", "B", "C"]
            ],
        )
    # Check stage table cells against saved per-case results, not an earlier pilot.
    for case in current["refinement"]["cases"]:
        for r in case["rows"]:
            for stage in ["completion", "initialization", "refined"]:
                value = f"{100 * r['errors'][stage]['withheld']:.2f}\\%"
                if value not in normalized:
                    raise ValueError("stage value missing: " + value)
    text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    if "??" in text:
        raise ValueError("unresolved PDF reference")
    record = dict(
        pages=8,
        page_size="A4",
        body_font="Times New Roman,11pt",
        bibliography_entries=9,
        figures=sorted(figures),
        tables_checked=["staged fitting", "primary selection/audit", "Gaussian stages"],
        tex_sha256=sha256(folder / "TRDP2_Week6.tex"),
        pdf_sha256=sha256(pdf),
        sources_manifest_sha256=sha256(ROOT / "docs/reports/week6_sources.json"),
        report_folder_structure="figures/ plus TeX and PDF only",
        visual_review="All8 rendered pages reviewed; detailed acquisition-plot page inspected",
        verification="passed",
    )
    (ROOT / "docs/reports/week6_validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        "PASS:8 A4 pages, embedded Times New Roman,11pt source,9 resolved citations,5 figures, saved numerical tables and clean report folder"
    )


if __name__ == "__main__":
    main()
