"""Check the report's principal tables, citations and copied figure provenance."""

import hashlib
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
tex = (BASE / 'TRDP2_Weeks4-5.tex').read_text()
source = json.loads((BASE / 'metrics_source.json').read_text())
resources = json.loads((BASE / 'resources_source.json').read_text())


def record(name):
    return source['records'][name]['data']


def require_row(text):
    if text not in tex:
        raise ValueError('Missing or incorrect table row: ' + text)


conditional = record('conditional_density_v1/results.json')
for row in conditional['baseline']['results']:
    noise = round(100 * row['sigma'])
    withheld = '---' if row['withheld_relative_l2'] is None else f"{100 * row['withheld_relative_l2']:.2f}"
    require_row(f"{row['span']}\\degree & {noise}\\% & {100 * row['measured_relative_l2']:.2f} & {withheld}")
for row in conditional['posterior']['results']:
    if row['withheld_relative_l2'] is None:
        continue
    native = next(x for x in conditional['baseline']['results'] if x['case'] == row['case'])
    c = row['coverage']
    require_row(f"{row['span']}\\degree & {100 * native['withheld_relative_l2']:.2f}\\% & "
                f"{100 * row['withheld_relative_l2']:.2f}\\% & {100 * c['withheld_latent_clean']['coverage95']:.2f}\\% & "
                f"{100 * c['withheld_predictive_noisy']['coverage95']:.2f}\\% & {c['withheld_predictive_noisy']['mean_width95']:.4f}")
for row in record('known_3d_diagnosis_v1/results.json')['report']['uncertainty']:
    m = row['methods']
    require_row(f"{row['span']}\\degree & {100 * m['oracle_density']['coverage95']:.2f}\\% & "
                f"{100 * m['learned_density']['coverage95']:.2f}\\% & {100 * m['learned_joint']['coverage95']:.2f}\\% & "
                f"{m['learned_joint']['mean_width95']:.6f}")
labels = {
    'fresh_spread_120_30': 'Spread, 120\\degree{}/30\\degree',
    'fresh_spread_90_75': 'Spread, 90\\degree{}/75\\degree',
    'fresh_cluster_120_30': 'Cluster, 120\\degree{}/30\\degree',
    'fresh_cluster_90_75': 'Cluster, 90\\degree{}/75\\degree',
}
for row in record('measured_init_fallback_v1/results.json')['audit']['rows']:
    m = row['selected_metrics']['voxel']
    require_row(f"{labels[row['case']]} & {row['passes']}/{row['draws']} & {100 * m['coverage95']:.2f}\\% & "
                f"{m['mean_width95']:.5f} & {m['rmse']:.6f}")
controls = record('worm_fit_controls_v1/results.json')
for row in controls['rows']:
    arm = row['arm']
    prior = 'old' if arm['prior'] == 'parent' else 'new'
    require_row(f"New start / {prior} prior & {arm['components']} & {arm['budget_nfev']} & "
                f"{row['gate']['converged_candidates']}/4 & {100 * row['measured_relative_l2']:.2f}\\% & "
                f"{100 * row['withheld_relative_l2']:.2f}\\%")
assert len(controls['fit_diagnostics']) == 16
assert sum(x['converged'] for x in controls['fit_diagnostics']) == 2
total = next(value for path, value in resources['storage_bytes'].items() if Path(path).name == 'trdp2')
assert f'{total / 2**30:.2f}' in tex
for name, entry in source['figures'].items():
    assert hashlib.sha256((BASE / 'figures' / name).read_bytes()).hexdigest() == entry['sha256'], name
for name in re.findall(r'\\fig\{([^}]+)\}', tex):
    assert (BASE / 'figures' / name).is_file(), name
citations = set()
for group in re.findall(r'\\cite\{([^}]+)\}', tex):
    citations.update(group.split(','))
references = set(re.findall(r'\\bibitem\{([^}]+)\}', tex))
assert citations == references, (citations - references, references - citations)
assert r'\setmainfont{Times New Roman}' in tex
assert r'\IfFileExists{inconsolata.sty}{\usepackage{inconsolata}}{}' in tex
log = BASE / 'TRDP2_Weeks4-5.log'
if log.exists():
    for forbidden in ['undefined citations', 'undefined references', 'Overfull', 'Missing character']:
        assert forbidden not in log.read_text(errors='replace'), forbidden
assert 'Uncertainty-Aware 3D Gaussian Reconstruction for Limited-Angle CT' in tex
assert 'TRDP2 Weeks 4-5 | 12-23 October 2026' in tex
assert r'\section{Introduction}' in tex
assert r'\section{Experiments and results}' in tex
for omitted in ['Week 3', 'How to judge the plots and numbers', 'Acceptance and reproducibility', 'supervisor', 'cleanup', r'\bibitem{week3}', r'\bibitem{brun}', r'\bibitem{project}']:
    assert omitted not in tex, omitted
print(f'Verified numerical tables, total storage, {len(references)} external citations, 12 figure hashes and requested editorial changes.')
