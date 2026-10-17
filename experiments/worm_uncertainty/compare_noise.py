"""Paired CPU FBP pilot, not posterior uncertainty or independent-volume evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import tifffile
from skimage.transform import radon
from check_fbp import reconstruct, phantom_check, plt
from prepare import angle_mask, noisy_copy, sha256


def masked_fbp(noisy, mask, angles):
    # Fixed 1.8-degree quadrature: missing views contribute zero, not pi/Nobserved.
    measured_only = np.zeros_like(noisy)
    measured_only[mask] = noisy[mask]
    return reconstruct(measured_only, angles)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('output exists; use a fresh directory')
    checks, _, _ = phantom_check()
    with tifffile.TiffFile(args.source) as tif:
        if tif.series[0].shape != (510,100,310):
            raise ValueError('unexpected shape')
        clean = tif.pages[255].asarray().astype(float)
    if not np.isfinite(clean).all() or np.linalg.norm(clean)==0:
        raise ValueError('invalid clean signal')
    scale = float(np.sqrt(np.mean(clean**2)))
    angles = np.arange(100)*1.8
    arrays = dict(clean=clean, angles_deg=angles)
    rows = []
    figures = {}
    for fraction in [0., .01, .05]:
        for seed in ([0] if fraction==0 else [0,1,2]):
            noisy = noisy_copy(clean, scale, fraction, seed)
            arrays[f'noisy_{fraction}_{seed}'] = noisy
            for span in [180,120,90]:
                _, mask = angle_mask(0,span)
                result = masked_fbp(noisy, mask, angles)
                prediction = radon(result,theta=angles,circle=True,preserve_range=True).T
                residual = prediction-clean
                row = dict(span_deg=span, start_deg=0, fraction=fraction, seed=seed,
                           measured_views=int(mask.sum()), withheld_views=int((~mask).sum()))
                for name, selected in [('measured',mask),('withheld',~mask)]:
                    row[name+'_relative_l2'] = (float(np.linalg.norm(residual[selected])/np.linalg.norm(clean[selected]))
                                                 if selected.any() else None)
                rows.append(row)
                key=f'reconstruction_{span}_{fraction}_{seed}'
                arrays[key]=result
                arrays[f'mask_{span}']=mask
                if seed==0:
                    figures[span,fraction]=result
    # Noise-response variability is not missing-wedge/posterior uncertainty.
    for span in [180,120,90]:
        for fraction in [.01,.05]:
            arrays[f'noise_std_{span}_{fraction}']=np.std(
                [arrays[f'reconstruction_{span}_{fraction}_{seed}'] for seed in [0,1,2]],axis=0,ddof=1)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'arrays.npz',**arrays)
    lo=min(v.min() for v in figures.values()); hi=max(v.max() for v in figures.values())
    fig, axes=plt.subplots(3,3,figsize=(11,10),constrained_layout=True)
    for i,span in enumerate([180,120,90]):
        for j,fraction in enumerate([0.,.01,.05]):
            ax=axes[i,j]
            im=ax.imshow(figures[span,fraction],cmap='gray',vmin=lo,vmax=hi)
            ax.set_title(f'{span} deg, noise sigma={100*fraction:g}% RMS')
            ax.set_xlabel('pixel x'); ax.set_ylabel('pixel y')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Stored projection units per pixel (uncalibrated)')
    fig.savefig(args.output/'comparison.png',dpi=140); plt.close(fig)
    report=dict(kind='paired_fbp_development_pilot', source=str(args.source.resolve()),
                source_sha256=sha256(args.source),z=255,signal_scale=scale,
                scale_policy='development slice RMS; provisional, freeze before held-out evaluation',
                phantom_checks=checks, results=rows,
                reconstruction='ramp FBP; missing views zero-filled on full grid to preserve angular quadrature',
                limitations=['one slice of one deformation','3 noise realizations only',
                             'std maps measure noise response, not posterior uncertainty',
                             'no true voxel density target','simulator units and orientation not independently verified',
                             'projection error uses same numerical forward operator'],
                display_range=[float(lo),float(hi)],
                code_sha256={p:sha256(Path(__file__).with_name(p)) for p in ['compare_noise.py','prepare.py','check_fbp.py']},
                artifacts={p:sha256(args.output/p) for p in ['arrays.npz','comparison.png']})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:
        if row['seed']==0: print(row)


if __name__=='__main__':
    main()
