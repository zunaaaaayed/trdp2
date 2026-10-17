"""Fixed-budget nonnegative SART vs FBP, one development slice, seed zero."""
import argparse
import json
from pathlib import Path
import numpy as np
import skimage
from skimage.transform import iradon_sart, radon
from check_fbp import plt
from prepare import sha256


def fit_sart(noisy, angles, mask, iterations=10):
    image = np.zeros((noisy.shape[1], noisy.shape[1]), dtype=float)
    yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
    outside = (yy-image.shape[0]//2)**2 + (xx-image.shape[1]//2)**2 > (image.shape[0]//2)**2
    for _ in range(iterations):
        image = iradon_sart(noisy[mask].T, theta=angles[mask], image=image,
                            relaxation=.15, clip=(0, np.inf))
        image[outside] = 0
    if not np.isfinite(image).all():
        raise ValueError('nonfinite reconstruction')
    return image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('output exists')
    report=json.loads((args.input/'report.json').read_text())
    if sha256(args.input/'arrays.npz') != report['artifacts']['arrays.npz']:
        raise ValueError('input checksum mismatch')
    data=np.load(args.input/'arrays.npz')
    angles=data['angles_deg']; clean=data['clean']
    rows=[]; results={}
    for span in [180,120,90]:
        mask=data[f'mask_{span}']
        for fraction in [0., .05]:
            noisy=data[f'noisy_{fraction}_0']
            image=fit_sart(noisy,angles,mask)
            results[f'sart_{span}_{fraction}']=image
            prediction=radon(image,theta=angles,circle=True,preserve_range=True).T
            residual=prediction-clean
            row=dict(span=span,fraction=fraction,
                     measured_relative_l2=float(np.linalg.norm(residual[mask])/np.linalg.norm(clean[mask])),
                     withheld_relative_l2=float(np.linalg.norm(residual[~mask])/np.linalg.norm(clean[~mask])) if (~mask).any() else None)
            rows.append(row); print(row,flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'arrays.npz',**results)
    fig,axes=plt.subplots(3,4,figsize=(13,9),constrained_layout=True)
    # Identical fixed spatial crop chosen around detector centre, not per-method peaks.
    lo=min(data[f'reconstruction_{s}_{f}_0'].min() for s in [180,120,90] for f in [0.,.05])
    hi=max(max(v.max() for v in results.values()),max(data[f'reconstruction_{s}_{f}_0'].max() for s in [180,120,90] for f in [0.,.05]))
    for i,span in enumerate([180,120,90]):
        for j,(method,fraction) in enumerate([('fbp',0.),('sart',0.),('fbp',.05),('sart',.05)]):
            value=data[f'reconstruction_{span}_{fraction}_0'] if method=='fbp' else results[f'sart_{span}_{fraction}']
            im=axes[i,j].imshow(value[105:205,105:205],extent=(105,205,205,105),vmin=lo,vmax=hi,cmap='gray')
            axes[i,j].set_title(f'{span}° {method.upper()}, {fraction*100:g}% noise')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Uncalibrated stored projection units per pixel')
    fig.savefig(args.output/'comparison.png',dpi=140);plt.close(fig)
    summary=dict(kind='development_sart_comparison',input_report_sha256=sha256(args.input/'report.json'),
                 input_arrays_sha256=sha256(args.input/'arrays.npz'), iterations=10, relaxation=.15,
                 initialization='zeros',constraint='nonnegative and circular support after each sweep',seed=0,skimage_version=skimage.__version__,
                 selection='fixed 10 iterations; no withheld data used for fitting or stopping',
                 code_sha256=sha256(__file__),results=rows,display_range=[float(lo),float(hi)],crop=[105,205,105,205],
                 limitations=['one slice, one noise seed','FBP is unconstrained, SART is nonnegative: not an isolated algorithm ablation',
                              'different forward discretizations in SART and radon evaluation',
                              'no voxel ground truth, density calibration or uncertainty estimates'],
                 artifacts={name:sha256(args.output/name) for name in ['arrays.npz','comparison.png']})
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
