"""Short full-angle 3D Gaussian integration pilot; no posterior inference yet."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import tifffile
from skimage.transform import iradon
from prepare import sha256
from native_parallel import camera,model_from_arrays,PIPE,REPO


def slab_layout(depth,width=310):
    if depth<=0 or depth>width or (width-depth)%2:
        raise ValueError('depth must be positive, <= width, and match width parity')
    first=(width-depth)//2
    return slice(first,first+depth),2/width


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise ValueError('output exists')
    if not args.execute:
        print('Dry run: 8 slices z=252:260, 100 clean angles, 1500 Gaussians, 300 updates; --execute required')
        return
    import torch
    sys.path.insert(0,str(REPO))
    from r2_gaussian.gaussian import render,query
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    started=time.perf_counter();torch.manual_seed(0);rng=np.random.default_rng(0)
    depth=8;width=310;zstart=252;count=1500;steps=300
    with tifffile.TiffFile(args.source) as tif:
        if tif.series[0].shape!=(510,100,310):raise ValueError('unexpected source shape')
        stack=np.stack([tif.pages[i].asarray() for i in range(zstart,zstart+depth)])
    if not np.isfinite(stack).all() or (stack<0).any():raise ValueError('invalid source values')
    # angle,z,detector, no reflection or interpolation of measured projections.
    projections=stack.transpose(1,0,2)
    scale=float(np.sqrt(np.mean(projections**2)))
    if scale<=0:raise ValueError('empty slab')
    target=projections/scale; angles=np.arange(100)*1.8
    rows,spacing=slab_layout(depth,width)
    initial=np.stack([iradon(target[:,z,:].T,theta=angles,filter_name='ramp',circle=True,preserve_range=True) for z in range(depth)])
    positive=np.maximum(initial,0)
    probs=positive.ravel()/positive.sum()
    selected=rng.choice(positive.size,size=count,replace=True,p=probs)
    iz,iy,ix=np.unravel_index(selected,positive.shape)
    # Detector and voxel centres have symmetric half-pixel convention.
    xyz=np.stack([(ix+.5-width/2)*spacing,(iy+.5-width/2)*spacing,
                  (iz+.5-depth/2)*spacing],axis=1)
    xyz+=rng.uniform(-.25,.25,xyz.shape)*spacing
    scales=np.tile(np.array([2.,2.,1.])*spacing,(count,1))
    density=positive.sum()*spacing**2/(count*(2*np.pi)**1.5*np.prod(scales[0]))
    model=model_from_arrays(xyz,scales,np.full(count,density))
    optimizer=torch.optim.Adam([dict(params=[model._xyz],lr=.0002),dict(params=[model._scaling],lr=.005),
                                dict(params=[model._rotation],lr=.001),dict(params=[model._density],lr=.05)])
    cameras=[camera(np.deg2rad(a),width) for a in angles]
    tensors=torch.tensor(target,dtype=torch.float32,device='cuda')
    def predict():
        with torch.no_grad():return np.stack([render(c,model,PIPE)['render'][0,rows,:].cpu().numpy() for c in cameras])
    before=predict();history=[]
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'status.json').write_text(json.dumps(dict(status='running',steps=steps))+'\n')
    for step in range(steps):
        if step%100==0:order=rng.permutation(100)
        index=int(order[step%100]);optimizer.zero_grad(set_to_none=True)
        pred=render(cameras[index],model,PIPE)['render'][0,rows,:]
        loss=((pred-tensors[index])**2).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward()
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():raise ValueError('nonfinite gradient')
        optimizer.step()
        with torch.no_grad():
            model._xyz[:,0:2].clamp_(-.99,.99)
            model._xyz[:,2].clamp_(-depth*spacing/2,depth*spacing/2)
            model._scaling.clamp_(np.log(.5*spacing),np.log(8*spacing))
        history.append(float(loss))
        if (step+1)%100==0:print(f'Update {step+1}/{steps}: normalized measured MSE {float(loss):.6g}',flush=True)
    after=predict()
    with torch.no_grad():
        volume=query(model,[0,0,0],[width,width,depth],[2,2,depth*spacing],PIPE)['vol'].cpu().numpy()
    if not np.isfinite(volume).all():raise ValueError('nonfinite volume')
    np.savez_compressed(args.output/'arrays.npz',target_stored=projections,target_normalized=target,
                        before=before,after=after,volume_xyz=volume,initial_fbp_zyx=initial/spacing,
                        loss=np.array(history),angles_deg=angles)
    np.savez(args.output/'gaussians.npz',xyz=model.get_xyz.detach().cpu().numpy(),
             scales=model.get_scaling.detach().cpu().numpy(),rotation=model.get_rotation.detach().cpu().numpy(),
             density=model.get_density.detach().cpu().numpy())
    from check_fbp import plt
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    z=depth//2;ref=initial[z]/spacing;final=volume[:,:,z].T
    lo=min(ref.min(),final.min());hi=max(ref.max(),final.max())
    for ax,array,title in [(axes[0],ref,'FBP initialization (comparison only)'),(axes[1],final,'3D Gaussian query after 300 updates')]:
        im=ax.imshow(array[105:205,105:205],vmin=lo,vmax=hi,cmap='gray');ax.set_title(title,fontsize=9)
        ax.set_xlabel('crop x index');ax.set_ylabel('crop y index')
    fig.colorbar(im,ax=axes[:2].tolist(),label='normalized projection per scene length; brightness is not accuracy',shrink=.7)
    axes[2].plot(history);axes[2].set(xlabel='update',ylabel='measured-view MSE (lower is better)',title='Different view each update')
    fig.savefig(args.output/'diagnostic.png',dpi=140);plt.close(fig)
    report=dict(kind='short_full_angle_worm_3d_gaussian_pilot',status='completed',source=str(args.source.resolve()),
                source_sha256=sha256(args.source),z_range=[252,260],native_spacing_um=1,
                projection_axes=['angle','z','detector'],volume_axes=['x','y','z'],angles=100,steps=steps,gaussians=count,
                source_rms=scale,scene_spacing=spacing,observed_detector_rows=[rows.start,rows.stop],
                relative_projection_l2_before=float(np.linalg.norm(before-target)/np.linalg.norm(target)),
                relative_projection_l2_after=float(np.linalg.norm(after-target)/np.linalg.norm(target)),
                elapsed_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                seed=0,code_sha256={n:sha256(Path(__file__).with_name(n)) for n in ['fit_gaussian_slab.py','native_parallel.py']},
                limitations=['integration pilot, not converged reconstruction','all 100 angles used; no held-out score',
                             'no density calibration or posterior inference','square image with loss on observed slab rows only',
                             'assumed symmetric pixel centres and skimage angle convention; simulator convention unresolved',
                             'FBP initialization is not ground truth','fixed Gaussian count; no densification',
                             'normalization derived from this full-angle development slab; not transferable test normalization'],
                artifacts={n:sha256(args.output/n) for n in ['arrays.npz','gaussians.npz','diagnostic.png']})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'status.json').write_text(json.dumps(dict(status='completed',steps=steps))+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
