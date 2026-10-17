"""Fixed-protocol measured-only slab fits; final withheld-view diagnostics."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import tifffile
from skimage.transform import iradon
from native_parallel import REPO,PIPE,camera,model_from_arrays
from fit_gaussian_slab import slab_layout
from prepare import sha256,angle_mask


def initialize_measured(noisy,angles,mask,spacing,count,seed):
    # Zero-fill only for angular quadrature, never as measured fitting targets.
    selected=np.zeros_like(noisy);selected[mask]=noisy[mask]
    depth,width=noisy.shape[1:]
    initial=np.stack([iradon(selected[:,z,:].T,theta=angles,filter_name='ramp',circle=True,preserve_range=True) for z in range(depth)])
    positive=np.maximum(initial,0)
    if positive.sum()<=0:raise ValueError('empty initializer')
    rng=np.random.default_rng(seed)
    indices=rng.choice(positive.size,size=count,replace=True,p=positive.ravel()/positive.sum())
    iz,iy,ix=np.unravel_index(indices,positive.shape)
    xyz=np.stack([(ix+.5-width/2)*spacing,(iy+.5-width/2)*spacing,(iz+.5-depth/2)*spacing],axis=1)
    xyz+=rng.uniform(-.25,.25,xyz.shape)*spacing
    scales=np.tile(np.array([2.,2.,1.])*spacing,(count,1))
    density=positive.sum()*spacing**2/(count*(2*np.pi)**1.5*np.prod(scales[0]))
    return xyz,scales,np.full(count,density)


def projection_errors(prediction,clean,mask):
    return {name:float(np.linalg.norm((prediction-clean)[m])/np.linalg.norm(clean[m])) if m.any() else None
            for name,m in [('measured_relative_l2',mask),('withheld_relative_l2',~mask)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=Path(__file__).with_name('slab_protocol_v1.json'))
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.output.exists():raise ValueError('output exists')
    cfg=json.loads(args.protocol.read_text())
    if not args.execute:print(json.dumps(cfg,indent=2));return
    import torch
    sys.path.insert(0,str(REPO));from r2_gaussian.gaussian import render,query
    if sha256(args.source)!=cfg['source_sha256']:raise ValueError('source hash mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    depth,width=cfg['depth'],cfg['width'];rows,spacing=slab_layout(depth,width)
    with tifffile.TiffFile(args.source) as tif:
        if tif.series[0].shape!=(510,100,310):raise ValueError('source axes mismatch')
        clean=np.stack([tif.pages[z].asarray() for z in range(cfg['z_start'],cfg['z_start']+depth)]).transpose(1,0,2)/cfg['normalization']
    angles=np.arange(100)*1.8
    if not np.isfinite(clean).all():raise ValueError('nonfinite inputs')
    unit_noise=np.random.default_rng(cfg['noise_seed']).normal(size=clean.shape)
    cameras=[camera(np.deg2rad(a),width) for a in angles]
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    np.savez_compressed(args.output/'inputs.npz',clean=clean,unit_noise=unit_noise,angles_deg=angles)
    summary=[]
    for sigma in cfg['noise_sigmas_normalized']:
        noisy=clean+sigma*unit_noise
        for span in cfg['spans_deg']:
            key=f'span{span}_sigma{sigma:g}';destination=args.output/key;destination.mkdir()
            (destination/'status.json').write_text('{"status":"running"}\n')
            torch.manual_seed(cfg['optimization_seed']);rng=np.random.default_rng(cfg['optimization_seed'])
            _,mask=angle_mask(0,span);indices=np.flatnonzero(mask)
            xyz,scales,densities=initialize_measured(noisy,angles,mask,spacing,cfg['gaussian_count'],cfg['optimization_seed'])
            model=model_from_arrays(xyz,scales,densities)
            optimizer=torch.optim.Adam([dict(params=[model._xyz],lr=.0002),dict(params=[model._scaling],lr=.005),dict(params=[model._rotation],lr=.001),dict(params=[model._density],lr=.05)])
            tensors=torch.tensor(noisy,dtype=torch.float32,device='cuda');history=[]
            for step in range(cfg['updates']):
                if step%len(indices)==0:order=rng.permutation(indices)
                index=int(order[step%len(indices)]);optimizer.zero_grad(set_to_none=True)
                prediction=render(cameras[index],model,PIPE)['render'][0,rows,:]
                loss=((prediction-tensors[index])**2).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward()
                for parameter in [model._xyz,model._scaling,model._rotation,model._density]:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():raise ValueError('nonfinite gradients')
                optimizer.step()
                with torch.no_grad():
                    model._xyz[:,:2].clamp_(-.99,.99);model._xyz[:,2].clamp_(-depth*spacing/2,depth*spacing/2)
                    model._scaling.clamp_(np.log(.5*spacing),np.log(8*spacing))
                history.append(float(loss))
            with torch.no_grad():
                prediction=np.stack([render(c,model,PIPE)['render'][0,rows,:].cpu().numpy() for c in cameras])
                volume=query(model,[0,0,0],[width,width,depth],[2,2,depth*spacing],PIPE)['vol'].cpu().numpy()
            if not np.isfinite(prediction).all() or not np.isfinite(volume).all():raise ValueError('nonfinite output')
            np.savez_compressed(destination/'arrays.npz',prediction=prediction,volume_xyz=volume,mask=mask,loss=history)
            np.savez(destination/'gaussians.npz',xyz=model.get_xyz.detach().cpu().numpy(),scales=model.get_scaling.detach().cpu().numpy(),rotation=model.get_rotation.detach().cpu().numpy(),density=model.get_density.detach().cpu().numpy())
            record=dict(case=key,span=span,sigma=sigma,measured_views=int(mask.sum()),withheld_views=int((~mask).sum()),
                        **projection_errors(prediction,clean,mask),artifacts={name:sha256(destination/name) for name in ['arrays.npz','gaussians.npz']})
            (destination/'report.json').write_text(json.dumps(record,indent=2)+'\n');(destination/'status.json').write_text('{"status":"completed"}\n')
            summary.append(record);print(json.dumps(record),flush=True)
            del model,optimizer,tensors
    report=dict(kind='measured_only_limited_gaussian_baselines',protocol_sha256=sha256(args.protocol),inputs_sha256=sha256(args.output/'inputs.npz'),results=summary,
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,code_sha256={n:sha256(Path(__file__).with_name(n)) for n in ['limited_gaussian.py','native_parallel.py','fit_gaussian_slab.py']})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
