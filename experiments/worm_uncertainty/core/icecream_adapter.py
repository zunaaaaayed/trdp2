"""Geometry and execution adapters around the pinned, unmodified ICECREAM loss/model.

Upstream MIT source stays in an ignored checkout. We call its compute_loss;
we do not substitute a supervised loss. Fixed scaling, arbitrary CT orientation,
streamed FP32 patch accumulation and precise iteration limits are declared adapters.
"""

import contextlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "output/worm_uncertainty/alternative_reconstruction_sources/icecream"
COMMIT = "39c891a55af3e1cd96ef0d585668b0c2af4c177d"


def load_official():
    head = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "status", "--porcelain"], text=True
    )
    if head != COMMIT or dirty:
        raise ValueError("official code must be the clean pinned checkout")
    path = str(OFFICIAL / "src")
    if path not in sys.path:
        sys.path.insert(0, path)
    from icecream.models import get_model
    from icecream.trainer.eq_trainer import EquivariantTrainer
    from icecream.utils.utils import get_measurement

    return get_model, EquivariantTrainer, get_measurement


def native_axes(volume):
    """Canonical (z,y,x) -> upstream computational (x,y,z); no spatial resampling."""
    return np.ascontiguousarray(np.transpose(volume, (2, 1, 0)))


def canonical_axes(volume):
    return np.ascontiguousarray(np.transpose(volume, (2, 1, 0)))


def wedge(size, start, span, low_support=0.0, spherical=True):
    """Odd symmetric upstream grid; extrusion along physical axial z, last axis.

    Angle theta samples Fourier normals (kx,ky)=(cos(theta),-sin(theta)). The
    continuous wedge follows the declared interval; finite rays only approximate it.
    """
    q = np.linspace(-1, 1, int(size) + 1)
    fx, fy, fz = np.meshgrid(q, q, q, indexing="ij")
    theta = np.mod(np.rad2deg(np.arctan2(-fy, fx)) - start, 180)
    visible = theta < span - 1e-9 if span < 180 else np.ones_like(theta, dtype=bool)
    visible[(fx == 0) & (fy == 0)] = True
    ball = fx**2 + fy**2 + fz**2
    if spherical:
        visible &= ball < 1
    if low_support > 0:
        visible |= ball < low_support
    return torch.tensor(visible.astype(np.float32))


def rotate(tensor, k):
    kx, ky, kz, axis = k
    v = torch.rot90(tensor, kx, (1, 2))
    v = torch.rot90(v, ky, (0, 2))
    v = torch.rot90(v, kz, (0, 1))
    return torch.flip(v, [axis]) if axis != -1 else v


def rotation_set(w, minimum_distance=0.5):
    """Unique cubic isometries in the exact coordinate convention used by upstream.

    Generate rotation descriptors with the same axes as upstream's application;
    test their actual transformed wedges. Include reflections as upstream does.
    """
    labels = torch.arange(27).reshape(3, 3, 3)
    seen, result = set(), []
    norm = torch.linalg.vector_norm(w)
    for kx, ky, kz, flip in itertools.product(range(4), range(4), range(4), [-1, 0, 1, 2]):
        k = [kx, ky, kz, flip]
        key = tuple(rotate(labels, k).ravel().tolist())
        if key in seen:
            continue
        seen.add(key)
        if float(torch.linalg.vector_norm(rotate(w, k) - w) / norm) > minimum_distance:
            result.append(k)
    return result


def make_trainer(size, start, span, device="cpu", mixed=False, seed=2026111001):
    get_model, base, _ = load_official()
    defaults = yaml.safe_load((OFFICIAL / "src/icecream/defaults.yaml").read_text())
    cfg = defaults["train_params"]
    cfg.update(
        crop_size=size,
        batch_size=1,
        device=device,
        use_mixed_precision=mixed,
        num_workers=0,
        compile=False,
        load_device=False,
    )
    if span == 180:
        cfg["eq_weight"] = 0.0  # full-angle N2N control; no missing-wedge rotations exist

    class CTTrainer(base):
        def initialize_wedge(self, angle_max, angle_min, crop_size, wedge_support=None):
            low = self.configs.wedge_low_support if wedge_support is None else wedge_support
            w = wedge(crop_size, start, span, low, self.configs.use_spherical_support).to(
                self.device
            )
            return self.get_real_binary_filter(w)

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    model = get_model(**defaults["model_params"])
    trainer = CTTrainer(SimpleNamespace(**cfg), model, angle_max_set=[span], angle_min_set=[0])
    trainer.k_sets = rotation_set(trainer.wedge_full_set[0].cpu())
    if span < 180 and not trainer.k_sets:
        raise ValueError("no admissible wedge-changing rotation")
    if span == 180:
        trainer.k_sets = [[0, 0, 0, -1]]
    trainer.autocast = (
        torch.autocast("cuda", dtype=torch.float16) if mixed else contextlib.nullcontext()
    )
    return trainer, defaults["model_params"]


def crop_pair(pair, size, rng, xy_roi=None):
    # pair stores native x,y,z tensors; all axes have the same physical spacing.
    dims = pair[0].shape
    starts = []
    for axis, n in enumerate(dims):
        lo, hi = xy_roi if axis < 2 and xy_roi is not None else (0, n)
        if hi - lo < size:
            raise ValueError("patch does not fit contiguous support")
        starts.append(int(rng.integers(lo, hi - size + 1)))
    sl = tuple(slice(i, i + size) for i in starts)
    return [v[sl].unsqueeze(0).contiguous() for v in pair], starts


def infer_pair(trainer, pair, size, mixed=False, deadline=None):
    """Official f(A(f(y))) on each crop, central-half window, then pair average.

    Stream instead of materializing all crops; keep inputs/accumulator FP32 instead
    of upstream's unconditional FP16. Equal strides and padding have identity tests.
    """
    import time

    _, _, measure = load_official()
    trainer.model.eval()
    device = trainer.device
    stride, pad = size // 2, size // 4
    dims = tuple(pair[0].shape)
    padded = tuple(int(np.ceil(n / stride)) * stride + 2 * pad for n in dims)
    widths = [(pad, p - n - pad) for p, n in zip(padded, dims)]
    outputs = []
    for volume in pair:
        v = torch.nn.functional.pad(volume, tuple(a for lohi in widths[::-1] for a in lohi))
        out, counts = torch.zeros(padded), torch.zeros(padded)
        with torch.no_grad():
            for position in itertools.product(*(range(0, n - size + 1, stride) for n in padded)):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("inference wall cap reached; no silent partial volume")
                sl = tuple(slice(p, p + size) for p in position)
                crop = v[sl].to(device)[None, None]
                with torch.autocast("cuda", dtype=torch.float16, enabled=mixed):
                    first = trainer.model(crop)[:, 0]
                corrupted = measure(first.float(), trainer.wedge_input_set[0].to(device))
                with torch.autocast("cuda", dtype=torch.float16, enabled=mixed):
                    predicted = trainer.model(corrupted[:, None])[:, 0]
                window = trainer.window.cpu()
                out[sl] += predicted[0].float().cpu() * window
                counts[sl] += window
        original = tuple(slice(pad, pad + n) for n in dims)
        if (counts[original] == 0).any():
            raise ValueError("uncovered voxels in patch merger")
        outputs.append((out[original] / counts[original]).numpy())
    return (outputs[0] + outputs[1]) / 2


def method_record(trainer, model_config):
    return dict(
        official_commit=COMMIT,
        loss="unmodified EquivariantTrainer.compute_loss",
        model=model_config,
        parameter_count=sum(p.numel() for p in trainer.model.parameters()),
        config=vars(trainer.configs),
        rotation_count=len(trainer.k_sets),
        rotations=trainer.k_sets,
        inference="average of f(A(f(y0))) and f(A(f(y1))); FP32 streaming merge",
        normalization="fixed projection scale, fixed volume gain; no per-scan standardization",
    )
