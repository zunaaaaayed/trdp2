"""Frozen sparse mask inventories, population guards, paired noise and evaluation."""

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from experiments.worm_uncertainty.data.completion_data import (
    acquisition_mask,
    channels,
    interpolate_missing,
)
from experiments.worm_uncertainty.data.matched_completion import grid, load_clean, sparse_noise
from experiments.worm_uncertainty.data.prepare import sha256

BASE = Path(__file__).resolve().parents[1]
PROTOCOL = BASE / "protocols/acquisition_diversity_v3.json"


def mask_hash(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def inventory():
    degrees = np.arange(25) * 7.2

    def record(start_index, count, group):
        span = {10: 72, 13: 90, 15: 108, 16: 115.2, 17: 120, 18: 129.6, 20: 144}[count]
        start = round(start_index * 7.2, 8)
        mask = acquisition_mask(start, span, degrees)
        if mask.sum() != count:
            raise ValueError("unachievable count")
        return dict(
            start=start,
            span=span,
            missing_wedge_degrees=180 - span,
            count=count,
            start_index=start_index,
            group=group,
            mask=mask.astype(int).tolist(),
            hash=mask_hash(mask),
        )

    familiar = [record(s, c, "familiar") for s in [0, 5, 10, 15] for c in [17, 20]]
    broad = [
        record(s, c, "train_broad")
        for c in [13, 15, 17, 20]
        for s in [0, 2, 5, 7, 10, 12, 15, 17, 20, 22]
    ]
    selection = familiar + [
        record(s, c, "new_in_distribution") for s in [1, 6, 11, 16, 21] for c in [13, 15]
    ]
    audit = familiar + [record(s, c, "new_in_distribution") for s in [8, 18, 23] for c in [13, 15]]
    audit += [record(s, c, "held_orientation") for s in [3, 4, 13, 14] for c in [13, 17]]
    audit += [record(s, c, "held_count") for s in [0, 10, 20] for c in [16, 18]]
    audit += [record(s, 10, "extrapolation") for s in [8, 18]]
    result = dict(A=familiar, B=broad, C=broad, selection=selection, audit=audit)
    validate_inventory(result)
    return result


def validate_inventory(inv):
    for name, rows in inv.items():
        hashes = [r["hash"] for r in rows]
        if len(hashes) != len(set(hashes)):
            raise ValueError("duplicate actual mask: " + name)
        for r in rows:
            mask = acquisition_mask(r["start"], r["span"], np.arange(25) * 7.2)
            if mask_hash(mask) != r["hash"] or mask.sum() != r["count"]:
                raise ValueError("mask record inconsistent")
    train = {r["hash"] for key in ["A", "B", "C"] for r in inv[key]}
    selection = {r["hash"] for r in inv["selection"] if r["group"] != "familiar"}
    audit = {r["hash"] for r in inv["audit"] if r["group"] != "familiar"}
    if train & selection or train & audit or selection & audit:
        raise ValueError("actual mask leakage")
    # Familiar masks intentionally recur for familiar-acquisition transfer.
    if {r["count"] for r in inv["B"]} & {
        r["count"] for r in inv["audit"] if r["group"] == "held_count"
    }:
        raise ValueError("held count trained")


def read_design(path=PROTOCOL):
    d = json.loads(Path(path).read_text())
    manifest = json.loads((BASE / "protocols/completion_manifest_v1.json").read_text())
    if sha256(BASE / "protocols/completion_manifest_v1.json") != d["manifest_sha256"]:
        raise ValueError("manifest changed")
    validate_inventory(d["inventories"])
    return d, manifest


def load_role(root, d, manifest, role):
    if role not in ["train", "selection", "audit"]:
        raise ValueError("reserved or unknown role forbidden")
    expected = "train" if role == "train" else "validation"
    records = {r["name"]: r for r in manifest["files"]}
    names = d["roles"][role]
    if any(records[n]["role"] != expected for n in names):
        raise ValueError("population role mismatch before I/O")
    cfg = dict(d, subsets={expected: names})
    return load_clean(Path(root), cfg, manifest, expected)


def noisy_bank(clean, d, names, seed, bank):
    cfg = dict(d, seeds={"training": seed})
    out = np.empty_like(clean)
    for f, name in enumerate(names):
        for zi, z in enumerate(d["sampling"]["z_indices"]):
            out[f, zi] = clean[f, zi] + sparse_noise(cfg, name, z, bank, "train")
    return out


def schedule(length, batch, updates, seed, rows):
    """Same deformation order and noise-bank sequence across arms; balanced counts."""
    rng = np.random.default_rng(seed)
    ids = np.concatenate(
        [rng.permutation(length) for _ in range((batch * updates + length - 1) // length)]
    )[: batch * updates]
    # An independent mask RNG keeps the data/noise schedule invariant to inventory size.
    masks = np.random.default_rng(seed + 1)
    groups = [
        [i for i, r in enumerate(rows) if r["count"] == c]
        for c in sorted({r["count"] for r in rows})
    ]
    levels = np.resize(np.arange(len(groups)), batch * updates)
    masks.shuffle(levels)
    mi = np.array([masks.choice(groups[c]) for c in levels], dtype=np.int64)
    banks = (np.arange(batch * updates) // length) % 16
    return ids.reshape(updates, batch), mi.reshape(updates, batch), banks.reshape(updates, batch)


def evaluation_arrays(clean, d, role):
    names = d["roles"][role]
    rows = d["inventories"][role]
    noisy = noisy_bank(clean, d, names, d["noise_seeds"][role], 0)
    obs = np.stack(
        [np.where(np.array(r["mask"], bool)[None, None, :, None], noisy, 0) for r in rows], axis=1
    )
    # [file,mask,z,angle,detector], canonical float64 observations.
    x = np.stack(
        [
            channels(obs[f, mi, zi], r["mask"], r["start"], r["span"])
            for f in range(len(names))
            for mi, r in enumerate(rows)
            for zi in range(8)
        ]
    )
    interp = np.stack(
        [
            interpolate_missing(obs[f, mi, zi], r["mask"])
            for f in range(len(names))
            for mi, r in enumerate(rows)
            for zi in range(8)
        ]
    ).reshape(obs.shape)
    return np.ascontiguousarray(x), obs, interp


def preserve(raw, observed, rows):
    masks = np.array([r["mask"] for r in rows], bool)[None, :, None, :, None]
    result = np.where(masks, observed, np.asarray(raw).reshape(observed.shape))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite completion")
    return result


def angular_distance(mask):
    delta = np.abs(np.arange(len(mask))[:, None] - np.flatnonzero(mask)[None, :])
    return np.minimum(delta, len(mask) - delta).min(axis=1)


def score(pred, clean, rows, names):
    records = []
    for f, name in enumerate(names):
        for mi, row in enumerate(rows):
            mask = np.array(row["mask"], bool)
            err = pred[f, mi] - clean[f]
            distance = angular_distance(mask)
            miss = ~mask

            def mse(m):
                return float(np.mean(err[:, m] ** 2)) if m.any() else None

            records.append(
                dict(
                    file=name,
                    mask_hash=row["hash"],
                    group=row["group"],
                    start=row["start"],
                    span=row["span"],
                    count=row["count"],
                    relative_l2=float(
                        np.linalg.norm(err[:, miss]) / np.linalg.norm(clean[f, :, miss])
                    ),
                    mse=mse(miss),
                    boundary_mse=mse(miss & (distance <= 2)),
                    center_mse=mse(miss & (distance >= 3)),
                )
            )
    groups = sorted({r["group"] for r in records}) + ["all", "primary_narrow"]
    summary = {}
    for group in groups:
        rr = [
            r
            for r in records
            if group == "all"
            or r["group"] == group
            or (
                group == "primary_narrow"
                and r["group"] == "new_in_distribution"
                and r["count"] == 13
            )
        ]
        if not rr:
            continue
        values = [
            float(np.mean([r["relative_l2"] for r in rr if r["file"] == name])) for name in names
        ]
        summary[group] = dict(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            worst=float(np.max(values)),
            per_file=dict(zip(names, values)),
            mse=float(np.mean([r["mse"] for r in rr])),
        )
    return dict(groups=summary, per_case=records)


def paired(summary, baseline):
    result = {}
    for group, s in summary["groups"].items():
        b = baseline["groups"][group]
        differences = {n: b["per_file"][n] - v for n, v in s["per_file"].items()}
        result[group] = dict(
            mean_improvement=b["mean"] - s["mean"],
            relative_improvement=1 - s["mean"] / b["mean"],
            wins=sum(v > 0 for v in differences.values()),
            win_fraction=float(np.mean([v > 0 for v in differences.values()])),
            per_file_difference=differences,
        )
    return result


def risk_curve(error, risk, fractions=(0.1, 0.25, 0.5, 0.75, 1.0)):
    error = np.asarray(error).ravel()
    risk = np.asarray(risk).ravel()
    order = np.argsort(risk, kind="stable")
    return [float(np.mean(error[order[: max(1, int(len(order) * f))]])) for f in fractions]


def disagreement(predictions, clean, rows, names):
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0, ddof=1)
    out = []
    for f, name in enumerate(names):
        e = []
        s = []
        distances = []
        for mi, r in enumerate(rows):
            mask = ~np.array(r["mask"], bool)
            e.append(np.abs(mean[f, mi] - clean[f])[:, mask].ravel())
            s.append(std[f, mi][:, mask].ravel())
            distances.append(
                np.broadcast_to(angular_distance(~mask)[None, :, None], clean[f].shape)[
                    :, mask
                ].ravel()
            )
        e = np.concatenate(e)
        s = np.concatenate(s)
        dist = np.concatenate(distances)
        random = np.random.default_rng(2026102909 + f).permutation(len(e))
        correlation = spearmanr(s, e).statistic if np.ptp(s) > 0 else 0.0
        out.append(
            dict(
                file=name,
                spearman=float(correlation),
                disagreement=risk_curve(e, s),
                angular_distance=risk_curve(e, dist),
                random=risk_curve(e, random),
                high_error_low_disagreement_fraction=float(
                    np.mean((e >= np.quantile(e, 0.9)) & (s <= np.quantile(s, 0.25)))
                ),
            )
        )
    return mean, std, out
