"""Training-only profile, paired A/B/C training and sealed development audit."""

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import time
import zipfile
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from experiments.worm_uncertainty.core.completion_model import CompletionCNN, angular_connectivity
from experiments.worm_uncertainty.data.acquisition_diversity import (
    PROTOCOL,
    read_design,
    load_role,
    noisy_bank,
    schedule,
    evaluation_arrays,
    preserve,
    score,
    paired,
    disagreement,
)
from experiments.worm_uncertainty.data.completion_data import channels
from experiments.worm_uncertainty.data.prepare import sha256
from experiments.worm_uncertainty.studies.train_completion import configure


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2) + "\n")


def fresh(path):
    path.mkdir(parents=True, exist_ok=False)


def provenance(folder, design):
    sources = list(Path("experiments/worm_uncertainty").rglob("*.py"))
    with zipfile.ZipFile(folder / "executed_sources.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p, str(p))
    write(
        folder / "provenance.json",
        dict(
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            dirty_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
            sources={str(p): sha256(p) for p in sources},
            host=socket.gethostname(),
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            protocol_sha256=sha256(PROTOCOL),
            reserved_read=False,
        ),
    )
    write(folder / "protocol.json", design)


def gpu_status():
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()


def input_tensor(noisy, mi, templates, ablate=False):
    geometry = templates[mi]
    measured = noisy * geometry[:, 0]
    x = torch.cat([measured[:, None], geometry], dim=1)
    if ablate:
        x[:, 2:] = 0
    return x.contiguous()


def templates(rows, device):
    return torch.tensor(
        np.stack(
            [channels(np.zeros((25, 62)), r["mask"], r["start"], r["span"])[1:] for r in rows]
        ),
        device=device,
    )


def missing_loss(pred, target, mask):
    missing = 1 - mask
    return (
        ((pred.float() - target.float()) ** 2 * missing).sum((1, 2)) / missing.sum((1, 2))
    ).mean()


def infer(model, x, batch):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(x[i : i + batch]).cpu().numpy())
    return np.concatenate(out)


def profile(args, d, m):
    folder = args.root / "profile"
    fresh(folder)
    provenance(folder, d)
    begun = time.monotonic()
    clean = load_role(args.data, d, m, "train")
    np.save(folder / "train_clean.npy", clean)
    load_seconds = time.monotonic() - begun
    configure(d["training_seeds"][0])
    cpu = CompletionCNN()
    x = torch.ones(4, 7, 25, 62)
    cpu(x).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in cpu.parameters())
    cache_before = time.monotonic()
    banks = np.array(
        [noisy_bank(clean, d, d["roles"]["train"], d["training_seeds"][0], i) for i in range(16)],
        dtype=np.float32,
    ).reshape(16, -1, 25, 62)
    cpu_preprocess = time.monotonic() - cache_before
    np.save(folder / "noise_seed0.npy", banks)
    transfer_before = time.monotonic()
    bank = torch.tensor(banks, device="cuda")
    target = torch.tensor(clean.reshape(-1, 25, 62), dtype=torch.float32, device="cuda")
    tmp = templates(d["inventories"]["B"], "cuda")
    torch.cuda.synchronize()
    transfer_seconds = time.monotonic() - transfer_before
    free, total = torch.cuda.mem_get_info()
    limit = min(free - 4 * 1024**3, total - 4 * 1024**3)
    candidates = []
    for batch in d["training"]["batch_candidates"]:
        if time.monotonic() - begun > 550:
            break
        configure(d["training_seeds"][0])
        model = CompletionCNN().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        x = input_tensor(bank[0, :batch], torch.arange(batch, device="cuda") % len(tmp), tmp)
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(3):
                opt.zero_grad()
                loss = missing_loss(model(x), target[:batch], x[:, 1])
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            started = time.monotonic()
            for _ in range(20):
                opt.zero_grad()
                loss = missing_loss(model(x), target[:batch], x[:, 1])
                loss.backward()
                if not torch.isfinite(loss) or not all(
                    torch.isfinite(p.grad).all() for p in model.parameters()
                ):
                    raise ValueError("nonfinite profile gradient")
                opt.step()
            torch.cuda.synchronize()
            duration = time.monotonic() - started
            row = dict(
                batch=batch,
                step_seconds=duration / 20,
                examples_per_second=batch * 20 / duration,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                status="passed",
                gpu_status=gpu_status(),
            )
            # Batch independence at tested weights, no batch normalization or dropout.
            model.eval()
            with torch.no_grad():
                single = model(x[:1])
                combined = model(x[: min(batch, 4)])[:1]
            row["batch_independence_max_difference"] = float((single - combined).abs().max())
            if not torch.allclose(single, combined, atol=2e-6, rtol=2e-5):
                raise ValueError("batch dependence")
            candidates.append(row)
        except torch.cuda.OutOfMemoryError:
            candidates.append(dict(batch=batch, status="OOM"))
        del model, opt, x
        torch.cuda.empty_cache()
    good = [r for r in candidates if r["status"] == "passed" and r["peak_reserved"] < limit]
    if not good:
        raise ValueError("no safe profile batch")
    best = max(r["examples_per_second"] for r in good)
    chosen = min(
        [r for r in good if r["examples_per_second"] >= 0.95 * best], key=lambda r: r["batch"]
    )
    # Allow validation/checkpoint overhead and cold kernels. Freeze common count now.
    updates = min(4000, int(240 / (chosen["step_seconds"] * 1.5)) // 200 * 200)
    if updates < 200:
        raise ValueError("throughput insufficient")
    budget = dict(
        batch_size=chosen["batch"],
        updates=updates,
        validation_frequency=200,
        precision="float32",
        profile_sha256=None,
        estimated_training_seconds_per_seed=updates * chosen["step_seconds"],
        maximum_seconds_per_seed=390,
    )
    report = dict(
        load_seconds=load_seconds,
        cpu_noise_preprocess_seconds=cpu_preprocess,
        initial_transfer_seconds=transfer_seconds,
        candidates=candidates,
        chosen_batch=chosen["batch"],
        updates=updates,
        elapsed_seconds=time.monotonic() - begun,
        cache_bytes=int(clean.nbytes + banks.nbytes),
        memory_headroom_bytes=4 * 1024**3,
        amp="not enabled; FP32 meets budget",
        workers=0,
        workers_reason="required slice/noise cache fits GPU; no per-update I/O or worker contention",
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    write(folder / "report.json", report)
    write(folder / "cache_hashes.json", {p.name: sha256(p) for p in folder.glob("*.npy")})
    budget["profile_sha256"] = sha256(folder / "report.json")
    write(args.root / "training_budget.json", budget)
    print(json.dumps(report), flush=True)


def train(args, d, m):
    folder = args.root / "primary"
    fresh(folder)
    provenance(folder, d)
    budget = json.loads((args.root / "training_budget.json").read_text())
    write(folder / "budget.json", budget)
    if sha256(args.root / "profile/report.json") != budget["profile_sha256"]:
        raise ValueError("profile changed")
    for name, digest in json.loads((args.root / "profile/cache_hashes.json").read_text()).items():
        if sha256(args.root / "profile" / name) != digest:
            raise ValueError("profile input cache changed")
    clean = np.load(args.root / "profile/train_clean.npy")
    before = time.monotonic()
    val = load_role(args.data, d, m, "selection")
    np.save(folder / "selection_clean.npy", val)
    x, obs, interp = evaluation_arrays(val, d, "selection")
    np.savez_compressed(folder / "selection_inputs.npz", observed=obs, interpolation=interp)
    vx = torch.tensor(x, device="cuda")
    target = torch.tensor(clean.reshape(-1, 25, 62), dtype=torch.float32, device="cuda")
    base = score(interp, val, d["inventories"]["selection"], d["roles"]["selection"])
    zero = score(obs, val, d["inventories"]["selection"], d["roles"]["selection"])
    write(folder / "baselines.json", dict(interpolation=base, zero_fill=zero))
    preparation = time.monotonic() - before
    results = []
    arm_used = {a: 0.0 for a in "ABC"}
    for si, seed in enumerate(d["training_seeds"]):
        before = time.monotonic()
        banks = (
            np.load(args.root / "profile/noise_seed0.npy")
            if si == 0
            else np.array(
                [noisy_bank(clean, d, d["roles"]["train"], seed, i) for i in range(16)],
                dtype=np.float32,
            ).reshape(16, -1, 25, 62)
        )
        bank = torch.tensor(banks, device="cuda")
        bank_digest = __import__("hashlib").sha256(banks.tobytes()).hexdigest()
        prep = time.monotonic() - before
        for arm in "ABC":
            dest = folder / f"{arm}_{seed}"
            fresh(dest)
            configure(seed)
            model = CompletionCNN().cuda()
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            ids, masks, noise_ids = schedule(
                len(target), budget["batch_size"], budget["updates"], seed, d["inventories"][arm]
            )
            np.savez_compressed(
                dest / "schedule.npz", samples=ids, masks=masks, noise_banks=noise_ids
            )
            tmp = templates(d["inventories"][arm], "cuda")
            vid = vx.clone() if arm == "C" else vx
            if arm == "C":
                vid[:, 2:] = 0
            torch.cuda.reset_peak_memory_stats()
            begun = time.monotonic()
            history = []
            best = float("inf")
            best_step = None
            losses = []
            updates = 0
            validation_seconds = 0
            train_seconds = 0
            micro = budget["batch_size"]
            ooms = []
            for step in range(budget["updates"]):
                if (
                    time.monotonic() - begun > 380
                    or arm_used[arm] + time.monotonic() - begun > 1180
                ):
                    break
                sample = torch.as_tensor(ids[step], device="cuda")
                mi = torch.as_tensor(masks[step], device="cuda")
                nb = torch.as_tensor(noise_ids[step], device="cuda")
                xx = input_tensor(bank[nb, sample], mi, tmp, arm == "C")
                yy = target[sample]
                model.train()
                start = time.monotonic()
                while True:
                    try:
                        opt.zero_grad(set_to_none=True)
                        total_loss = 0.0
                        for j in range(0, len(xx), micro):
                            fraction = min(micro, len(xx) - j) / len(xx)
                            loss = (
                                missing_loss(
                                    model(xx[j : j + micro]),
                                    yy[j : j + micro],
                                    xx[j : j + micro, 1],
                                )
                                * fraction
                            )
                            loss.backward()
                            total_loss += float(loss.detach())
                        if not np.isfinite(total_loss) or not all(
                            torch.isfinite(p.grad).all() for p in model.parameters()
                        ):
                            raise ValueError("nonfinite training")
                        opt.step()
                        break
                    except torch.cuda.OutOfMemoryError:
                        if micro == 1:
                            raise
                        micro = max(1, micro // 2)
                        ooms.append(dict(step=step, microbatch=micro))
                        opt.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                torch.cuda.synchronize()
                train_seconds += time.monotonic() - start
                losses.append(total_loss)
                updates = step + 1
                if updates % budget["validation_frequency"] == 0 or updates == budget["updates"]:
                    start = time.monotonic()
                    raw = infer(model, vid, budget["batch_size"])
                    pred = preserve(raw, obs, d["inventories"]["selection"])
                    metric = score(
                        pred, val, d["inventories"]["selection"], d["roles"]["selection"]
                    )
                    current = metric["groups"]["all"]["mean"]
                    if current < best:
                        best = current
                        best_step = updates
                        torch.save(model.state_dict(), dest / "best.pt")
                        np.save(dest / "selection_prediction.npy", pred)
                        write(dest / "best_metrics.json", metric)
                    validation_seconds += time.monotonic() - start
                    row = dict(
                        update=updates,
                        loss=float(np.mean(losses)),
                        selection_relative_l2=current,
                        selection_mse=metric["groups"]["all"]["mse"],
                        elapsed_seconds=time.monotonic() - begun,
                        gpu_status=gpu_status(),
                    )
                    history.append(row)
                    losses = []
                    write(dest / "history.json", history)
                    print(json.dumps(dict(arm=arm, seed=seed, **row)), flush=True)
            elapsed = time.monotonic() - begun
            arm_used[arm] += elapsed
            if best_step is None:
                raise RuntimeError("no completed validation checkpoint within budget")
            best_metrics = json.loads((dest / "best_metrics.json").read_text())
            comparisons = paired(best_metrics, base)
            n = comparisons["primary_narrow"]
            f = comparisons["familiar"]
            gate = (
                n["relative_improvement"] >= 0.05
                and n["wins"] >= 8
                and f["mean_improvement"] >= -0.02
            )
            result = dict(
                arm=arm,
                seed=seed,
                updates=updates,
                planned_updates=budget["updates"],
                best_update=best_step,
                gate=gate,
                paired=comparisons,
                metrics=best_metrics,
                elapsed_seconds=elapsed,
                training_seconds=train_seconds,
                validation_seconds=validation_seconds,
                noise_preparation_seconds_shared=prep,
                noise_sha256=bank_digest,
                examples_per_second=updates * budget["batch_size"] / max(train_seconds, 1e-9),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                parameters=sum(p.numel() for p in model.parameters()),
                effective_batch=budget["batch_size"],
                microbatch=micro,
                oom_retries=ooms,
                checkpoint_sha256=sha256(dest / "best.pt"),
                prediction_sha256=sha256(dest / "selection_prediction.npy"),
            )
            write(dest / "report.json", result)
            results.append(result)
            del model, opt, tmp, vid
            torch.cuda.empty_cache()
        del bank
        torch.cuda.empty_cache()
    passing = [
        a
        for a in "ABC"
        if all(r["gate"] and r["updates"] == budget["updates"] for r in results if r["arm"] == a)
    ]
    selected = (
        min(
            passing,
            key=lambda a: (
                np.mean(
                    [
                        r["metrics"]["groups"]["primary_narrow"]["mean"]
                        for r in results
                        if r["arm"] == a
                    ]
                ),
                a,
            ),
        )
        if passing
        else None
    )
    selection = dict(
        selected_arm=selected,
        passing_arms=passing,
        results=results,
        arm_seconds=arm_used,
        preparation_seconds=preparation,
        protocol_sha256=sha256(PROTOCOL),
        budget_sha256=sha256(args.root / "training_budget.json"),
        audit_not_opened=True,
    )
    write(folder / "selection.json", selection)
    print(json.dumps(dict(selected=selected, passing=passing, arm_seconds=arm_used)), flush=True)


def audit(args, d, m):
    frozen = args.root / "analysis_freeze.json"
    if not frozen.exists():
        raise ValueError("freeze capacity decision and analysis before audit")
    f = json.loads(frozen.read_text())
    if sha256(args.root / "primary/selection.json") != f["selection_sha256"]:
        raise ValueError("selection changed")
    folder = args.root / "audit"
    fresh(folder)
    provenance(folder, d)
    begun = time.monotonic()
    clean = load_role(args.data, d, m, "audit")
    np.save(folder / "clean.npy", clean)
    x, obs, interp = evaluation_arrays(clean, d, "audit")
    np.savez_compressed(folder / "inputs.npz", observed=obs, interpolation=interp)
    vx = torch.tensor(x, device="cuda")
    base = score(interp, clean, d["inventories"]["audit"], d["roles"]["audit"])
    zero = score(obs, clean, d["inventories"]["audit"], d["roles"]["audit"])
    results = []
    preds = {}
    budget = json.loads((args.root / "training_budget.json").read_text())
    selection = json.loads((args.root / "primary/selection.json").read_text())
    for r in selection["results"]:
        if time.monotonic() - begun > 570:
            raise TimeoutError("audit budget")
        arm, seed = r["arm"], r["seed"]
        dest = args.root / "primary" / f"{arm}_{seed}"
        if sha256(dest / "best.pt") != r["checkpoint_sha256"]:
            raise ValueError("checkpoint changed")
        configure(seed)
        model = CompletionCNN().cuda()
        model.load_state_dict(torch.load(dest / "best.pt", map_location="cuda"))
        inputs = vx.clone() if arm == "C" else vx
        if arm == "C":
            inputs[:, 2:] = 0
        start = time.monotonic()
        raw = infer(model, inputs, budget["batch_size"])
        duration = time.monotonic() - start
        pred = preserve(raw, obs, d["inventories"]["audit"])
        np.save(folder / f"{arm}_{seed}.npy", pred)
        metric = score(pred, clean, d["inventories"]["audit"], d["roles"]["audit"])
        results.append(
            dict(
                arm=arm,
                seed=seed,
                metrics=metric,
                paired=paired(metric, base),
                inference_seconds=duration,
                measured_preservation_error=0.0,
            )
        )
        preds.setdefault(arm, []).append(pred)
        del model, inputs
    selected = selection["selected_arm"]
    risk = None
    if selected is not None:
        mean, std, risk = disagreement(
            np.array(preds[selected]), clean, d["inventories"]["audit"], d["roles"]["audit"]
        )
        # Seed means can round identical FP64 measured values; restore them exactly.
        mean = preserve(mean, obs, d["inventories"]["audit"])
        observed_mask = np.array([r["mask"] for r in d["inventories"]["audit"]], bool)[
            None, :, None, :, None
        ]
        std = np.where(observed_mask, 0.0, std)
        np.savez_compressed(folder / "ensemble.npz", mean=mean, std=std)
    report = dict(
        baselines=dict(interpolation=base, zero_fill=zero),
        results=results,
        selected_arm=selected,
        risk=risk,
        elapsed_seconds=time.monotonic() - begun,
        analysis_freeze_sha256=sha256(frozen),
        reserved_read=False,
        artifacts_sha256={p.name: sha256(p) for p in folder.iterdir() if p.is_file()},
    )
    write(folder / "report.json", report)
    print(
        json.dumps(dict(audit_complete=True, selected=selected, seconds=report["elapsed_seconds"])),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["profile", "train", "audit"])
    parser.add_argument("--data", type=Path, default=Path("../worm_data"))
    parser.add_argument(
        "--root", type=Path, default=Path("output/worm_uncertainty/acquisition_diversity_v3")
    )
    args = parser.parse_args()
    d, m = read_design()
    configure(d["training_seeds"][0])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not silently change experiment")
    access = angular_connectivity()
    for rows in d["inventories"].values():
        for r in rows:
            mask = np.array(r["mask"], bool)
            if not access[~mask][:, mask].any(axis=1).all():
                raise ValueError("insufficient context")
    dict(profile=profile, train=train, audit=audit)[args.stage](args, d, m)


if __name__ == "__main__":
    main()
