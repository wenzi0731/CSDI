"""python -m step_study.run --help"""
import argparse
import copy
import csv
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import load_series, prepare, synthetic_series
from .metrics import crps, energy_score, step_summary
from .models import DLinear, ForecastCSDI


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit(model, datasets, config, device, seed, path, baseline=None, stage1=False):
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True,
                       generator=generator)
    valid = DataLoader(datasets["val"], batch_size=config["batch_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    epochs = config["dlinear_epochs" if stage1 else "diffusion_epochs"]
    best, log = float("inf"), []
    def objective(history, truth):
        if stage1:
            return torch.nn.functional.mse_loss(model(history), truth)
        with torch.no_grad():
            target = truth if baseline is None else truth - baseline(history)
        return model.loss(history, target)
    for epoch in range(epochs):
        model.train()
        train_sum, seen = 0., 0
        for history, truth in train:
            history, truth = history.to(device), truth.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(history, truth)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            train_sum += loss.item() * len(history)
            seen += len(history)
        model.eval()
        val_sum = 0.
        # Fixed validation timesteps/noise, isolated from training's RNG stream.
        devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed + 10000)
            for history, truth in valid:
                val_sum += objective(history.to(device), truth.to(device)).item() * len(history)
        val_loss = val_sum / len(datasets["val"])
        if not np.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), path)
        log.append({"epoch": epoch + 1, "train_loss": train_sum / seen, "val_loss": val_loss})
        print(f"{path.stem} epoch={epoch+1}/{epochs} train={train_sum/seen:.5f} val={val_loss:.5f}", flush=True)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    dump_json(path.with_suffix(".history.json"), log)


@torch.no_grad()
def evaluate(models, baseline, datasets, config, scaler, names, device, seed):
    rows, joint = [], []
    scale = torch.tensor(scaler["scale"], device=device)
    for split in ("val", "test"):
        loader = DataLoader(datasets[split], batch_size=config["batch_size"])
        for method, model in models.items():
            for steps in config["steps"]:
                # CPU generator makes noise independent of CUDA RNG and method order.
                generator = torch.Generator().manual_seed(seed + (20000 if split == "val" else 30000))
                sums = torch.zeros(len(names), device=device)
                squares = torch.zeros_like(sums)
                es_sum, count, windows = 0., 0, 0
                elapsed = 0.
                for history, truth in loader:
                    history, truth = history.to(device), truth.to(device)
                    noise = torch.randn((len(history), config["samples"], *truth.shape[1:]),
                                        generator=generator).to(device)
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    samples = model.sample(history, noise, steps)
                    if method == "residual":
                        samples = samples + baseline(history)[:, None]
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    elapsed += time.perf_counter() - start
                    if not torch.isfinite(samples).all():
                        raise RuntimeError(f"Nonfinite samples: {method}, steps={steps}")
                    sums += crps(samples, truth).sum((0, 1))
                    squares += (samples.mean(1) - truth).square().sum((0, 1))
                    es_sum += energy_score(samples, truth).sum().item()
                    windows += len(history)
                    count += len(history) * truth.shape[1]
                for i, name in enumerate(names):
                    rows.append({"seed": seed, "split": split, "method": method,
                                 "steps": steps, "variable": name,
                                 "crps_scaled": (sums[i] / count).item(),
                                 "crps_original": (sums[i] / count * scale[i]).item(),
                                 "rmse_scaled": (squares[i] / count).sqrt().item()})
                joint.append({"seed": seed, "split": split, "method": method, "steps": steps,
                              "energy_score_scaled": es_sum / windows,
                              "sampling_seconds": elapsed, "windows": windows,
                              "samples_per_window": config["samples"]})
                print(f"eval seed={seed} {split} {method} K={steps} CRPS={sums.mean().item()/count:.5f}", flush=True)
    return rows, joint


@torch.no_grad()
def residual_diagnostics(baseline, dataset, batch_size, device, names):
    targets, residuals = [], []
    for history, truth in DataLoader(dataset, batch_size=batch_size):
        mu = baseline(history.to(device)).cpu()
        targets.append(truth)
        residuals.append(truth - mu)
    target, residual = torch.cat(targets).flatten(0, 1), torch.cat(residuals).flatten(0, 1)
    return [{"variable": name, "target_variance_scaled": target[:, i].var(unbiased=False).item(),
             "residual_variance_scaled": residual[:, i].var(unbiased=False).item(),
             "dlinear_mse_scaled": residual[:, i].square().mean().item()}
            for i, name in enumerate(names)]


def plot_curves(rows, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r["variable"] for r in rows))
    for split in ("val", "test"):
        # Paginate to keep high-dimensional datasets readable.
        for page, offset in enumerate(range(0, len(names), 12), 1):
            group = names[offset:offset + 12]
            fig, axes = plt.subplots((len(group) + 2) // 3, min(3, len(group)),
                                     figsize=(min(3, len(group)) * 4.2, ((len(group)+2)//3)*3.3), squeeze=False)
            for ax, name in zip(axes.flat, group):
                for method, color in (("direct", "#2463a6"), ("residual", "#c45326")):
                    subset = [r for r in rows if r["variable"] == name and r["method"] == method and r["split"] == split]
                    steps = sorted({r["steps"] for r in subset})
                    values = np.array([[r["crps_scaled"] for r in subset if r["steps"] == k] for k in steps])
                    mean, sd = values.mean(1), values.std(1)
                    ax.plot(steps, mean, "o-", label=method, color=color)
                    ax.fill_between(steps, mean-sd, mean+sd, color=color, alpha=.15)
                ax.set(title=name, xlabel="DDIM steps (NFE)", ylabel="CRPS / train std (lower is better)", xscale="log")
                ax.grid(alpha=.2)
                ax.legend()
            for ax in list(axes.flat)[len(group):]:
                ax.set_visible(False)
            fig.suptitle(f"{split}: mean ± population SD across training seeds")
            fig.tight_layout()
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"performance_steps_{split}_{page}.{suffix}", dpi=180)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ("direct", "residual"):
        selected = [r for r in summary if r["method"] == method]
        values = np.array([r["validation_shared_regret_curve"] for r in selected])
        steps = selected[0]["steps"]
        ax.plot(steps, values.mean(0), "o-", label=method)
        ax.fill_between(steps, values.mean(0)-values.std(0), values.mean(0)+values.std(0), alpha=.15)
    ax.set(xscale="log", xlabel="DDIM steps (NFE)", ylabel="Mean relative CRPS regret", title="Validation shared-step regret")
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"shared_step_regret.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/step_study.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Chronologically sorted CSV or .npy [time, variables]")
    source.add_argument("--synthetic", action="store_true", help="Pipeline demonstration only")
    parser.add_argument("--columns", help="Comma-separated numeric CSV fields")
    parser.add_argument("--output", required=True, help="New, empty run directory")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:N")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--smoke", action="store_true", help="Small model/one epoch; not scientific evidence")
    parser.add_argument("--evaluate-only", action="store_true", help="Reuse checkpoints in --output, no retraining")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not (args.device == "cpu" or args.device.startswith("cuda")):
        parser.error("Use cpu or cuda:N for reproducible validation RNG isolation")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 - 30000 for s in args.seeds):
        parser.error("Seeds must be distinct nonnegative integers below 2**32-30000")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.smoke:
        config = copy.deepcopy(config)
        config["data"].update(history=12, horizon=4, stride=16)
        config["dlinear"]["kernel"] = 3
        config["train"].update(dlinear_epochs=1, diffusion_epochs=1, batch_size=8)
        config["diffusion"].update(layers=1, channels=8, nheads=2, diffusion_embedding_dim=16)
        config["model"].update(timeemb=8, featureemb=4)
        config["evaluation"].update(steps=[2, 4, 8], samples=4, batch_size=2)
    if args.synthetic:
        values = synthetic_series()
        names = [f"variable_{i}" for i in range(values.shape[1])]
    else:
        values, names = load_series(args.data, args.columns)
    train_config, eval_config = config["train"], config["evaluation"]
    if min(train_config["dlinear_epochs"], train_config["diffusion_epochs"], train_config["batch_size"],
           eval_config["batch_size"], eval_config["samples"]) < 1:
        raise ValueError("Epochs, batch sizes and sample count must be positive")
    steps = eval_config["steps"]
    if not steps or len(set(steps)) != len(steps) or any(not isinstance(k, int) or not 1 <= k <= config["diffusion"]["num_steps"] for k in steps):
        raise ValueError("Use distinct integer sampling steps in [1, training steps]")
    eval_config["steps"] = sorted(steps)
    if eval_config["near_optimal_tolerance"] < 0:
        raise ValueError("Near-optimal tolerance must be nonnegative")
    datasets, scaler = prepare(values, **config["data"])
    output = Path(args.output)
    if args.evaluate_only:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["config"] != config or manifest["scaler"] != scaler or manifest["variables"] != names or manifest["seeds"] != args.seeds:
            raise ValueError("Evaluation-only requires original config, data, variable order and seeds")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing to overwrite a nonempty run directory")
        output.mkdir(parents=True, exist_ok=True)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dump_json(output / "manifest.json", {"config": config, "scaler": scaler, "variables": names,
                  "seeds": args.seeds, "synthetic": args.synthetic, "smoke": args.smoke,
                  "device": args.device, "torch": torch.__version__, "numpy": np.__version__,
                  "python": platform.python_version(), "base_commit": commit})
    rows, joint = [], []
    for seed in args.seeds:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        seed_everything(seed)
        baseline = DLinear(config["data"]["history"], config["data"]["horizon"], len(names), **config["dlinear"]).to(args.device)
        if args.evaluate_only:
            baseline.load_state_dict(torch.load(seed_dir / "dlinear.pt", map_location=args.device, weights_only=True))
        else:
            fit(baseline, datasets, train_config, args.device, seed, seed_dir / "dlinear.pt", stage1=True)
        baseline.eval().requires_grad_(False)
        models = {}
        for method in ("direct", "residual"):
            # Same initialization AND training data/noise/dropout sequence for each arm.
            seed_everything(seed)
            model = ForecastCSDI(len(names), config, args.device).to(args.device)
            if args.evaluate_only:
                model.load_state_dict(torch.load(seed_dir / f"{method}.pt", map_location=args.device, weights_only=True))
                model.eval()
            else:
                fit(model, datasets, train_config, args.device, seed, seed_dir / f"{method}.pt",
                    baseline=baseline if method == "residual" else None)
            models[method] = model
        assert sum(p.numel() for p in models["direct"].parameters()) == sum(p.numel() for p in models["residual"].parameters())
        scores, joint_scores = evaluate(models, baseline, datasets, eval_config, scaler, names, args.device, seed)
        rows.extend(scores)
        joint.extend(joint_scores)
        dump_json(seed_dir / "residual_diagnostics.json", residual_diagnostics(baseline, datasets["val"], train_config["batch_size"], args.device, names))
        write_csv(output / "per_variable.csv", rows)
        write_csv(output / "joint_scores.csv", joint)
    summary = step_summary(rows, eval_config["near_optimal_tolerance"])
    dump_json(output / "summary.json", summary)
    plot_curves(rows, summary, output)
    print(f"Finished: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
