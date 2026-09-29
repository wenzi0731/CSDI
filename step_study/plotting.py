"""Four-variable nCRPS and relative-loss panels; also supports offline replotting."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def select_plot_variables(names, selection=None):
    chosen = [name.strip() for name in selection.split(",")] if selection else list(names)
    if len(chosen) != 4 or len(set(chosen)) != 4 or not set(chosen).issubset(names):
        raise ValueError("Provide four input variables, or use --plot-variables with exactly four distinct input names")
    return chosen


def curve_values(rows, method, split, variable, metric):
    selected = [r for r in rows if r["method"] == method and r["split"] == split and r["variable"] == variable]
    steps = sorted({r["diffusion_steps"] for r in selected})
    seeds = sorted({r["seed"] for r in selected})
    lookup = {(r["diffusion_steps"], r["seed"]): r[metric] for r in selected}
    if not steps or len(lookup) != len(selected) or len(lookup) != len(steps) * len(seeds):
        raise ValueError(f"Missing or duplicate curve observations: {method}/{split}/{variable}")
    values = np.array([[lookup[t, seed] for seed in seeds] for t in steps])
    return steps, values


def plot_curves(rows, relative_rows, summary, output, names, demonstration=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = select_plot_variables(list(dict.fromkeys(r["variable"] for r in rows)), ",".join(names))
    output = Path(output)
    methods = (("direct", "Direct CSDI", "#2563a6", "o"),
               ("residual", "DLinear + Residual CSDI", "#ce5a28", "s"))
    demo = "DEMONSTRATION ONLY · " if demonstration else ""
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    for split in ("val", "test"):
        split_label = "Validation" if split == "val" else "Test"
        for metric, source, stem in (("ncrps", rows, "ncrps_vs_diffusion_steps"),
                                     ("relative_loss", relative_rows, "relative_loss_vs_diffusion_steps")):
            fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), layout="constrained")
            title = "nCRPS" if metric == "ncrps" else "Per-variable relative loss"
            subtitle = "Mean ± SD across training seeds"
            if metric == "relative_loss":
                subtitle += "; each method uses its own minimum"
                if split == "test":
                    subtitle += "\nTest minima are descriptive, not used for step selection"
            fig.suptitle(f"{demo}{split_label}: {title}\n{subtitle}", fontsize=12)
            for ax, variable in zip(axes.flat, names):
                for method, label, color, marker in methods:
                    steps, values = curve_values(source, method, split, variable, metric)
                    mean, sd = values.mean(1), values.std(1)
                    ax.plot(steps, mean, marker=marker, color=color, label=label, linewidth=1.8)
                    ax.fill_between(steps, np.maximum(0, mean-sd), mean+sd, color=color, alpha=.15)
                ax.set(title=variable, xlabel="Total diffusion steps T",
                       ylabel="nCRPS (lower is better)" if metric == "ncrps" else "Relative loss rᵢ(T)",
                       xticks=steps, ylim=(0, None))
                if metric == "relative_loss":
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            save(fig, f"{stem}_{split}")

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for method, label, color, marker in methods:
        selected = [r for r in summary if r["method"] == method]
        values = np.array([r["validation_shared_regret_curve"] for r in selected])
        steps = selected[0]["steps"]
        mean, sd = values.mean(0), values.std(0)
        ax.plot(steps, mean, marker=marker, color=color, label=label)
        ax.fill_between(steps, np.maximum(0, mean-sd), mean+sd, alpha=.15, color=color)
    ax.set(xlabel="Total diffusion steps T", ylabel="Mean relative loss across all variables",
           title=f"{demo}Validation: shared-step relative loss", xticks=steps, ylim=(0, None))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, "shared_step_relative_loss_val")
    (output / "plot_metadata.json").write_text(json.dumps({
        "variables": names, "x_axis": "total training and DDPM diffusion steps T",
        "uncertainty": "population SD across training seeds; lower band clipped to zero",
        "relative_loss_aggregation": "per-seed minima and losses, then mean and SD",
        "demonstration": demonstration,
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Completed total-diffusion-step run directory")
    parser.add_argument("--plot-variables", help="Four comma-separated names; otherwise use saved names")
    args = parser.parse_args()
    output = Path(args.run)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("experiment") != "paper_daily_total_diffusion_steps_v4":
        raise ValueError("Expected a completed total-T study, not a legacy DDIM run")
    names = select_plot_variables(manifest["variables"], args.plot_variables or ",".join(manifest["plot_variables"]))
    def read_rows(path):
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            row["seed"], row["diffusion_steps"] = int(row["seed"]), int(row["diffusion_steps"])
            row["ncrps"] = float(row["ncrps"])
            if "relative_loss" in row:
                row["relative_loss"] = float(row["relative_loss"])
        return rows
    plot_curves(read_rows(output / "per_variable.csv"), read_rows(output / "per_variable_relative_loss.csv"),
                json.loads((output / "summary.json").read_text()), output, names,
                demonstration=manifest["synthetic"] or manifest["smoke"])


if __name__ == "__main__":
    main()
