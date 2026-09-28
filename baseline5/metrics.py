from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def crps_map(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pointwise ensemble CRPS for samples [N,S,C,T], target [N,C,T]."""
    values = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if values.ndim != 4 or truth.shape != (
        values.shape[0],
        values.shape[2],
        values.shape[3],
    ):
        raise ValueError("Expected samples [N,S,C,T] and target [N,C,T].")
    term1 = np.mean(np.abs(values - truth[:, None]), axis=1)
    ordered = np.sort(values, axis=1)
    size = values.shape[1]
    coefficients = (2 * np.arange(size) - size + 1).reshape(1, size, 1, 1)
    mean_pairwise = 2.0 * np.sum(coefficients * ordered, axis=1) / size**2
    return term1 - 0.5 * mean_pairwise


def macro_ncrps(samples: np.ndarray, target: np.ndarray) -> float:
    scores = crps_map(samples, target)
    per_channel = [
        np.sum(scores[:, channel])
        / (np.sum(np.abs(target[:, channel])) + 1e-8)
        for channel in range(target.shape[1])
    ]
    return float(np.mean(per_channel))


def compute_precision_recall(
    generated: np.ndarray, real: np.ndarray, k: int = 5
) -> tuple[float, float]:
    generated_values = np.asarray(generated, dtype=np.float64).reshape(
        len(generated), -1
    )
    real_values = np.asarray(real, dtype=np.float64).reshape(len(real), -1)
    if len(generated_values) < 2 or len(real_values) < 2:
        return float("nan"), float("nan")
    effective_k = min(k, len(generated_values) - 1, len(real_values) - 1)

    def radius(values: np.ndarray) -> np.ndarray:
        neighbours = NearestNeighbors(n_neighbors=effective_k + 1).fit(values)
        distances, _ = neighbours.kneighbors(values)
        return distances[:, effective_k]

    real_radius = radius(real_values)
    generated_radius = radius(generated_values)
    distance, index = NearestNeighbors(n_neighbors=1).fit(real_values).kneighbors(
        generated_values
    )
    precision = np.mean(distance[:, 0] <= real_radius[index[:, 0]])
    distance, index = NearestNeighbors(n_neighbors=1).fit(
        generated_values
    ).kneighbors(real_values)
    recall = np.mean(distance[:, 0] <= generated_radius[index[:, 0]])
    return float(precision), float(recall)


def summarize(
    samples: np.ndarray,
    target: np.ndarray,
    labels: list[str],
    target_mean: np.ndarray,
    target_std: np.ndarray,
    seed: int = 42,
    precision_recall_k: int = 5,
    max_precision_samples: int = 10_000,
) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if values.ndim != 4 or truth.shape != (
        values.shape[0],
        values.shape[2],
        values.shape[3],
    ):
        raise ValueError("Expected samples [N,S,C,T] and target [N,C,T].")
    channels = values.shape[2]
    if len(labels) != channels:
        raise ValueError("The number of labels must match the channel count.")
    mean = np.asarray(target_mean, dtype=np.float64).reshape(-1)
    std = np.maximum(np.asarray(target_std, dtype=np.float64).reshape(-1), 1e-8)
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError("target_mean/std must contain one value per channel.")
    values_z = (values - mean.reshape(1, 1, channels, 1)) / std.reshape(
        1, 1, channels, 1
    )
    truth_z = (truth - mean.reshape(1, channels, 1)) / std.reshape(
        1, channels, 1
    )
    error = values.mean(axis=1) - truth
    error_z = values_z.mean(axis=1) - truth_z
    scores = crps_map(values, truth)
    result: dict[str, float] = {}
    channel_ncrps: list[float] = []
    for channel, label in enumerate(labels):
        channel_target = truth[:, channel]
        channel_crps = scores[:, channel]
        lower90 = np.quantile(values[:, :, channel], 0.05, axis=1)
        upper90 = np.quantile(values[:, :, channel], 0.95, axis=1)
        lower95 = np.quantile(values[:, :, channel], 0.025, axis=1)
        upper95 = np.quantile(values[:, :, channel], 0.975, axis=1)
        generated_pool = values_z[:, :, channel].reshape(-1, values.shape[-1])
        if len(generated_pool) > max_precision_samples:
            random = np.random.default_rng(seed + 1000 + channel)
            selected = random.choice(
                len(generated_pool), size=max_precision_samples, replace=False
            )
            generated_pool = generated_pool[selected]
        precision, recall = compute_precision_recall(
            generated_pool, truth_z[:, channel], precision_recall_k
        )
        ncrps = float(
            np.sum(channel_crps) / (np.sum(np.abs(channel_target)) + 1e-8)
        )
        channel_ncrps.append(ncrps)
        result[f"{label}_RMSE"] = float(np.sqrt(np.mean(error[:, channel] ** 2)))
        result[f"{label}_MAE"] = float(np.mean(np.abs(error[:, channel])))
        result[f"{label}_RMSE_Z"] = float(
            np.sqrt(np.mean(error_z[:, channel] ** 2))
        )
        result[f"{label}_MAE_Z"] = float(np.mean(np.abs(error_z[:, channel])))
        result[f"{label}_CRPS"] = float(np.mean(channel_crps))
        result[f"{label}_nCRPS"] = ncrps
        result[f"{label}_Coverage90"] = float(
            np.mean((channel_target >= lower90) & (channel_target <= upper90))
        )
        result[f"{label}_IntervalWidth90"] = float(np.mean(upper90 - lower90))
        result[f"{label}_Precision_Z"] = precision
        result[f"{label}_Recall_Z"] = recall
        result[f"{label}_CR"] = float(
            np.mean((channel_target >= lower95) & (channel_target <= upper95))
        )
        result[f"{label}_IW"] = float(np.mean(upper95 - lower95))
        squared_error = np.sum(error[:, channel] ** 2)
        total_variation = np.sum((channel_target - channel_target.mean()) ** 2)
        result[f"{label}_R2"] = float(1 - squared_error / total_variation) if total_variation > 0 else float("nan")
        interval = interval_score(channel_target, lower95, upper95, alpha=0.05)
        result[f"{label}_IS"] = float(interval.mean())
        result[f"{label}_IS95"] = result[f"{label}_IS"]
        result[f"{label}_IS_Z"] = float(interval.mean() / std[channel])
    result["mean_nCRPS"] = float(
        np.sum(scores) / (np.sum(np.abs(truth)) + 1e-8)
    )
    result["macro_nCRPS"] = float(np.mean(channel_ncrps))
    return result


def interval_score(target, lower, upper, alpha=0.05):
    """Negatively oriented central (1-alpha) interval score (Winkler score)."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    y, lo, hi = np.broadcast_arrays(np.asarray(target, dtype=float), lower, upper)
    if not np.isfinite(y).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("Interval score inputs must be finite")
    if np.any(lo > hi):
        raise ValueError("lower cannot exceed upper")
    return hi - lo + 2 / alpha * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def joint_scores(samples, target, target_mean, target_std, p=0.5):
    """Per-day ES and VS for channel-major 96-D standardized trajectories.

    ES uses the empirical distribution (S**2 denominator, diagonal included).
    VS uses p=0.5 and uniform weights summing to 1 over unordered i<j pairs.
    These conventions must be identical across every compared baseline.
    """
    from scipy.spatial.distance import pdist

    values, truth = np.asarray(samples, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if values.ndim != 4 or truth.shape != (values.shape[0], values.shape[2], values.shape[3]):
        raise ValueError("Expected [N,S,C,T] and [N,C,T]")
    if min(values.shape) < 1 or values.shape[2] * values.shape[3] < 2 or not 0 < p < 2:
        raise ValueError("Require nonempty data, dimension >= 2, and 0 < p < 2")
    mean, std = np.asarray(target_mean, dtype=float).reshape(-1), np.asarray(target_std, dtype=float).reshape(-1)
    if mean.shape != (values.shape[2],) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("One positive training std and mean per channel required")
    if not all(np.isfinite(a).all() for a in (values, truth, mean, std)):
        raise ValueError("Scoring inputs must be finite")
    x = ((values - mean[None, None, :, None]) / std[None, None, :, None]).reshape(len(values), values.shape[1], -1)
    y = ((truth - mean[None, :, None]) / std[None, :, None]).reshape(len(truth), -1)
    left, right = np.triu_indices(x.shape[-1], k=1)
    es, vs = [], []
    for ensembles, observation in zip(x, y):
        # pdist stores each unordered pair once; equal to half the ordered sum.
        es.append(np.linalg.norm(ensembles - observation, axis=1).mean()
                  - pdist(ensembles, metric="euclidean").sum() / len(ensembles) ** 2)
        expected_difference = np.abs(ensembles[:, left] - ensembles[:, right]) ** p
        observed_difference = np.abs(observation[left] - observation[right]) ** p
        vs.append(np.mean((observed_difference - expected_difference.mean(axis=0)) ** 2))
    return np.asarray(es), np.asarray(vs)


def save_additional_scores(samples, target, mean, std, dates, labels, output_dir):
    """Joint scores and per-day univariate scores for paired/block-bootstrap analysis."""
    import csv
    import json

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    es, vs = joint_scores(samples, target, mean, std)
    scores = crps_map(samples, target)
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=1)
    intervals = interval_score(target, lo, hi)
    rows = []
    for day, date in enumerate(dates):
        row = {"date": date, "ES_Z": float(es[day]), "VS_Z": float(vs[day])}
        for channel, label in enumerate(labels):
            row[f"{label}_CRPS"] = float(scores[day, channel].mean())
            row[f"{label}_IS"] = float(intervals[day, channel].mean())
            row[f"{label}_IS_Z"] = float(intervals[day, channel].mean() / std[channel])
        rows.append(row)
    with (destination / "daily_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    definition = {
        "point_estimate": "ensemble mean; R2 over all test days and hours within each channel",
        "constant_target_R2": "undefined; JSON null / CSV blank",
        "interval": {"coverage": 0.95, "alpha": 0.05, "quantile_method": "linear",
                     "IS": "physical units; alias IS95", "IS_Z": "IS divided by training channel std"},
        "joint_dimension": int(target.shape[1] * target.shape[2]),
        "joint_layout": "channel-major: Electricity, Heat, Cooling, PV; hours 0..23",
        "joint_normalization": "training-set per-channel z-score; after PV physical clipping",
        "ES": "alias ES_Z; mean_s ||x_s-y|| - sum_{s,r} ||x_s-x_r||/(2*S^2)",
        "VS": "alias VS_Z; mean_{i<j} (abs(y_i-y_j)^p - mean_s abs(x_si-x_sj)^p)^2",
        "VS_p": 0.5, "VS_weights": "uniform unordered pairs, sum=1",
        "direction": "R2 higher is better; IS, ES, VS lower is better",
        "ensemble_estimator": "empirical distribution (not ensemble-size bias-corrected/fair score)",
        "precision_recall": "legacy nearest-center-radius approximation, not full kNN-ball union",
    }
    (destination / "metric_definitions.json").write_text(json.dumps(definition, indent=2))
    return {"ES": float(es.mean()), "ES_Z": float(es.mean()),
            "VS": float(vs.mean()), "VS_Z": float(vs.mean())}


def compute_global_pearson_matrix(total_data: np.ndarray) -> np.ndarray:
    values = np.asarray(total_data, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected [N,C,T], got {values.shape}.")
    flattened = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    return np.corrcoef(flattened, rowvar=False)


def _save_heatmap(
    matrix: np.ndarray,
    title: str,
    save_path: str | Path,
    labels: list[str],
    cmap: str,
    vmin: float,
    vmax: float,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def save_global_pearson_plot(
    total_real: np.ndarray, save_path: str | Path, labels: list[str]
) -> Path:
    return _save_heatmap(
        compute_global_pearson_matrix(total_real),
        "Global Pearson Correlation",
        save_path,
        labels,
        "coolwarm",
        -1.0,
        1.0,
    )


def save_global_pearson_comparison(
    total_real: np.ndarray,
    total_generated: np.ndarray,
    output_dir: str | Path,
    labels: list[str],
) -> dict[str, float]:
    real_correlation = compute_global_pearson_matrix(total_real)
    generated = np.asarray(total_generated, dtype=np.float64)
    generated_correlation = compute_global_pearson_matrix(
        generated.reshape(-1, generated.shape[2], generated.shape[3])
    )
    difference = np.abs(generated_correlation - real_correlation)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for matrix, title, filename, cmap, vmin, vmax in (
        (real_correlation, "Real Global Pearson", "real_global_pearson.png", "coolwarm", -1.0, 1.0),
        (generated_correlation, "Generated Global Pearson", "generated_global_pearson.png", "coolwarm", -1.0, 1.0),
        (difference, "Pearson Absolute Error", "pearson_difference.png", "Reds", 0.0, 1.0),
    ):
        _save_heatmap(matrix, title, destination / filename, labels, cmap, vmin, vmax)
    return {
        "Pearson_MAE": float(difference.mean()),
        "Pearson_RMSE": float(np.sqrt(np.mean(difference**2))),
    }


def save_random_timeseries_plots(
    total_real: np.ndarray,
    total_generated: np.ndarray,
    dates: list[str],
    output_dir: str | Path,
    labels: list[str],
    seed: int = 42,
    n_plots: int = 50,
) -> list[Path]:
    import matplotlib.pyplot as plt

    real = np.asarray(total_real)
    generated = np.asarray(total_generated)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected = np.random.default_rng(seed).choice(
        real.shape[0], size=min(n_plots, real.shape[0]), replace=False
    )
    saved: list[Path] = []
    for index in selected:
        figure, axes = plt.subplots(
            len(labels), 1, figsize=(10, 3 * len(labels)), sharex=True
        )
        if len(labels) == 1:
            axes = [axes]
        for channel, label in enumerate(labels):
            for scenario in range(min(100, generated.shape[1])):
                axes[channel].plot(
                    generated[index, scenario, channel],
                    color="red",
                    alpha=0.1,
                    linewidth=1,
                )
            axes[channel].plot(
                real[index, channel],
                color="black",
                linewidth=2,
                linestyle="--",
                label="Ground Truth",
            )
            axes[channel].set_title(label)
            axes[channel].grid(True, alpha=0.3)
            axes[channel].legend()
        figure.suptitle(f"Generated scenarios - {dates[index]}")
        figure.tight_layout()
        save_path = destination / f"{dates[index]}.png"
        figure.savefig(save_path, dpi=200)
        plt.close(figure)
        saved.append(save_path)
    return saved
