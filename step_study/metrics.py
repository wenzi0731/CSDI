"""Empirical ensemble scores and validation-selected shared-step diagnostics."""
import numpy as np
import torch


def crps(samples, truth):
    """Empirical CRPS [batch, horizon, variable], no quadratic pair tensor."""
    count = samples.shape[1]
    ordered = samples.sort(dim=1).values
    weights = (2 * torch.arange(1, count + 1, device=samples.device) - count - 1)
    spread = (ordered * weights[None, :, None, None]).sum(1) / count ** 2
    return (samples - truth[:, None]).abs().mean(1) - spread


def energy_score(samples, truth):
    """Joint trajectory energy score; standardize variables before calling."""
    flattened = samples.flatten(2)
    return ((flattened - truth.flatten(1)[:, None]).norm(dim=-1).mean(1)
            - .5 * torch.cdist(flattened, flattened).mean((1, 2)))


def ncrps_from_sums(crps_sum_original, abs_target_sum_original):
    """CSDI-style magnitude normalization, using exact empirical ensemble CRPS.

    Inputs are per-variable sums over all evaluation windows/horizons. Reject an
    all-zero target variable: its normalized score is undefined, not zero.
    """
    if not torch.isfinite(abs_target_sum_original).all() or (abs_target_sum_original <= 0).any():
        raise ValueError("nCRPS undefined for nonfinite or all-zero original-scale targets")
    return crps_sum_original / abs_target_sum_original


def relative_loss_rows(rows, epsilon=1e-8):
    """Within each method/split/seed/variable: (M(T)-min M)/(min M+eps).

    Test-grid minima are descriptive oracles, never used for model selection.
    """
    if epsilon <= 0:
        raise ValueError("relative-loss epsilon must be positive")
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("seed", "split", "method", "variable"))
        groups.setdefault(key, []).append(row)
    result = []
    for group in groups.values():
        best = min(group, key=lambda row: (row["ncrps"], row["diffusion_steps"]))
        for row in sorted(group, key=lambda row: row["diffusion_steps"]):
            result.append({**{k: row[k] for k in ("seed", "split", "method", "variable", "diffusion_steps", "ncrps")},
                           "minimum_ncrps": best["ncrps"], "optimal_diffusion_steps": best["diffusion_steps"],
                           "relative_loss": (row["ncrps"] - best["ncrps"]) / (best["ncrps"] + epsilon),
                           "reference": "same_split_grid_minimum_descriptive" if row["split"] == "test" else "validation_grid_minimum"})
    return result


def step_summary(rows, tolerance=.02, epsilon=1e-8):
    """Argmins selected on validation only; test minima are descriptive oracles."""
    result = []
    for seed, method in sorted({(r["seed"], r["method"]) for r in rows}):
        selected = [r for r in rows if r["seed"] == seed and r["method"] == method]
        steps = sorted({r["diffusion_steps"] for r in selected})
        names = sorted({r["variable"] for r in selected})
        matrices = {}
        for split in ("val", "test"):
            lookup = {(r["diffusion_steps"], r["variable"]): r["ncrps"]
                      for r in selected if r["split"] == split}
            matrices[split] = np.array([[lookup[k, v] for v in names] for k in steps])
        validation, test = matrices["val"], matrices["test"]
        best = validation.argmin(0)
        minima = validation.min(0)
        # Common step selected by mean nCRPS; minimum mean regret is a separate diagnostic.
        shared = int(validation.mean(1).argmin())
        regret = (validation - minima) / (minima + epsilon)
        near = validation <= minima + tolerance * (minima + epsilon)
        oracle_test = test.min(0)
        result.append({
            "seed": seed, "method": method, "variables": names, "steps": steps,
            "validation_optimal_steps": [steps[i] for i in best],
            "validation_std_log_optimal_steps": float(np.log(np.array(steps)[best]).std()),
            "validation_shared_steps": steps[shared],
            "validation_shared_regret_curve": regret.mean(1).tolist(),
            "validation_minimum_shared_relative_loss_G": float(regret.mean(1).min()),
            "validation_minimum_regret_steps": steps[int(regret.mean(1).argmin())],
            "near_optimal_relative_tolerance": tolerance,
            "validation_near_optimal_steps": {name: np.array(steps)[near[:, i]].tolist()
                                              for i, name in enumerate(names)},
            "validation_common_near_optimal_steps": np.array(steps)[near.all(1)].tolist(),
            "test_mean_ncrps_at_validation_shared_steps": float(test[shared].mean()),
            "test_shared_minus_variable_selected_ncrps": float(
                (test[shared] - test[best, np.arange(len(names))]).mean()),
            "test_oracle_relative_regret_curve_descriptive": (
                (test - oracle_test) / (oracle_test + epsilon)).mean(1).tolist(),
        })
    return result
