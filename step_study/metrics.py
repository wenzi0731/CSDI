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


def step_summary(rows, tolerance=.02):
    """Argmins selected on validation only; test minima are descriptive oracles."""
    result = []
    for seed, method in sorted({(r["seed"], r["method"]) for r in rows}):
        selected = [r for r in rows if r["seed"] == seed and r["method"] == method]
        steps = sorted({r["steps"] for r in selected})
        names = sorted({r["variable"] for r in selected})
        matrices = {}
        for split in ("val", "test"):
            lookup = {(r["steps"], r["variable"]): r["crps_scaled"]
                      for r in selected if r["split"] == split}
            matrices[split] = np.array([[lookup[k, v] for v in names] for k in steps])
        validation, test = matrices["val"], matrices["test"]
        best = validation.argmin(0)
        minima = validation.min(0)
        # Average standardized CRPS gives equal units across variables.
        shared = int(validation.mean(1).argmin())
        regret = (validation - minima) / np.maximum(minima, 1e-8)
        near = validation <= minima + tolerance * np.maximum(minima, 1e-8)
        oracle_test = test.min(0)
        result.append({
            "seed": seed, "method": method, "variables": names, "steps": steps,
            "validation_optimal_steps": [steps[i] for i in best],
            "validation_std_log_optimal_steps": float(np.log(np.array(steps)[best]).std()),
            "validation_shared_steps": steps[shared],
            "validation_shared_regret_curve": regret.mean(1).tolist(),
            "near_optimal_relative_tolerance": tolerance,
            "validation_near_optimal_steps": {name: np.array(steps)[near[:, i]].tolist()
                                              for i, name in enumerate(names)},
            "validation_common_near_optimal_steps": np.array(steps)[near.all(1)].tolist(),
            "test_mean_scaled_crps_at_validation_shared_steps": float(test[shared].mean()),
            "test_shared_minus_variable_selected_scaled_crps": float(
                (test[shared] - test[best, np.arange(len(names))]).mean()),
            "test_oracle_relative_regret_curve_descriptive": (
                (test - oracle_test) / np.maximum(oracle_test, 1e-8)).mean(1).tolist(),
        })
    return result
