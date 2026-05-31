import math

import numpy as np
import torch

from data_generation import function_library, sample_function_grid


def evaluate_model(model, function, p, interval, n_points):
    x, y_true = sample_function_grid(function=function, p=p, interval=interval, n_points=n_points)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred = model(x_tensor).detach().cpu().numpy()
    if was_training:
        model.train()
    return x, y_true, y_pred


def prediction_metrics(y_true, y_pred, prefix):
    diff = np.asarray(y_pred) - np.asarray(y_true)
    abs_diff = np.abs(diff)
    return {
        f"{prefix}_mse": float(np.mean(diff**2)),
        f"{prefix}_mae": float(np.mean(abs_diff)),
        f"{prefix}_max_abs_error": float(np.max(abs_diff)),
        f"{prefix}_component_mse": np.mean(diff**2, axis=0).astype(float).tolist(),
        f"{prefix}_component_mae": np.mean(abs_diff, axis=0).astype(float).tolist(),
    }


def reference_fidelity_metrics(y_reference, y_candidate, prefix):
    diff = np.asarray(y_candidate) - np.asarray(y_reference)
    abs_diff = np.abs(diff)
    return {
        f"{prefix}_mse": float(np.mean(diff**2)),
        f"{prefix}_mae": float(np.mean(abs_diff)),
        f"{prefix}_max_abs": float(np.max(abs_diff)),
        f"{prefix}_component_mse": np.mean(diff**2, axis=0).astype(float).tolist(),
    }


def fidelity_metrics(y_baseline, y_reduced):
    return reference_fidelity_metrics(y_baseline, y_reduced, "baseline_vs_reduced")


def relative_delta(new_value, reference_value):
    if reference_value == 0:
        return math.inf if new_value != 0 else 0.0
    return float((new_value - reference_value) / abs(reference_value))


def parameter_count(model):
    return int(sum(param.numel() for param in model.parameters()))


def stack_hidden_counts(model):
    return {
        "stack1_neurons": int(model.stack1.affine1.out_features),
        "stack2_neurons": int(model.stack2.affine1.out_features),
        "total_stack_neurons": int(model.stack1.affine1.out_features + model.stack2.affine1.out_features),
    }


def bottleneck_reference_metrics(model, reference_function, p, interval, n_points):
    x = np.linspace(interval[0], interval[1], n_points).reshape(-1, 1)
    reference = function_library(reference_function, p=p)(x)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = model.stack1(x_tensor).detach().cpu().numpy()
    if was_training:
        model.train()

    ref = reference[:, 0]
    correlations = [safe_corrcoef(ref, z[:, idx]) for idx in range(z.shape[1])]
    abs_correlations = [abs(value) for value in correlations if not math.isnan(value)]
    return {
        "bottleneck_corr_with_scaled_sine": correlations,
        "bottleneck_best_abs_corr_with_scaled_sine": float(max(abs_correlations)) if abs_correlations else math.nan,
    }


def safe_corrcoef(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])
