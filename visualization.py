import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generation import FUNCTION_COMPONENTS, function_library, sample_function_grid


DEFAULT_P = [0.4, 1.0, 2.0, -3.0]


def evaluate_model_on_grid(model, function="compositeSinePlus", p=None, interval=(-1, 1), n_points=400):
    p = DEFAULT_P if p is None else p
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


def bottleneck_on_grid(model, interval=(-1, 1), n_points=400):
    x = np.linspace(interval[0], interval[1], n_points).reshape(-1, 1)

    if not hasattr(model, "stack1"):
        raise AttributeError("Expected a doubleStack-like model with a stack1 bottleneck.")

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    with torch.no_grad():
        z = model.stack1(x_tensor).detach().cpu().numpy()
    if was_training:
        model.train()

    if z.ndim == 1:
        z = z.reshape(-1, 1)
    return x, z


def plot_target_components(function="compositeSinePlus", p=None, interval=(-1, 1), n_points=400):
    p = DEFAULT_P if p is None else p
    x, y = sample_function_grid(function=function, p=p, interval=interval, n_points=n_points)
    names = _component_names(function, y.shape[1])
    fig, axes = _component_grid(y.shape[1])

    for idx in range(y.shape[1]):
        ax = axes[idx]
        ax.plot(x[:, 0], y[:, idx], color="blue")
        ax.set_title(names[idx])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    _delete_extra_axes(fig, axes, y.shape[1])
    fig.suptitle(f"{function} target components")
    fig.tight_layout()
    return fig, axes[: y.shape[1]]


def plot_model_predictions(model, function="compositeSinePlus", p=None, interval=(-1, 1), n_points=400):
    x, y_true, y_pred = evaluate_model_on_grid(model, function=function, p=p, interval=interval, n_points=n_points)
    names = _component_names(function, y_true.shape[1])
    fig, axes = _component_grid(y_true.shape[1])

    for idx in range(y_true.shape[1]):
        ax = axes[idx]
        ax.plot(x[:, 0], y_true[:, idx], label="true", color="blue")
        ax.plot(x[:, 0], y_pred[:, idx], label="prediction", color="orange", linestyle="--")
        ax.set_title(names[idx])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    _delete_extra_axes(fig, axes, y_true.shape[1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Model predictions vs target")
    fig.tight_layout()
    return fig, axes[: y_true.shape[1]]


def plot_residuals(model, function="compositeSinePlus", p=None, interval=(-1, 1), n_points=400):
    x, y_true, y_pred = evaluate_model_on_grid(model, function=function, p=p, interval=interval, n_points=n_points)
    residuals = y_pred - y_true
    names = _component_names(function, y_true.shape[1])
    fig, axes = _component_grid(y_true.shape[1])

    for idx in range(y_true.shape[1]):
        ax = axes[idx]
        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        ax.plot(x[:, 0], residuals[:, idx], color="crimson")
        ax.set_title(names[idx])
        ax.set_xlabel("x")
        ax.set_ylabel("prediction - true")
        ax.grid(True, alpha=0.3)

    _delete_extra_axes(fig, axes, y_true.shape[1])
    fig.suptitle("Prediction residuals")
    fig.tight_layout()
    return fig, axes[: y_true.shape[1]]


def plot_bottleneck_representation(model, reference_function="scaledSine", p=None, interval=(-1, 1), n_points=400):
    p = DEFAULT_P if p is None else p
    x, z = bottleneck_on_grid(model, interval=interval, n_points=n_points)
    reference = function_library(reference_function, p=p)(x)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)

    fig, axes = plt.subplots(z.shape[1], 3, figsize=(14, 3.6 * z.shape[1]), squeeze=False)
    x_flat = x[:, 0]
    ref_flat = reference[:, 0]
    ref_std = _standardize(ref_flat)

    for idx in range(z.shape[1]):
        z_flat = z[:, idx]
        z_std = _standardize(z_flat)
        corr = np.corrcoef(ref_std, z_std)[0, 1]

        axes[idx, 0].plot(x_flat, ref_flat, label=f"{reference_function}(x)", color="blue", linestyle="--")
        axes[idx, 0].plot(x_flat, z_flat, label=f"bottleneck {idx}", color="green")
        axes[idx, 0].set_title(f"raw comparison for bottleneck {idx}")
        axes[idx, 0].set_xlabel("x")
        axes[idx, 0].set_ylabel("raw value")
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)

        axes[idx, 1].plot(
            x_flat,
            ref_std,
            label=f"{reference_function}(x), standardized",
            color="blue",
            linestyle="--",
        )
        axes[idx, 1].plot(x_flat, z_std, label=f"bottleneck {idx}, standardized", color="green")
        axes[idx, 1].set_title(f"standardized comparison for bottleneck {idx}")
        axes[idx, 1].set_xlabel("x")
        axes[idx, 1].set_ylabel("standardized value")
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)

        axes[idx, 2].scatter(ref_flat, z_flat, s=12, alpha=0.65, color="purple")
        _plot_linear_fit(axes[idx, 2], ref_flat, z_flat)
        axes[idx, 2].set_title(f"raw scatter, corr={corr:.3f}")
        axes[idx, 2].set_xlabel(f"{reference_function}(x)")
        axes[idx, 2].set_ylabel(f"bottleneck {idx}")
        axes[idx, 2].grid(True, alpha=0.3)

    fig.suptitle("Bottleneck representation")
    fig.tight_layout()
    return fig, axes


def plot_bottleneck_plane(model, reference_function="scaledSine", p=None, interval=(-1, 1), n_points=400):
    p = DEFAULT_P if p is None else p
    x, z = bottleneck_on_grid(model, interval=interval, n_points=n_points)
    if z.shape[1] < 2:
        raise ValueError("Expected a bottleneck with at least 2 dimensions.")

    reference = function_library(reference_function, p=p)(x)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    axes = axes.ravel()

    by_x = axes[0].scatter(z[:, 0], z[:, 1], c=x[:, 0], cmap="viridis", s=16, alpha=0.8)
    axes[0].set_title("bottleneck plane colored by x")
    axes[0].set_xlabel("bottleneck 0")
    axes[0].set_ylabel("bottleneck 1")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(by_x, ax=axes[0], label="x")

    by_reference = axes[1].scatter(z[:, 0], z[:, 1], c=reference[:, 0], cmap="coolwarm", s=16, alpha=0.8)
    axes[1].set_title(f"bottleneck plane colored by {reference_function}")
    axes[1].set_xlabel("bottleneck 0")
    axes[1].set_ylabel("bottleneck 1")
    axes[1].grid(True, alpha=0.3)
    fig.colorbar(by_reference, ax=axes[1], label=reference_function)

    fig.suptitle("2D bottleneck plane")
    fig.tight_layout()
    return fig, axes


def _component_names(function, n_outputs):
    names = FUNCTION_COMPONENTS.get(function)
    if names is None:
        names = [f"output_{idx}" for idx in range(n_outputs)]
    return names[:n_outputs]


def _component_grid(n_outputs):
    n_cols = min(4, n_outputs)
    n_rows = math.ceil(n_outputs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), squeeze=False)
    return fig, axes.ravel()


def _delete_extra_axes(fig, axes, used_count):
    for idx in range(used_count, len(axes)):
        fig.delaxes(axes[idx])


def _standardize(values):
    values = np.asarray(values)
    std = values.std()
    if std == 0:
        return values - values.mean()
    return (values - values.mean()) / std


def _plot_linear_fit(ax, x, y):
    if len(np.unique(x)) < 2:
        return
    slope, intercept = np.polyfit(x, y, deg=1)
    order = np.argsort(x)
    x_sorted = x[order]
    ax.plot(x_sorted, slope * x_sorted + intercept, color="black", linewidth=1.5, alpha=0.8)
