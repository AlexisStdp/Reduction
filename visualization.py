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
    return bottleneck_from_inputs(model, x)


def bottleneck_from_inputs(model, x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

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


def plot_model_predictions(model, function="compositeSinePlus", p=None, interval=(-1, 1), n_points=400, train_x=None, train_y=None):
    x, y_true, y_pred = evaluate_model_on_grid(model, function=function, p=p, interval=interval, n_points=n_points)
    names = _component_names(function, y_true.shape[1])
    fig, axes = _component_grid(y_true.shape[1])

    for idx in range(y_true.shape[1]):
        ax = axes[idx]
        ax.plot(x[:, 0], y_true[:, idx], label="true", color="blue")
        ax.plot(x[:, 0], y_pred[:, idx], label="prediction", color="orange", linestyle="--")
        if train_x is not None and train_y is not None:
            ax.scatter(train_x[:, 0], train_y[:, idx], label="train data" if idx == 0 else None, color="black", s=12, alpha=0.35)
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


def plot_bottleneck_representation(
    model,
    reference_function="scaledSine",
    p=None,
    interval=(-1, 1),
    n_points=400,
    train_x=None,
):
    p = DEFAULT_P if p is None else p
    x, z = bottleneck_on_grid(model, interval=interval, n_points=n_points)
    reference = function_library(reference_function, p=p)(x)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    train_reference = None
    train_z = None
    if train_x is not None:
        train_x, train_z = bottleneck_from_inputs(model, train_x)
        train_reference = function_library(reference_function, p=p)(train_x)
        if train_reference.ndim == 1:
            train_reference = train_reference.reshape(-1, 1)

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
        if train_x is not None:
            axes[idx, 0].scatter(train_x[:, 0], train_reference[:, 0], color="blue", s=12, alpha=0.2)
            axes[idx, 0].scatter(train_x[:, 0], train_z[:, idx], color="green", s=12, alpha=0.2)
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
        if train_reference is not None:
            axes[idx, 2].scatter(train_reference[:, 0], train_z[:, idx], s=10, alpha=0.15, color="black")
        _plot_linear_fit(axes[idx, 2], ref_flat, z_flat)
        axes[idx, 2].set_title(f"raw scatter, corr={corr:.3f}")
        axes[idx, 2].set_xlabel(f"{reference_function}(x)")
        axes[idx, 2].set_ylabel(f"bottleneck {idx}")
        axes[idx, 2].grid(True, alpha=0.3)

    fig.suptitle("Bottleneck representation")
    fig.tight_layout()
    return fig, axes


def plot_bottleneck_plane(model, reference_function="scaledSine", p=None, interval=(-1, 1), n_points=400, train_x=None):
    p = DEFAULT_P if p is None else p
    x, z = bottleneck_on_grid(model, interval=interval, n_points=n_points)
    if z.shape[1] < 2:
        raise ValueError("Expected a bottleneck with at least 2 dimensions.")

    reference = function_library(reference_function, p=p)(x)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)
    train_z = None
    train_reference = None
    if train_x is not None:
        _, train_z = bottleneck_from_inputs(model, train_x)
        train_reference = function_library(reference_function, p=p)(train_x)
        if train_reference.ndim == 1:
            train_reference = train_reference.reshape(-1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), squeeze=False)
    axes = axes.ravel()

    by_x = axes[0].scatter(z[:, 0], z[:, 1], c=x[:, 0], cmap="viridis", s=16, alpha=0.8)
    if train_z is not None:
        axes[0].scatter(train_z[:, 0], train_z[:, 1], facecolors="none", edgecolors="black", s=18, alpha=0.35)
    axes[0].set_title("bottleneck plane colored by x")
    axes[0].set_xlabel("bottleneck 0")
    axes[0].set_ylabel("bottleneck 1")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(by_x, ax=axes[0], label="x")

    by_reference = axes[1].scatter(z[:, 0], z[:, 1], c=reference[:, 0], cmap="coolwarm", s=16, alpha=0.8)
    if train_z is not None:
        axes[1].scatter(train_z[:, 0], train_z[:, 1], facecolors="none", edgecolors="black", s=18, alpha=0.35)
    axes[1].set_title(f"bottleneck plane colored by {reference_function}")
    axes[1].set_xlabel("bottleneck 0")
    axes[1].set_ylabel("bottleneck 1")
    axes[1].grid(True, alpha=0.3)
    fig.colorbar(by_reference, ax=axes[1], label=reference_function)

    fig.suptitle("2D bottleneck plane")
    fig.tight_layout()
    return fig, axes


def plot_stack2_kink_boundaries(model, reference_function="scaledSine", p=None, interval=(-1, 1), n_points=400, train_x=None):
    p = DEFAULT_P if p is None else p
    if not hasattr(model, "stack2") or model.stack2.affine1.in_features != 2:
        raise ValueError("Expected a second stack with 2D inputs.")

    x, z = bottleneck_on_grid(model, interval=interval, n_points=n_points)
    reference = function_library(reference_function, p=p)(x)
    if reference.ndim == 1:
        reference = reference.reshape(-1, 1)

    train_z = None
    if train_x is not None:
        _, train_z = bottleneck_from_inputs(model, train_x)

    v = model.stack2.affine1.weight.detach().cpu().numpy()
    b = model.stack2.affine1.bias.detach().cpu().numpy()
    w = model.stack2.affine2.weight.detach().cpu().numpy()
    strengths = np.linalg.norm(v, axis=1) * np.linalg.norm(w, axis=0)
    strengths = strengths / strengths.max() if strengths.size and strengths.max() > 0 else strengths

    crossings = _stack2_boundary_crossings(x, z, reference[:, 0], v, b, strengths)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), squeeze=False)
    axes = axes.ravel()
    top_axes = axes[:2]
    scatters = [
        axes[0].scatter(z[:, 0], z[:, 1], c=x[:, 0], cmap="viridis", s=18, alpha=0.85),
        axes[1].scatter(z[:, 0], z[:, 1], c=reference[:, 0], cmap="coolwarm", s=18, alpha=0.85),
    ]
    titles = ["Stack 2 kink lines colored by x", f"Stack 2 kink lines colored by {reference_function}"]
    labels = ["x", reference_function]

    if train_z is not None:
        for ax in top_axes:
            ax.scatter(train_z[:, 0], train_z[:, 1], facecolors="none", edgecolors="black", s=18, alpha=0.35)

    x_min, x_max = float(z[:, 0].min()), float(z[:, 0].max())
    y_min, y_max = float(z[:, 1].min()), float(z[:, 1].max())
    if train_z is not None:
        x_min = min(x_min, float(train_z[:, 0].min()))
        x_max = max(x_max, float(train_z[:, 0].max()))
        y_min = min(y_min, float(train_z[:, 1].min()))
        y_max = max(y_max, float(train_z[:, 1].max()))
    margin_x = 0.08 * max(x_max - x_min, 1e-6)
    margin_y = 0.08 * max(y_max - y_min, 1e-6)
    bounds = (x_min - margin_x, x_max + margin_x, y_min - margin_y, y_max + margin_y)

    for neuron_idx, (normal, bias) in enumerate(zip(v, b)):
        segment = _line_segment_in_box(normal, bias, bounds)
        if segment is None:
            continue
        line_alpha = 0.15 + 0.55 * float(strengths[neuron_idx]) if strengths.size else 0.3
        line_width = 0.6 + 1.8 * float(strengths[neuron_idx]) if strengths.size else 1.0
        for ax in top_axes:
            ax.plot(segment[:, 0], segment[:, 1], color="black", alpha=line_alpha, linewidth=line_width)

    for ax, scatter, title, label in zip(axes, scatters, titles, labels):
        ax.set_title(title)
        ax.set_xlabel("bottleneck 0")
        ax.set_ylabel("bottleneck 1")
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.grid(True, alpha=0.3)
        fig.colorbar(scatter, ax=ax, label=label)

    x_flat = x[:, 0]
    train_x_flat = train_x[:, 0] if train_x is not None else None
    bottom_titles = ["bottleneck 0 trajectory with stack 2 crossings", "bottleneck 1 trajectory with stack 2 crossings"]
    for bottleneck_idx, ax in enumerate(axes[2:]):
        ax.plot(x_flat, z[:, bottleneck_idx], color="forestgreen", linewidth=1.8, label=f"bottleneck {bottleneck_idx}(x)")
        if train_x is not None and train_z is not None:
            ax.scatter(
                train_x_flat,
                train_z[:, bottleneck_idx],
                facecolors="none",
                edgecolors="black",
                s=18,
                alpha=0.25,
                label="train data" if bottleneck_idx == 0 else None,
            )
        if crossings["count"] > 0:
            ax.scatter(
                crossings["x"],
                crossings["z"][:, bottleneck_idx],
                s=20 + 36 * crossings["strength"],
                c=crossings["reference"],
                cmap="coolwarm",
                alpha=0.8,
                edgecolors="black",
                linewidths=0.25,
                label="stack 2 boundary crossing" if bottleneck_idx == 0 else None,
            )
        ax.set_title(bottom_titles[bottleneck_idx])
        ax.set_xlabel("x")
        ax.set_ylabel(f"bottleneck {bottleneck_idx}")
        ax.grid(True, alpha=0.3)
        if crossings["count"] > 0:
            ax.text(
                0.01,
                0.02,
                f"{crossings['count']} crossings across {len(v)} stack-2 neurons",
                transform=ax.transAxes,
                fontsize=9,
                alpha=0.8,
            )
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(loc="best")

    fig.suptitle("Stack 2 kink boundaries in bottleneck space")
    fig.tight_layout()
    return fig, axes


def _stack2_boundary_crossings(x, z, reference, v, b, strengths, zero_tol=1e-10):
    x_flat = np.asarray(x[:, 0], dtype=float)
    z = np.asarray(z, dtype=float)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    v = np.asarray(v, dtype=float)
    b = np.asarray(b, dtype=float)
    strengths = np.asarray(strengths, dtype=float)

    crossing_x = []
    crossing_z = []
    crossing_strength = []
    crossing_reference = []

    preactivation = z @ v.T + b
    for neuron_idx in range(preactivation.shape[1]):
        h = preactivation[:, neuron_idx]
        for idx in range(len(h) - 1):
            h0 = h[idx]
            h1 = h[idx + 1]
            crossing = None
            if abs(h0) <= zero_tol and abs(h1) <= zero_tol:
                continue
            if abs(h0) <= zero_tol:
                crossing = 0.0
            elif abs(h1) <= zero_tol:
                crossing = 1.0
            elif h0 * h1 < 0:
                crossing = h0 / (h0 - h1)
            if crossing is None:
                continue
            crossing = float(np.clip(crossing, 0.0, 1.0))
            x_cross = x_flat[idx] + crossing * (x_flat[idx + 1] - x_flat[idx])
            z_cross = z[idx] + crossing * (z[idx + 1] - z[idx])
            ref_cross = reference[idx] + crossing * (reference[idx + 1] - reference[idx])
            crossing_x.append(x_cross)
            crossing_z.append(z_cross)
            crossing_strength.append(float(strengths[neuron_idx]) if strengths.size else 0.0)
            crossing_reference.append(ref_cross)

    if not crossing_x:
        return {
            "count": 0,
            "x": np.empty((0,), dtype=float),
            "z": np.empty((0, z.shape[1]), dtype=float),
            "strength": np.empty((0,), dtype=float),
            "reference": np.empty((0,), dtype=float),
        }

    return {
        "count": len(crossing_x),
        "x": np.asarray(crossing_x, dtype=float),
        "z": np.asarray(crossing_z, dtype=float),
        "strength": np.asarray(crossing_strength, dtype=float),
        "reference": np.asarray(crossing_reference, dtype=float),
    }


def plot_stack_kink_contributions(
    stack,
    output_names=None,
    interval=None,
    title="ReLU kink contributions",
    min_abs_v=1e-12,
    contribution_mode="v_abs_w",
):
    v = stack.affine1.weight.detach().cpu().numpy()
    b = stack.affine1.bias.detach().cpu().numpy()
    w = stack.affine2.weight.detach().cpu().numpy()

    if v.shape[1] != 1:
        raise ValueError("Kink plots are only defined here for stacks with 1D inputs.")

    incoming = v[:, 0]
    keep = np.isfinite(incoming) & np.isfinite(b) & (np.abs(incoming) >= min_abs_v)
    skipped = int((~keep).sum())

    if not keep.any():
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.text(0.5, 0.5, "No finite 1D neuron kinks to plot.", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        return fig, np.array([ax])

    kink_positions = -b[keep] / incoming[keep]
    contributions, y_label = _kink_contributions(incoming[keep], w[:, keep], contribution_mode)
    order = np.argsort(kink_positions)
    kink_positions = kink_positions[order]
    contributions = contributions[:, order]

    n_outputs = contributions.shape[0]
    if output_names is None:
        output_names = [f"output_{idx}" for idx in range(n_outputs)]

    fig, axes = _component_grid(n_outputs)
    outside_count = 0
    if interval is not None:
        outside_count = int(((kink_positions < interval[0]) | (kink_positions > interval[1])).sum())

    for idx in range(n_outputs):
        ax = axes[idx]
        values = contributions[idx]
        positive = values >= 0
        negative = values < 0

        ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
        if interval is not None:
            ax.axvspan(interval[0], interval[1], color="black", alpha=0.04)
            ax.axvline(interval[0], color="black", linestyle="--", linewidth=1, alpha=0.45)
            ax.axvline(interval[1], color="black", linestyle="--", linewidth=1, alpha=0.45)
        if positive.any():
            ax.vlines(kink_positions[positive], 0.0, values[positive], color="royalblue", alpha=0.5, linewidth=1.0)
            ax.scatter(kink_positions[positive], values[positive], color="royalblue", alpha=0.85, s=18)
        if negative.any():
            ax.vlines(kink_positions[negative], 0.0, values[negative], color="crimson", alpha=0.5, linewidth=1.0)
            ax.scatter(kink_positions[negative], values[negative], color="crimson", alpha=0.85, s=18)
        ax.set_title(output_names[idx])
        ax.set_xlabel("kink position (-b / v)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    _delete_extra_axes(fig, axes, n_outputs)
    notes = []
    if skipped:
        notes.append(f"Skipped {skipped} neurons with tiny or non-finite incoming weights.")
    if interval is not None:
        notes.append(f"{outside_count} / {len(kink_positions)} kinks lie outside the highlighted interval.")
    if notes:
        fig.text(0.01, 0.01, "  ".join(notes), fontsize=9, alpha=0.8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes[:n_outputs]


def plot_model_kink_figures(
    model,
    function="compositeSinePlus",
    p=None,
    interval=(-1, 1),
    n_points=400,
    min_abs_v=1e-12,
    contribution_mode="v_abs_w",
    train_x=None,
):
    if not hasattr(model, "stack1") or not hasattr(model, "stack2"):
        raise AttributeError("Expected a doubleStack-like model with stack1 and stack2 modules.")

    figures = []
    if model.stack1.affine1.in_features == 1:
        figures.append(
            (
                "stack1_kink_contributions",
                plot_stack_kink_contributions(
                    model.stack1,
                    output_names=[f"bottleneck_{idx}" for idx in range(model.stack1.affine2.out_features)],
                    interval=interval,
                    title="Stack 1 ReLU kink contributions",
                    min_abs_v=min_abs_v,
                    contribution_mode=contribution_mode,
                )[0],
            )
        )

    if model.stack2.affine1.in_features == 1:
        _, z = bottleneck_on_grid(model, interval=interval, n_points=n_points)
        stack2_interval = (float(z[:, 0].min()), float(z[:, 0].max()))
        figures.append(
            (
                "stack2_kink_contributions",
                plot_stack_kink_contributions(
                    model.stack2,
                    output_names=_component_names(function, model.stack2.affine2.out_features),
                    interval=stack2_interval,
                    title="Stack 2 ReLU kink contributions",
                    min_abs_v=min_abs_v,
                    contribution_mode=contribution_mode,
                )[0],
            )
        )
    elif model.stack2.affine1.in_features == 2:
        figures.append(
            (
                "stack2_kink_boundaries",
                plot_stack2_kink_boundaries(
                    model,
                    reference_function="scaledSine",
                    p=p,
                    interval=interval,
                    n_points=n_points,
                    train_x=train_x,
                )[0],
            )
        )

    return figures


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


def _kink_contributions(incoming, outgoing, contribution_mode):
    if contribution_mode == "v_abs_w":
        return incoming[None, :] * np.abs(outgoing), "v_k * abs(w_k)"
    if contribution_mode == "w_abs_v":
        return outgoing * np.abs(incoming)[None, :], "w_k * abs(v_k)"
    raise ValueError(f"Unknown contribution_mode: {contribution_mode}")


def _line_segment_in_box(normal, bias, bounds):
    x_min, x_max, y_min, y_max = bounds
    intersections = []

    if abs(normal[1]) > 1e-12:
        for x_value in (x_min, x_max):
            y_value = -(normal[0] * x_value + bias) / normal[1]
            if y_min <= y_value <= y_max:
                intersections.append((x_value, y_value))

    if abs(normal[0]) > 1e-12:
        for y_value in (y_min, y_max):
            x_value = -(normal[1] * y_value + bias) / normal[0]
            if x_min <= x_value <= x_max:
                intersections.append((x_value, y_value))

    if len(intersections) < 2:
        return None

    unique = []
    for point in intersections:
        if not any(np.allclose(point, seen, atol=1e-8) for seen in unique):
            unique.append(point)
    if len(unique) < 2:
        return None
    return np.asarray(unique[:2], dtype=float)
