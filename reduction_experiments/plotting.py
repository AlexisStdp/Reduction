import json
import math

import matplotlib.pyplot as plt
import numpy as np

from data_generation import FUNCTION_COMPONENTS
from visualization import (
    plot_bottleneck_plane,
    plot_bottleneck_representation,
    plot_model_kink_figures,
    plot_model_predictions,
    plot_residuals,
)


def component_names(function, n_outputs):
    names = FUNCTION_COMPONENTS.get(function)
    if names is None:
        names = [f"output_{idx}" for idx in range(n_outputs)]
    return names[:n_outputs]


def plot_baseline_candidate_predictions(
    x,
    y_true,
    y_baseline,
    y_candidate,
    function,
    candidate_label="candidate",
    title="Baseline and candidate predictions",
    train_x=None,
    train_y=None,
):
    n_outputs = y_true.shape[1]
    names = component_names(function, n_outputs)
    n_cols = min(4, n_outputs)
    n_rows = math.ceil(n_outputs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), squeeze=False)
    axes = axes.ravel()

    for idx in range(n_outputs):
        ax = axes[idx]
        ax.plot(x[:, 0], y_true[:, idx], label="target", color="blue", linewidth=1.5)
        ax.plot(x[:, 0], y_baseline[:, idx], label="baseline", color="gray", linestyle="--", linewidth=1.3)
        ax.plot(x[:, 0], y_candidate[:, idx], label=candidate_label, color="orange", linestyle=":", linewidth=1.8)
        if train_x is not None and train_y is not None:
            ax.scatter(train_x[:, 0], train_y[:, idx], label="train data" if idx == 0 else None, color="black", s=12, alpha=0.35)
        ax.set_title(names[idx])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    for idx in range(n_outputs, len(axes)):
        fig.delaxes(axes[idx])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_baseline_reduced_predictions(x, y_true, y_baseline, y_reduced, function):
    return plot_baseline_candidate_predictions(
        x,
        y_true,
        y_baseline,
        y_reduced,
        function,
        candidate_label="reduced",
        title="Baseline and reduced predictions",
    )


def plot_reduction_tradeoff(summary_rows):
    rows = [row for row in summary_rows if row.get("status") == "ok"]
    if not rows:
        return None

    x = np.array([float(row["compression_ratio"]) for row in rows])
    y = np.array([float(row["relative_mse_delta"]) for row in rows])
    labels = [row["variant"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, color="purple", alpha=0.75)
    for idx, label in enumerate(labels):
        ax.annotate(label, (x[idx], y[idx]), fontsize=7, alpha=0.75)
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("compression ratio")
    ax.set_ylabel("relative MSE delta vs baseline")
    ax.set_title("Reduction tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_variant_figures(
    model,
    x,
    y_true,
    y_baseline,
    y_reduced,
    function,
    p,
    interval,
    n_points,
    output_dir,
    file_format,
    dpi,
    stage_records=None,
    kink_contribution_mode="v_abs_w",
    train_x=None,
    train_y=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        (
            "baseline_vs_reduced_predictions",
            plot_baseline_candidate_predictions(
                x,
                y_true,
                y_baseline,
                y_reduced,
                function,
                candidate_label="reduced",
                title="Baseline and reduced predictions",
                train_x=train_x,
                train_y=train_y,
            ),
        ),
        ("reduced_residuals", plot_residuals(model, function=function, p=p, interval=interval, n_points=n_points)[0]),
        (
            "reduced_bottleneck_representation",
            plot_bottleneck_representation(
                model,
                reference_function="scaledSine",
                p=p,
                interval=interval,
                n_points=n_points,
                train_x=train_x,
            )[0],
        ),
    ]
    figures.extend(
        plot_model_kink_figures(
            model,
            function=function,
            p=p,
            interval=interval,
            n_points=n_points,
            contribution_mode=kink_contribution_mode,
            train_x=train_x,
        )
    )
    if model.stack1.affine2.out_features >= 2:
        figures.append(
            (
                "reduced_bottleneck_plane",
                plot_bottleneck_plane(
                    model,
                    reference_function="scaledSine",
                    p=p,
                    interval=interval,
                    n_points=n_points,
                    train_x=train_x,
                )[0],
            )
        )

    for name, fig in figures:
        fig.savefig(output_dir / f"{name}.{file_format}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if stage_records:
        save_reduction_stage_figures(
            stage_records,
            x,
            y_true,
            y_baseline,
            function,
            p,
            interval,
            n_points,
            output_dir / "stages",
            file_format,
            dpi,
            kink_contribution_mode,
            train_x,
            train_y,
        )


def save_reduction_stage_figures(
    stage_records,
    x,
    y_true,
    y_baseline,
    function,
    p,
    interval,
    n_points,
    output_dir,
    file_format,
    dpi,
    kink_contribution_mode,
    train_x,
    train_y,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for stage in stage_records:
        stage_dir = output_dir / stage["directory_name"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        if stage["stack_label"] == "baseline":
            figures = [
                (
                    "predictions",
                    plot_model_predictions(
                        stage["model"],
                        function=function,
                        p=p,
                        interval=interval,
                        n_points=n_points,
                        train_x=train_x,
                        train_y=train_y,
                    )[0],
                ),
                (
                    "bottleneck_representation",
                    plot_bottleneck_representation(
                        stage["model"],
                        reference_function="scaledSine",
                        p=p,
                        interval=interval,
                        n_points=n_points,
                        train_x=train_x,
                    )[0],
                ),
            ]
        else:
            figures = [
                (
                    "predictions",
                    plot_baseline_candidate_predictions(
                        x,
                        y_true,
                        y_baseline,
                        stage["y_pred"],
                        function,
                        candidate_label=stage["plot_label"],
                        title=stage["title"],
                        train_x=train_x,
                        train_y=train_y,
                    ),
                ),
                (
                    "bottleneck_representation",
                    plot_bottleneck_representation(
                        stage["model"],
                        reference_function="scaledSine",
                        p=p,
                        interval=interval,
                        n_points=n_points,
                        train_x=train_x,
                    )[0],
                ),
            ]
        figures.extend(
            plot_model_kink_figures(
                stage["model"],
                function=function,
                p=p,
                interval=interval,
                n_points=n_points,
                contribution_mode=kink_contribution_mode,
                train_x=train_x,
            )
        )
        if stage["model"].stack1.affine2.out_features >= 2:
            figures.append(
                (
                    "bottleneck_plane",
                    plot_bottleneck_plane(
                        stage["model"],
                        reference_function="scaledSine",
                        p=p,
                        interval=interval,
                        n_points=n_points,
                        train_x=train_x,
                    )[0],
                )
            )

        for name, fig in figures:
            fig.savefig(stage_dir / f"{name}.{file_format}", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        stage_manifest = {
            key: value
            for key, value in stage.items()
            if key not in {"model", "y_pred"}
        }
        with (stage_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(stage_manifest, handle, indent=2)
        manifest.append(stage_manifest)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
