from pathlib import Path
import argparse

import torch

from data_generation import sample_function_grid
from experiment_config import load_config, resolve_checkpoint_dir, resolve_figure_dir
from stack import doubleStack


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained composite sine model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/composite_sine_plus_bottleneck1.yaml"),
        help="Path to the training config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to <checkpoint.dirpath>/last.ckpt from the config.",
    )
    parser.add_argument("--function", default=None, help="Function name to visualize. Defaults to the config value.")
    parser.add_argument("--n-points", type=int, default=400, help="Number of grid points used in each plot.")
    parser.add_argument("--save", action="store_true", help="Save figures to --output-dir.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory. Defaults to the run's figures directory.",
    )
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Saved figure format.")
    parser.add_argument("--dpi", type=int, default=160, help="DPI used for raster figure output.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures interactively.")
    return parser.parse_args()


def build_model(config, function):
    dataset_config = config["dataset"]
    model_config = config["model"]
    _, y = sample_function_grid(
        function=function,
        p=dataset_config["p"],
        interval=tuple(dataset_config["interval"]),
        n_points=8,
    )
    neurons_per_layer = [
        model_config["input_dim"],
        model_config["hidden_dim_1"],
        model_config["bottleneck_dim"],
        model_config["hidden_dim_2"],
        y.shape[1],
    ]
    return doubleStack(neurons_per_layer, dropout=model_config.get("dropout", 0.0))


def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = normalize_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key.removeprefix("model.")
        normalized[key] = value
    return normalized


def save_figures(figures, output_dir, file_format, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        path = output_dir / f"{name}.{file_format}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {path}")


def main():
    args = parse_args()
    show = not args.no_show

    if not show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from visualization import (
        plot_bottleneck_representation,
        plot_bottleneck_plane,
        plot_model_predictions,
        plot_residuals,
        plot_target_components,
    )

    config = load_config(args.config)
    dataset_config = config["dataset"]
    function = args.function or dataset_config["function"]
    p = dataset_config["p"]
    interval = tuple(dataset_config["interval"])
    checkpoint_path = args.checkpoint or (resolve_checkpoint_dir(config) / "last.ckpt")
    output_dir = args.output_dir or resolve_figure_dir(config)

    model = build_model(config, function)
    load_checkpoint(model, checkpoint_path)

    figures = [
        (
            "target_components",
            plot_target_components(function=function, p=p, interval=interval, n_points=args.n_points)[0],
        ),
        (
            "model_predictions",
            plot_model_predictions(model, function=function, p=p, interval=interval, n_points=args.n_points)[0],
        ),
        (
            "residuals",
            plot_residuals(model, function=function, p=p, interval=interval, n_points=args.n_points)[0],
        ),
        (
            "bottleneck_representation",
            plot_bottleneck_representation(
                model,
                reference_function="scaledSine",
                p=p,
                interval=interval,
                n_points=args.n_points,
            )[0],
        ),
    ]
    if config["model"]["bottleneck_dim"] >= 2:
        figures.append(
            (
                "bottleneck_plane",
                plot_bottleneck_plane(
                    model,
                    reference_function="scaledSine",
                    p=p,
                    interval=interval,
                    n_points=args.n_points,
                )[0],
            )
        )

    if args.save:
        save_figures(figures, output_dir, args.format, args.dpi)

    if show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
