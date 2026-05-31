from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run reduction sweeps for trained composite sine models.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/reductions/composite_sine_reduction_sweep.yaml"),
        help="Path to a reduction sweep YAML config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from reduction_experiments.config import load_reduction_config

    config = load_reduction_config(args.config)
    if config["evaluation"]["save_figures"]:
        import matplotlib

        matplotlib.use("Agg")

    from reduction_experiments.pipeline import run_reduction_config

    run_reduction_config(config, config_path=args.config)
    print("Reduction sweep finished.")


if __name__ == "__main__":
    main()
