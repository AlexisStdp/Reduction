from pathlib import Path
import copy
import shutil


DEFAULT_CONFIG = {
    "seed": 42,
    "experiment": {
        "name": None,
        "run_dir": None,
    },
    "dataset": {
        "n_train": 256,
        "n_val": 256,
        "function": "compositeSinePlus",
        "p": [0.4, 1.0, 2.0, -3.0],
        "interval": [-1.0, 1.0],
        "noise_std": 0.05,
        "train_seed": 42,
        "val_seed": 43,
    },
    "dataloader": {
        "batch_size": None,
        "num_workers": "auto",
        "pin_memory": "auto",
        "persistent_workers": True,
        "prefetch_factor": 2,
    },
    "model": {
        "input_dim": 1,
        "hidden_dim_1": 1024,
        "bottleneck_dim": 1,
        "hidden_dim_2": 1024,
        "dropout": 0.1,
    },
    "optimizer": {
        "lr": 1e-3,
        "weight_decay": 1e-3,
    },
    "trainer": {
        "max_epochs": 2000,
        "check_val_every_n_epoch": 500,
        "accelerator": "auto",
        "devices": "auto",
        "enable_checkpointing": True,
        "log_every_n_steps": 1,
    },
    "checkpoint": {
        "dirpath": "auto",
        "filename": "epoch={epoch:04d}-val_loss={val_loss:.6f}",
        "monitor": "val_loss",
        "mode": "min",
        "save_top_k": 1,
        "save_last": True,
        "every_n_epochs": 500,
    },
    "visualization": {
        "output_dir": "auto",
    },
}


def deep_update(base, overrides):
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML with `pip install pyyaml` to read YAML configs.") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    config = deep_update(DEFAULT_CONFIG, user_config)
    if config["experiment"]["name"] is None:
        config["experiment"]["name"] = path.stem
    return config


def resolve_run_dir(config):
    run_dir = config["experiment"].get("run_dir")
    if run_dir in (None, "auto"):
        run_dir = Path("runs") / config["experiment"]["name"]
    return Path(run_dir)


def resolve_checkpoint_dir(config):
    dirpath = config["checkpoint"].get("dirpath")
    if dirpath in (None, "auto"):
        return resolve_run_dir(config) / "checkpoints"
    return Path(dirpath)


def resolve_figure_dir(config):
    output_dir = config["visualization"].get("output_dir")
    if output_dir in (None, "auto"):
        return resolve_run_dir(config) / "figures"
    return Path(output_dir)


def copy_config_to_run(config_path, run_dir):
    config_path = Path(config_path)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
