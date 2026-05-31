from pathlib import Path

from experiment_config import deep_update


DEFAULT_REDUCTION_CONFIG = {
    "name": None,
    "sources": [],
    "evaluation": {
        "n_points": 2048,
        "reduction_grid_points": 100,
        "save_figures": True,
        "save_stage_figures": True,
        "show_training_points": False,
        "training_points_split": "train",
        "figure_format": "png",
        "dpi": 160,
        "kink_contribution_mode": "v_abs_w",
    },
    "reduction": {
        "outside": {
            "enabled": True,
        },
        "weak": {
            "enabled": True,
            "threshold_enabled": True,
            "min_score": 1e-5,
        },
        "clustering": {
            "enabled": True,
            "sample_weight": "none",
            "reconstruction": "kink_center",
        },
        "svd_affine": {
            "enabled": False,
        },
        "numerical": {
            "min_v_norm": 1e-12,
        },
        "stack1": {
            "clusters": [64, 128, 256],
            "weak_discard": [0, 100],
        },
        "stack2": {
            "clusters": [64, 128, 256],
            "weak_discard": [0, 100],
        },
    },
}


def load_reduction_config(path):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML with `pip install pyyaml` to read YAML configs.") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reduction config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}

    config = deep_update(DEFAULT_REDUCTION_CONFIG, user_config)
    if config["name"] in (None, "auto"):
        config["name"] = path.stem
    return config


def write_yaml(data, path):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML with `pip install pyyaml` to write YAML configs.") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def as_list(value):
    if isinstance(value, list):
        return value
    return [value]
