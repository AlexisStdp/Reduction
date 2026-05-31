from pathlib import Path

import torch

from data_generation import CompositeSineDataset, sample_function_grid
from experiment_config import load_config, resolve_checkpoint_dir, resolve_run_dir
from stack import Stack, doubleStack


def load_training_config(path):
    return load_config(path)


def resolve_source_checkpoint(source, training_config):
    checkpoint = source.get("checkpoint", "auto")
    if checkpoint in (None, "auto"):
        return resolve_checkpoint_dir(training_config) / "last.ckpt"
    return Path(checkpoint)


def source_output_dir(source, training_config, reduction_name):
    output_dir = source.get("output_dir")
    if output_dir in (None, "auto"):
        return resolve_run_dir(training_config) / "reductions" / reduction_name
    return Path(output_dir)


def build_model(training_config, function=None, dropout=None):
    dataset_config = training_config["dataset"]
    model_config = training_config["model"]
    function = function or dataset_config["function"]
    dropout = model_config.get("dropout", 0.0) if dropout is None else dropout

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
    return doubleStack(neurons_per_layer, dropout=dropout)


def load_dataset_points(training_config, split="train"):
    dataset_config = training_config["dataset"]
    if split not in {"train", "val"}:
        raise ValueError(f"Unknown dataset split: {split}")

    dataset = CompositeSineDataset(
        n_samples=dataset_config[f"n_{split}"],
        function=dataset_config["function"],
        p=dataset_config["p"],
        interval=tuple(dataset_config["interval"]),
        noise_std=dataset_config["noise_std"],
        seed=dataset_config[f"{split}_seed"],
    )
    return dataset.x.detach().cpu().numpy(), dataset.y.detach().cpu().numpy()


def load_checkpoint(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(normalize_state_dict_keys(state_dict))
    model.eval()
    return model


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key.removeprefix("model.")
        normalized[key] = value
    return normalized


def clone_weights(weights):
    return [weight.detach().cpu().clone() for weight in weights]


def build_stack_from_weights(weights):
    weights = clone_weights(weights)
    input_dim = weights[0].shape[1]
    hidden_dim = weights[0].shape[0]
    output_dim = weights[2].shape[0]
    stack = Stack(input_dim, hidden_dim, output_dim, dropout=0.0)
    stack.set_weights(weights)
    stack.eval()
    return stack


def build_reduced_double_stack(stack1_weights, stack2_weights):
    stack1_weights = clone_weights(stack1_weights)
    stack2_weights = clone_weights(stack2_weights)
    neurons_per_layer = [
        stack1_weights[0].shape[1],
        stack1_weights[0].shape[0],
        stack1_weights[2].shape[0],
        stack2_weights[0].shape[0],
        stack2_weights[2].shape[0],
    ]
    model = doubleStack(neurons_per_layer, dropout=0.0)
    model.stack1.set_weights(stack1_weights)
    model.stack2.set_weights(stack2_weights)
    model.eval()
    return model
