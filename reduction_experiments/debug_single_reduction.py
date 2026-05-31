from pathlib import Path
import argparse

import numpy as np
import torch

from data_generation import sample_function_grid
from reduction import compute_bvv, remove_weak_neurons
from reduction_experiments.metrics import evaluate_model, prediction_metrics
from reduction_experiments.model_io import (
    build_model,
    build_reduced_double_stack,
    build_stack_from_weights,
    load_checkpoint,
    load_training_config,
    resolve_source_checkpoint,
)
from reduction_experiments.pipeline import (
    cluster_weights,
    compression_metrics,
    remove_degenerate_v_neurons,
    remove_outside_neurons_stable,
    remove_weak_neurons_below_threshold,
    validate_array,
    weak_neuron_scores,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug one composite sine reduction variant.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/composite_sine_bottleneck1.yaml"),
        help="Training config to reduce.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to reduce. Defaults to last.ckpt from the training config.",
    )
    parser.add_argument("--clusters", type=int, default=100, help="Clusters to keep in each stack.")
    parser.add_argument("--weak-discard", type=int, default=5, help="Weak neurons to discard in each stack.")
    parser.add_argument("--weak-threshold", type=float, default=1e-12, help="Drop neurons with weak score below this.")
    parser.add_argument("--reduction-points", type=int, default=100, help="Grid points used for outside-neuron removal.")
    parser.add_argument("--evaluation-points", type=int, default=2048, help="Grid points used for final metrics.")
    parser.add_argument("--min-v-norm", type=float, default=1e-12, help="Drop/fold neurons with smaller incoming V norm.")
    parser.add_argument("--sample-weight", choices=["none", "cluster_strength"], default="none")
    return parser.parse_args()


def main():
    args = parse_args()
    training_config = load_training_config(args.config)
    checkpoint_path = args.checkpoint or resolve_source_checkpoint({"checkpoint": "auto"}, training_config)
    dataset_config = training_config["dataset"]
    function = dataset_config["function"]
    p = dataset_config["p"]
    interval = tuple(dataset_config["interval"])

    print(f"source config: {args.config}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"clusters per stack: {args.clusters}")
    print(f"weak neurons discarded per stack: {args.weak_discard}")

    baseline_model = build_model(training_config, function=function)
    load_checkpoint(baseline_model, checkpoint_path)
    print_model_summary("baseline", baseline_model)

    x_reduce, _ = sample_function_grid(
        function=function,
        p=p,
        interval=interval,
        n_points=args.reduction_points,
    )
    stack1_inputs = torch.tensor(x_reduce, dtype=torch.float32)

    stack1_weights = reduce_stack_with_diagnostics(
        label="stack1",
        weights=baseline_model.stack1.get_weights(),
        inputs=stack1_inputs,
        clusters=args.clusters,
        weak_discard=args.weak_discard,
        weak_threshold=args.weak_threshold,
        min_v_norm=args.min_v_norm,
        sample_weight=args.sample_weight,
    )

    reduced_stack1 = build_stack_from_weights(stack1_weights)
    with torch.no_grad():
        stack2_inputs = reduced_stack1(stack1_inputs).detach()
    print_array_stats("stack2 local inputs from reduced stack1", stack2_inputs)
    validate_array("stack2 local inputs", stack2_inputs.detach().cpu().numpy())

    stack2_weights = reduce_stack_with_diagnostics(
        label="stack2",
        weights=baseline_model.stack2.get_weights(),
        inputs=stack2_inputs,
        clusters=args.clusters,
        weak_discard=args.weak_discard,
        weak_threshold=args.weak_threshold,
        min_v_norm=args.min_v_norm,
        sample_weight=args.sample_weight,
    )

    reduced_model = build_reduced_double_stack(stack1_weights, stack2_weights)
    print_model_summary("reduced", reduced_model)

    _, y_true, y_reduced = evaluate_model(
        reduced_model,
        function=function,
        p=p,
        interval=interval,
        n_points=args.evaluation_points,
    )
    validate_array("final reduced predictions", y_reduced)
    print(prediction_metrics(y_true, y_reduced, "reduced"))
    print(compression_metrics(baseline_model, reduced_model))
    print("debug reduction finished")


def reduce_stack_with_diagnostics(label, weights, inputs, clusters, weak_discard, weak_threshold, min_v_norm, sample_weight):
    weights = [weight.detach().cpu().clone() for weight in weights]
    print(f"\n== {label} ==")
    print_weights_stats("initial", weights)
    print_array_stats("local inputs", inputs)

    weights = remove_outside_neurons_stable(weights, inputs)
    print_weights_stats("after stable outside-neuron removal", weights)
    print_bvv_stats("after stable outside-neuron removal", weights, strict=False)

    if min_v_norm > 0:
        weights = remove_degenerate_v_neurons(weights, min_v_norm=min_v_norm)
        print_weights_stats("after degenerate-v removal", weights)
        print_bvv_stats("after degenerate-v removal", weights, strict=True)

    if weak_threshold > 0:
        before = int(weights[0].shape[0])
        print_score_stats("weak scores before threshold", weights)
        weights = remove_weak_neurons_below_threshold(weights, min_score=weak_threshold)
        print(f"weak threshold={weak_threshold}, removed={before - int(weights[0].shape[0])}")
        print_weights_stats("after weak-threshold removal", weights)
        print_score_stats("weak scores after threshold", weights)
        print_bvv_stats("after weak-threshold removal", weights, strict=True)

    if weak_discard > 0:
        effective_discard = min(int(weak_discard), max(int(weights[0].shape[0]) - 1, 0))
        print(f"weak discard requested={weak_discard}, effective={effective_discard}")
        weights = remove_weak_neurons(weights, discard_neurons=effective_discard)
        print_weights_stats("after weak-neuron removal", weights)
        print_bvv_stats("after weak-neuron removal", weights, strict=True)

    effective_clusters = min(int(clusters), int(weights[0].shape[0]))
    print(f"cluster requested={clusters}, effective={effective_clusters}")
    if effective_clusters < int(weights[0].shape[0]):
        weights = cluster_weights(weights, effective_clusters, sample_weight)
    print_weights_stats("after clustering", weights)
    print_bvv_stats("after clustering", weights, strict=True)
    return weights


def print_model_summary(label, model):
    print(
        f"{label}: stack1={model.stack1.affine1.out_features}, "
        f"stack2={model.stack2.affine1.out_features}, "
        f"params={sum(param.numel() for param in model.parameters())}"
    )


def print_weights_stats(label, weights):
    shapes = [tuple(weight.shape) for weight in weights]
    v_norm = weights[0].norm(dim=1)
    print(
        f"{label}: shapes={shapes}, neurons={weights[0].shape[0]}, "
        f"zero_v_norms={int((v_norm == 0).sum())}, "
        f"tiny_v_norms={int((v_norm < 1e-12).sum())}, "
        f"min_v_norm={float(v_norm.min()) if v_norm.numel() else 'nan'}, "
        f"max_v_norm={float(v_norm.max()) if v_norm.numel() else 'nan'}"
    )
    for idx, weight in enumerate(weights):
        print_array_stats(f"  weight[{idx}]", weight)


def print_score_stats(label, weights):
    scores = weak_neuron_scores(weights)
    print_array_stats(label, scores)
    if scores.numel():
        below_default = int((scores < 1e-12).sum())
        print(f"{label}: below_1e-12={below_default}")


def print_bvv_stats(label, weights, strict=True):
    try:
        bvv = compute_bvv(weights)
        print_array_stats(f"bvv {label}", bvv)
        if strict:
            validate_array(f"bvv {label}", bvv.detach().cpu().numpy(), weights=weights)
    except Exception as exc:
        print(f"bvv {label}: ERROR {exc!r}")
        raise


def print_array_stats(label, values):
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)

    finite_mask = np.isfinite(array)
    finite = array[finite_mask]
    stats = {
        "shape": array.shape,
        "nan": int(np.isnan(array).sum()),
        "posinf": int(np.isposinf(array).sum()),
        "neginf": int(np.isneginf(array).sum()),
    }
    if finite.size:
        stats["min"] = float(finite.min())
        stats["max"] = float(finite.max())
        stats["mean"] = float(finite.mean())
    print(f"{label}: {stats}")


if __name__ == "__main__":
    main()
