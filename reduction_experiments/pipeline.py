import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from data_generation import sample_function_grid
from reduction import (
    cluster_bvv,
    compute_bvv,
    get_cluster_weights,
    get_clustered_weights,
    remove_weak_neurons,
)
from reduction_experiments.config import as_list, write_yaml
from reduction_experiments.metrics import (
    bottleneck_reference_metrics,
    evaluate_model,
    fidelity_metrics,
    parameter_count,
    prediction_metrics,
    reference_fidelity_metrics,
    relative_delta,
    stack_hidden_counts,
)
from reduction_experiments.model_io import (
    build_model,
    build_reduced_double_stack,
    build_stack_from_weights,
    clone_weights,
    load_checkpoint,
    load_dataset_points,
    load_training_config,
    resolve_source_checkpoint,
    source_output_dir,
)
from reduction_experiments.plotting import plot_reduction_tradeoff, save_variant_figures


def run_reduction_config(config, config_path=None):
    if config["reduction"]["svd_affine"]["enabled"]:
        raise NotImplementedError("Affine SVD reduction is intentionally disabled until svdStack support is added.")

    all_results = []
    for source in config["sources"]:
        all_results.extend(run_source(config, source, config_path=config_path))
    return all_results


def run_source(config, source, config_path=None):
    training_config_path = Path(source["config"])
    training_config = load_training_config(training_config_path)
    dataset_config = training_config["dataset"]
    function = source.get("function") or dataset_config["function"]
    p = dataset_config["p"]
    interval = tuple(dataset_config["interval"])

    output_dir = source_output_dir(source, training_config, config["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(config, output_dir / "sweep_config.yaml")

    checkpoint_path = resolve_source_checkpoint(source, training_config)
    baseline_model = build_model(training_config, function=function)
    load_checkpoint(baseline_model, checkpoint_path)

    evaluation_config = config["evaluation"]
    n_points = int(evaluation_config["n_points"])
    _, y_true, y_baseline = evaluate_model(
        baseline_model,
        function=function,
        p=p,
        interval=interval,
        n_points=n_points,
    )
    baseline_quality = prediction_metrics(y_true, y_baseline, "baseline")
    train_x = None
    train_y = None
    if evaluation_config.get("show_training_points", False):
        train_split = evaluation_config.get("training_points_split", "train")
        train_x, train_y = load_dataset_points(training_config, split=train_split)

    variants = build_variants(config["reduction"])
    summary_rows = []
    for variant in variants:
        variant_dir = output_dir / "variants" / variant_name(variant)
        try:
            metrics = run_variant(
                variant=variant,
                baseline_model=baseline_model,
                baseline_quality=baseline_quality,
                y_true=y_true,
                y_baseline=y_baseline,
                training_config=training_config,
                function=function,
                p=p,
                interval=interval,
                config=config,
                checkpoint_path=checkpoint_path,
                training_config_path=training_config_path,
                variant_dir=variant_dir,
                train_x=train_x,
                train_y=train_y,
            )
            summary_rows.append(summary_row(metrics))
        except Exception as exc:
            variant_dir.mkdir(parents=True, exist_ok=True)
            error_metrics = {
                "status": "error",
                "variant": variant_name(variant),
                "error": repr(exc),
                "source_config": str(training_config_path),
                "checkpoint": str(checkpoint_path),
                "variant_config": variant,
            }
            write_json(error_metrics, variant_dir / "metrics.json")
            summary_rows.append(summary_row(error_metrics))

    write_summary(summary_rows, output_dir / "summary.csv")
    save_tradeoff(summary_rows, output_dir, evaluation_config)
    return summary_rows


def build_variants(reduction_config):
    stack1 = reduction_config["stack1"]
    stack2 = reduction_config["stack2"]
    variants = []
    for stack1_clusters, stack1_weak, stack2_clusters, stack2_weak in itertools.product(
        as_list(stack1["clusters"]),
        as_list(stack1["weak_discard"]),
        as_list(stack2["clusters"]),
        as_list(stack2["weak_discard"]),
    ):
        variants.append(
            {
                "stack1_clusters": int(stack1_clusters),
                "stack1_weak_discard": int(stack1_weak),
                "stack2_clusters": int(stack2_clusters),
                "stack2_weak_discard": int(stack2_weak),
            }
        )
    return variants


def variant_name(variant):
    return (
        f"s1_k{variant['stack1_clusters']:04d}_w{variant['stack1_weak_discard']:04d}_"
        f"s2_k{variant['stack2_clusters']:04d}_w{variant['stack2_weak_discard']:04d}"
    )


def run_variant(
    variant,
    baseline_model,
    baseline_quality,
    y_true,
    y_baseline,
    training_config,
    function,
    p,
    interval,
    config,
    checkpoint_path,
    training_config_path,
    variant_dir,
    train_x=None,
    train_y=None,
):
    variant_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(variant, variant_dir / "reduction_config.yaml")

    evaluation_config = config["evaluation"]
    reduction_config = config["reduction"]
    reduction_grid_points = int(evaluation_config["reduction_grid_points"])
    x_reduce, _ = sample_function_grid(
        function=function,
        p=p,
        interval=interval,
        n_points=reduction_grid_points,
    )
    stack1_inputs = torch.tensor(x_reduce, dtype=torch.float32)
    baseline_stack1_weights = clone_weights(baseline_model.stack1.get_weights())
    baseline_stack2_weights = clone_weights(baseline_model.stack2.get_weights())

    stack1_weights, stack1_diag, stack1_stages = reduce_stack(
        weights=baseline_stack1_weights,
        inputs=stack1_inputs,
        requested_clusters=variant["stack1_clusters"],
        weak_discard=variant["stack1_weak_discard"],
        reduction_config=reduction_config,
        stack_label="stack1",
    )
    reduced_stack1 = build_stack_from_weights(stack1_weights)
    with torch.no_grad():
        stack2_inputs = reduced_stack1(stack1_inputs).detach()

    stack2_weights, stack2_diag, stack2_stages = reduce_stack(
        weights=baseline_stack2_weights,
        inputs=stack2_inputs,
        requested_clusters=variant["stack2_clusters"],
        weak_discard=variant["stack2_weak_discard"],
        reduction_config=reduction_config,
        stack_label="stack2",
    )
    reduced_model = build_reduced_double_stack(stack1_weights, stack2_weights)

    n_points = int(evaluation_config["n_points"])
    x, _, y_reduced = evaluate_model(
        reduced_model,
        function=function,
        p=p,
        interval=interval,
        n_points=n_points,
    )
    validate_array("reduced predictions", y_reduced)
    stage_records = build_reduction_stage_records(
        baseline_model=baseline_model,
        baseline_stack2_weights=baseline_stack2_weights,
        stack1_stages=stack1_stages,
        stack2_stages=stack2_stages,
        y_true=y_true,
        y_baseline=y_baseline,
        function=function,
        p=p,
        interval=interval,
        n_points=n_points,
    )

    reduced_quality = prediction_metrics(y_true, y_reduced, "reduced")
    metrics = {
        "status": "ok",
        "variant": variant_name(variant),
        "source_experiment": training_config["experiment"]["name"],
        "source_config": str(training_config_path),
        "checkpoint": str(checkpoint_path),
        "variant_config": variant,
        **baseline_quality,
        **reduced_quality,
        **fidelity_metrics(y_baseline, y_reduced),
        **compression_metrics(baseline_model, reduced_model),
        **bottleneck_reference_metrics(reduced_model, "scaledSine", p, interval, n_points),
        "stack1_diagnostics": stack1_diag,
        "stack2_diagnostics": stack2_diag,
        "reduction_stage_metrics": [serialize_stage_record(record) for record in stage_records],
    }
    metrics["mse_delta"] = float(metrics["reduced_mse"] - metrics["baseline_mse"])
    metrics["relative_mse_delta"] = relative_delta(metrics["reduced_mse"], metrics["baseline_mse"])

    torch.save(
        {
            "state_dict": reduced_model.state_dict(),
            "variant": variant,
            "metrics": metrics,
        },
        variant_dir / "reduced_checkpoint.pt",
    )
    write_json(metrics, variant_dir / "metrics.json")

    if evaluation_config["save_figures"]:
        save_variant_figures(
            reduced_model,
            x,
            y_true,
            y_baseline,
            y_reduced,
            function,
            p,
            interval,
            n_points,
            variant_dir / "figures",
            evaluation_config["figure_format"],
            int(evaluation_config["dpi"]),
            stage_records=stage_records if evaluation_config.get("save_stage_figures", True) else None,
            kink_contribution_mode=evaluation_config.get("kink_contribution_mode", "v_abs_w"),
            train_x=train_x,
            train_y=train_y,
        )

    return metrics


def reduce_stack(weights, inputs, requested_clusters, weak_discard, reduction_config, stack_label):
    weights = [weight.detach().cpu().clone() for weight in weights]
    diagnostics = {
        "initial_neurons": int(weights[0].shape[0]),
        "outside_enabled": bool(reduction_config["outside"]["enabled"]),
        "weak_enabled": bool(reduction_config["weak"]["enabled"]),
        "clustering_enabled": bool(reduction_config["clustering"]["enabled"]),
        "clustering_reconstruction": reduction_config["clustering"].get("reconstruction", "kink_center"),
    }
    stages = []
    record_stack_stage(
        stages,
        stack_label=stack_label,
        step_key="initial",
        step_title="initial state",
        weights=weights,
        diagnostics=diagnostics,
        applied=True,
        changed=False,
        note="Original stack weights before reduction.",
        inputs=inputs,
    )

    if reduction_config["outside"]["enabled"]:
        before_weights = clone_weights(weights)
        before = int(weights[0].shape[0])
        weights = remove_outside_neurons_stable(weights, inputs)
        after = int(weights[0].shape[0])
        diagnostics["outside_removed"] = before - after
        diagnostics["after_outside_neurons"] = after
        outside_changed = weights_changed(before_weights, weights)
        outside_note = f"Removed {before - after} always-on/off neurons."
    else:
        diagnostics["outside_removed"] = 0
        diagnostics["after_outside_neurons"] = int(weights[0].shape[0])
        outside_changed = False
        outside_note = "Outside-neuron removal disabled."
    record_stack_stage(
        stages,
        stack_label=stack_label,
        step_key="outside",
        step_title="outside-neuron removal",
        weights=weights,
        diagnostics=diagnostics,
        applied=bool(reduction_config["outside"]["enabled"]),
        changed=outside_changed,
        note=outside_note,
        inputs=inputs,
    )

    min_v_norm = float(reduction_config.get("numerical", {}).get("min_v_norm", 0.0))
    if min_v_norm > 0:
        before_weights = clone_weights(weights)
        before = int(weights[0].shape[0])
        weights = remove_degenerate_v_neurons(weights, min_v_norm=min_v_norm)
        after = int(weights[0].shape[0])
        diagnostics["degenerate_v_min_norm"] = min_v_norm
        diagnostics["degenerate_v_removed"] = before - after
        diagnostics["after_degenerate_v_neurons"] = after
        degenerate_changed = weights_changed(before_weights, weights)
        degenerate_note = f"Removed/folded {before - after} neurons with ||v|| < {min_v_norm}."
    else:
        diagnostics["degenerate_v_min_norm"] = 0.0
        diagnostics["degenerate_v_removed"] = 0
        diagnostics["after_degenerate_v_neurons"] = int(weights[0].shape[0])
        degenerate_changed = False
        degenerate_note = "Degenerate-v removal disabled."
    record_stack_stage(
        stages,
        stack_label=stack_label,
        step_key="degenerate_v",
        step_title="degenerate-v removal",
        weights=weights,
        diagnostics=diagnostics,
        applied=min_v_norm > 0,
        changed=degenerate_changed,
        note=degenerate_note,
        inputs=inputs,
    )

    weak_config = reduction_config["weak"]
    if weak_config["enabled"] and weak_config.get("threshold_enabled", True):
        before_weights = clone_weights(weights)
        before = int(weights[0].shape[0])
        min_score = float(weak_config.get("min_score", 0.0))
        weights = remove_weak_neurons_below_threshold(weights, min_score=min_score)
        after = int(weights[0].shape[0])
        diagnostics["weak_threshold_enabled"] = True
        diagnostics["weak_min_score"] = min_score
        diagnostics["weak_threshold_removed"] = before - after
        diagnostics["after_weak_threshold_neurons"] = after
        threshold_changed = weights_changed(before_weights, weights)
        threshold_note = f"Removed {before - after} neurons below weak-score threshold {min_score}."
    else:
        diagnostics["weak_threshold_enabled"] = False
        diagnostics["weak_min_score"] = ""
        diagnostics["weak_threshold_removed"] = 0
        diagnostics["after_weak_threshold_neurons"] = int(weights[0].shape[0])
        threshold_changed = False
        threshold_note = "Weak-threshold pruning disabled."
    record_stack_stage(
        stages,
        stack_label=stack_label,
        step_key="weak_threshold",
        step_title="weak-threshold pruning",
        weights=weights,
        diagnostics=diagnostics,
        applied=bool(weak_config["enabled"] and weak_config.get("threshold_enabled", True)),
        changed=threshold_changed,
        note=threshold_note,
        inputs=inputs,
    )

    if weak_config["enabled"] and weak_discard > 0:
        before_weights = clone_weights(weights)
        before = int(weights[0].shape[0])
        effective_discard = min(int(weak_discard), max(before - 1, 0))
        weights = remove_weak_neurons(weights, discard_neurons=effective_discard)
        after = int(weights[0].shape[0])
        diagnostics["weak_requested_discard"] = int(weak_discard)
        diagnostics["weak_effective_discard"] = effective_discard
        diagnostics["after_weak_neurons"] = after
        weak_changed = weights_changed(before_weights, weights)
        weak_note = f"Discarded {effective_discard} weakest remaining neurons."
    else:
        diagnostics["weak_requested_discard"] = int(weak_discard)
        diagnostics["weak_effective_discard"] = 0
        diagnostics["after_weak_neurons"] = int(weights[0].shape[0])
        weak_changed = False
        weak_note = "Weak-count pruning skipped because weak_discard <= 0 or weak pruning disabled."
    record_stack_stage(
        stages,
        stack_label=stack_label,
        step_key="weak_discard",
        step_title="weak-count pruning",
        weights=weights,
        diagnostics=diagnostics,
        applied=bool(weak_config["enabled"] and weak_discard > 0),
        changed=weak_changed,
        note=weak_note,
        inputs=inputs,
    )

    if reduction_config["clustering"]["enabled"]:
        before_weights = clone_weights(weights)
        before = int(weights[0].shape[0])
        effective_clusters = min(int(requested_clusters), before)
        if effective_clusters < before:
            weights = cluster_weights(
                weights,
                effective_clusters,
                reduction_config["clustering"]["sample_weight"],
                reduction_config["clustering"].get("reconstruction", "kink_center"),
            )
        diagnostics["cluster_requested"] = int(requested_clusters)
        diagnostics["cluster_effective"] = effective_clusters
        diagnostics["after_cluster_neurons"] = int(weights[0].shape[0])
        cluster_changed = weights_changed(before_weights, weights)
        cluster_note = f"Requested {requested_clusters} clusters; used {effective_clusters}."
    else:
        diagnostics["cluster_requested"] = int(requested_clusters)
        diagnostics["cluster_effective"] = int(weights[0].shape[0])
        diagnostics["after_cluster_neurons"] = int(weights[0].shape[0])
        cluster_changed = False
        cluster_note = "Clustering disabled."
    record_stack_stage(
        stages,
        stack_label=stack_label,
        step_key="clustering",
        step_title="clustering",
        weights=weights,
        diagnostics=diagnostics,
        applied=bool(reduction_config["clustering"]["enabled"]),
        changed=cluster_changed,
        note=cluster_note,
        inputs=inputs,
    )

    diagnostics["final_neurons"] = int(weights[0].shape[0])
    return weights, diagnostics, stages


def remove_outside_neurons_stable(weights, inputs):
    v, b, w, c, w_affine = [weight.detach().cpu().clone() for weight in weights]
    inputs = inputs.detach().cpu()
    preactivation = inputs @ v.T + b
    always_on = (preactivation > 0).all(dim=0)
    always_off = (preactivation <= 0).all(dim=0)
    outside = always_on | always_off
    keep = ~outside

    if always_on.any():
        c = c + w[:, always_on] @ b[always_on]
        w_affine = w_affine + w[:, always_on] @ v[always_on]

    return [v[keep], b[keep], w[:, keep], c, w_affine]


def remove_degenerate_v_neurons(weights, min_v_norm=1e-12):
    v, b, w, c, w_affine = [weight.detach().cpu().clone() for weight in weights]
    v_norm = v.norm(dim=1)
    degenerate = v_norm < min_v_norm
    keep = ~degenerate

    active_degenerate = degenerate & (b > 0)
    if active_degenerate.any():
        c = c + w[:, active_degenerate] @ b[active_degenerate]
        w_affine = w_affine + w[:, active_degenerate] @ v[active_degenerate]

    return [v[keep], b[keep], w[:, keep], c, w_affine]


def weak_neuron_scores(weights):
    v, b, w, *_ = weights
    return torch.sqrt(v.norm(p=1, dim=1) ** 2 + b**2) * w.norm(p=1, dim=0)


def remove_weak_neurons_below_threshold(weights, min_score=1e-12):
    v, b, w, c, w_affine = [weight.detach().cpu().clone() for weight in weights]
    scores = weak_neuron_scores([v, b, w, c, w_affine])
    keep = scores >= min_score

    if not keep.any() and scores.numel() > 0:
        keep[scores.argmax()] = True

    return [v[keep], b[keep], w[:, keep], c, w_affine]


def cluster_weights(weights, n_clusters, sample_weight_mode, reconstruction_mode="kink_center"):
    bvv = compute_bvv(weights).detach().cpu().numpy()
    validate_array("bvv clustering features", bvv, weights=weights)
    cluster_weights_value = None
    if sample_weight_mode == "cluster_strength":
        cluster_weights_value = get_cluster_weights(weights).detach().cpu().numpy()
        validate_array("cluster sample weights", cluster_weights_value)
    elif sample_weight_mode not in ("none", None):
        raise ValueError(f"Unknown clustering.sample_weight value: {sample_weight_mode}")
    cluster = cluster_bvv(n_clusters, bvv, cluster_weights=cluster_weights_value, verbose=False)
    if reconstruction_mode in {"kink_center", "kink_center_1d"}:
        return get_clustered_weights_from_centers(weights, cluster)
    if reconstruction_mode != "legacy":
        raise ValueError(f"Unknown clustering.reconstruction value: {reconstruction_mode}")
    return get_clustered_weights(weights, cluster)


def get_clustered_weights_from_centers(weights, cluster):
    v, b, w, c, w_affine = [weight.detach().cpu().clone() for weight in weights]
    input_dim = v.shape[1]
    centers = torch.tensor(cluster.cluster_centers_, dtype=v.dtype)
    clustered_v = []
    clustered_b = []
    clustered_w = []

    for cluster_index in range(centers.shape[0]):
        members = torch.tensor(cluster.labels_ == cluster_index, dtype=torch.bool)
        if not members.any():
            continue

        center = centers[cluster_index]
        center_bvv = center[:input_dim]
        direction_hint = center[input_dim:]
        direction = normalized_direction(direction_hint, v[members])
        beta = torch.dot(center_bvv, direction)

        member_v = v[members]
        member_w = w[:, members]
        total_jump = member_w @ member_v
        w_new = total_jump @ direction

        clustered_v.append(direction)
        clustered_b.append(beta)
        clustered_w.append(w_new)

    if not clustered_v:
        raise ValueError("KMeans returned no non-empty clusters.")

    return [
        torch.stack(clustered_v, dim=0),
        torch.stack(clustered_b, dim=0),
        torch.stack(clustered_w, dim=1),
        c,
        w_affine,
    ]


def normalized_direction(direction_hint, member_v, eps=1e-8):
    direction = direction_hint.detach().cpu().clone()
    if direction.norm() < eps:
        direction = member_v.mean(dim=0)
    if direction.norm() < eps:
        direction = member_v[0]
    norm = direction.norm()
    if norm < eps:
        raise ValueError("Cannot reconstruct clustered neuron from a zero direction.")
    return direction / norm


def validate_array(name, values, weights=None):
    values = np.asarray(values)
    if np.isfinite(values).all():
        return

    finite = values[np.isfinite(values)]
    message = (
        f"{name} contains non-finite values: "
        f"shape={values.shape}, nan={int(np.isnan(values).sum())}, "
        f"posinf={int(np.isposinf(values).sum())}, neginf={int(np.isneginf(values).sum())}"
    )
    if finite.size:
        message += f", finite_min={float(finite.min())}, finite_max={float(finite.max())}"
    if weights is not None:
        v = weights[0].detach().cpu()
        v_norm = v.norm(dim=1)
        message += (
            f", zero_v_norms={int((v_norm == 0).sum())}, "
            f"tiny_v_norms={int((v_norm < 1e-12).sum())}, "
            f"min_v_norm={float(v_norm.min()) if v_norm.numel() else 'nan'}"
        )
    raise ValueError(message)


def compression_metrics(baseline_model, reduced_model):
    baseline_params = parameter_count(baseline_model)
    reduced_params = parameter_count(reduced_model)
    baseline_counts = stack_hidden_counts(baseline_model)
    reduced_counts = stack_hidden_counts(reduced_model)
    return {
        "baseline_params": baseline_params,
        "reduced_params": reduced_params,
        "params_removed": baseline_params - reduced_params,
        "compression_ratio": float(baseline_params / reduced_params) if reduced_params else float("inf"),
        "baseline_stack1_neurons": baseline_counts["stack1_neurons"],
        "baseline_stack2_neurons": baseline_counts["stack2_neurons"],
        "baseline_total_stack_neurons": baseline_counts["total_stack_neurons"],
        "reduced_stack1_neurons": reduced_counts["stack1_neurons"],
        "reduced_stack2_neurons": reduced_counts["stack2_neurons"],
        "reduced_total_stack_neurons": reduced_counts["total_stack_neurons"],
    }


def summary_row(metrics):
    keys = [
        "status",
        "variant",
        "source_experiment",
        "baseline_mse",
        "reduced_mse",
        "mse_delta",
        "relative_mse_delta",
        "baseline_vs_reduced_mse",
        "baseline_params",
        "reduced_params",
        "compression_ratio",
        "reduced_stack1_neurons",
        "reduced_stack2_neurons",
        "bottleneck_best_abs_corr_with_scaled_sine",
        "error",
    ]
    return {key: metrics.get(key, "") for key in keys}


def record_stack_stage(stages, stack_label, step_key, step_title, weights, diagnostics, applied, changed, note, inputs=None):
    stages.append(
        {
            "stack_label": stack_label,
            "step_key": step_key,
            "step_title": step_title,
            "weights": clone_weights(weights),
            "neurons": int(weights[0].shape[0]),
            "applied": bool(applied),
            "changed": bool(changed),
            "note": note,
            "diagnostics": dict(diagnostics),
            "geometry": stage_geometry_summary(weights, inputs),
        }
    )


def weights_changed(before_weights, after_weights):
    if len(before_weights) != len(after_weights):
        return True
    for before, after in zip(before_weights, after_weights):
        if before.shape != after.shape or not torch.equal(before, after):
            return True
    return False


def build_reduction_stage_records(
    baseline_model,
    baseline_stack2_weights,
    stack1_stages,
    stack2_stages,
    y_true,
    y_baseline,
    function,
    p,
    interval,
    n_points,
):
    records = [
        create_stage_record(
            order=0,
            directory_name="00_baseline",
            title="Baseline full model",
            plot_label="baseline",
            model=baseline_model,
            stage_info={
                "stack_label": "baseline",
                "step_key": "baseline",
                "step_title": "baseline full model",
                "applied": True,
                "changed": False,
                "note": "Unreduced source model.",
                "diagnostics": {},
            },
        )
    ]

    order = 1
    for stage in stack1_stages[1:]:
        records.append(
            create_stage_record(
                order=order,
                directory_name=f"{order:02d}_stack1_after_{stage['step_key']}",
                title=f"Stack 1 after {stage['step_title']}",
                plot_label=f"stack1:{stage['step_key']}",
                model=build_reduced_double_stack(stage["weights"], baseline_stack2_weights),
                stage_info=stage,
            )
        )
        order += 1

    final_stack1_weights = clone_weights(stack1_stages[-1]["weights"])
    for stage in stack2_stages[1:]:
        records.append(
            create_stage_record(
                order=order,
                directory_name=f"{order:02d}_stack2_after_{stage['step_key']}",
                title=f"Stack 2 after {stage['step_title']}",
                plot_label=f"stack2:{stage['step_key']}",
                model=build_reduced_double_stack(final_stack1_weights, stage["weights"]),
                stage_info=stage,
            )
        )
        order += 1

    baseline_mse = float(np.mean((np.asarray(y_baseline) - np.asarray(y_true)) ** 2))
    baseline_params = parameter_count(baseline_model)
    for record in records:
        _, _, y_stage = evaluate_model(
            record["model"],
            function=function,
            p=p,
            interval=interval,
            n_points=n_points,
        )
        record["y_pred"] = y_stage
        stage_params = parameter_count(record["model"])
        stage_counts = stack_hidden_counts(record["model"])
        stage_metrics = {
            **prediction_metrics(y_true, y_stage, "stage"),
            **reference_fidelity_metrics(y_baseline, y_stage, "baseline_vs_stage"),
            "stage_params": stage_params,
            "params_removed_vs_baseline": baseline_params - stage_params,
            "compression_ratio_vs_baseline": float(baseline_params / stage_params) if stage_params else float("inf"),
            "stage_stack1_neurons": stage_counts["stack1_neurons"],
            "stage_stack2_neurons": stage_counts["stack2_neurons"],
            "stage_total_stack_neurons": stage_counts["total_stack_neurons"],
        }
        stage_metrics["relative_mse_delta_vs_baseline"] = relative_delta(stage_metrics["stage_mse"], baseline_mse)
        record["metrics"] = stage_metrics

    return records


def create_stage_record(order, directory_name, title, plot_label, model, stage_info):
    return {
        "order": int(order),
        "directory_name": directory_name,
        "title": title,
        "plot_label": plot_label,
        "stack_label": stage_info["stack_label"],
        "step_key": stage_info["step_key"],
        "step_title": stage_info["step_title"],
        "applied": bool(stage_info["applied"]),
        "changed": bool(stage_info["changed"]),
        "note": stage_info["note"],
        "diagnostics": stage_info["diagnostics"],
        "geometry": stage_info.get("geometry"),
        "model": model,
    }


def serialize_stage_record(record):
    return {
        key: value
        for key, value in record.items()
        if key not in {"model", "y_pred"}
    }


def stage_geometry_summary(weights, inputs):
    summary = {
        "neurons": int(weights[0].shape[0]),
        "input_dim": int(weights[0].shape[1]),
    }
    if inputs is not None:
        inputs_cpu = inputs.detach().cpu()
        summary["observed_input_min"] = inputs_cpu.amin(dim=0).tolist()
        summary["observed_input_max"] = inputs_cpu.amax(dim=0).tolist()

    if weights[0].shape[1] != 1 or weights[0].shape[0] == 0:
        summary["kink_summary_available"] = False
        return summary

    v = weights[0].detach().cpu()[:, 0]
    b = weights[1].detach().cpu()
    finite = torch.isfinite(v) & torch.isfinite(b) & (v.abs() >= 1e-12)
    summary["kink_summary_available"] = True
    summary["finite_kink_neurons"] = int(finite.sum())
    summary["skipped_kink_neurons"] = int((~finite).sum())
    if not finite.any():
        return summary

    kinks = (-b[finite] / v[finite]).detach().cpu()
    summary["kink_min"] = float(kinks.min())
    summary["kink_max"] = float(kinks.max())
    summary["kink_mean"] = float(kinks.mean())
    if inputs is not None:
        x_min = float(inputs_cpu[:, 0].min())
        x_max = float(inputs_cpu[:, 0].max())
        outside = (kinks < x_min) | (kinks > x_max)
        summary["kinks_outside_observed_input"] = int(outside.sum())
    return summary


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_summary(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_tradeoff(summary_rows, output_dir, evaluation_config):
    if not evaluation_config["save_figures"]:
        return
    fig = plot_reduction_tradeoff(summary_rows)
    if fig is None:
        return
    import matplotlib.pyplot as plt

    fig.savefig(
        Path(output_dir) / f"tradeoff.{evaluation_config['figure_format']}",
        dpi=int(evaluation_config["dpi"]),
        bbox_inches="tight",
    )
    plt.close(fig)
