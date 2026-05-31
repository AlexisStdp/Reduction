# Reduction Experiments

This project treats training runs as immutable inputs and writes reduction outputs under each training run.

```text
runs/{experiment.name}/
  checkpoints/
    last.ckpt
  reductions/
    {reduction.name}/
      sweep_config.yaml
      summary.csv
      tradeoff.png
      variants/
        {variant}/
          reduction_config.yaml
          metrics.json
          reduced_checkpoint.pt
          figures/
            stages/
              manifest.json
              00_baseline/
              01_stack1_after_outside/
              ...
```

The reduction code lives in `reduction.py`. The experiment orchestration around it lives in `reduction_experiments/`.

If `name:` is omitted in a reduction YAML file, the runner now uses the config filename stem for the output folder name.

## Run The Default Sweep

The default sweep is:

```text
configs/reductions/composite_sine_reduction_sweep.yaml
```

It runs reductions for the three current composite sine training configs and uses `last.ckpt` for each source model.

Activate the project virtual environment first, or call its interpreter directly:

```powershell
python -m reduction_experiments.run --config configs/reductions/composite_sine_reduction_sweep.yaml
```

Equivalent shorter form:

```powershell
python -m reduction_experiments --config configs/reductions/composite_sine_reduction_sweep.yaml
```

Without an activated environment, use:

```powershell
.venv\Scripts\python.exe -m reduction_experiments --config configs/reductions/composite_sine_reduction_sweep.yaml
```

If you see `OSError: [WinError 1114]` while importing `torch`, Python is usually loading PyTorch from a different/global interpreter. In that case, use the `.venv\Scripts\python.exe` command above.

For each source, `checkpoint: auto` means:

```text
runs/{experiment.name}/checkpoints/last.ckpt
```

## Debug One Reduction

For a focused debugging pass, use the diagnostic runner:

```powershell
python -m reduction_experiments.debug_single_reduction --config configs/experiments/composite_sine_bottleneck1.yaml --weak-discard 5 --clusters 100
```

Or, without an activated environment:

```powershell
.venv\Scripts\python.exe -m reduction_experiments.debug_single_reduction --config configs/experiments/composite_sine_bottleneck1.yaml --weak-discard 5 --clusters 100
```

This prints tensor shapes, zero/tiny incoming-weight norms, BVV finiteness, and final metrics. It is meant to expose exactly where NaN or Inf values first appear.

The experiment runner uses a stable outside-neuron wrapper rather than calling `remove_outside_neurons()` directly. The wrapper evaluates `Vx + b` on the local input grid, folds always-on neurons into the affine/bias terms, drops always-off neurons, and then removes numerically degenerate neurons with `||V_i|| < reduction.numerical.min_v_norm`. This avoids BVV blow-ups from dividing by nearly-zero incoming weights while leaving `reduction.py` unchanged.

There is also a one-variant config for the normal sweep runner:

```powershell
python -m reduction_experiments.run --config configs/reductions/debug_single_reduction.yaml
```

## Configure A Sweep

Each source points to a trained experiment config:

```yaml
sources:
  - config: configs/experiments/composite_sine_plus_bottleneck1.yaml
    checkpoint: auto
    output_dir: auto
```

Reduction choices are swept by Cartesian product:

```yaml
reduction:
  weak:
    enabled: true
    threshold_enabled: true
    min_score: 1.0e-12
  clustering:
    enabled: true
    sample_weight: none
    reconstruction: kink_center_1d
  stack1:
    clusters: [32, 64, 128]
    weak_discard: [0, 100]
  stack2:
    clusters: [64, 128, 256]
    weak_discard: [0, 100]
```

For example, `stack1.clusters: [32, 64]` and `stack2.clusters: [64, 128]` produces four clustering combinations, before weak-neuron choices are counted.

Weak pruning has two parts. First, when `weak.threshold_enabled` is true, every neuron with weak score below `weak.min_score` is removed. The score matches the criterion used by `remove_weak_neurons()`:

```text
sqrt(||v_i||_1^2 + b_i^2) * ||w_i||_1
```

Second, each stack's `weak_discard` value optionally removes the N weakest remaining neurons. The threshold pass is enabled by default and always keeps at least one neuron.

`clustering.reconstruction: kink_center` rebuilds each clustered neuron from the KMeans center rather than the legacy `get_kink()` aggregate. For 1D inputs this directly fixes the new kink position. For 2D bottlenecks it fixes the new ReLU boundary by taking the cluster center's `bvv` coordinates and direction as the new hyperplane geometry. The older value `kink_center_1d` is still accepted as an alias for backward compatibility.

Affine SVD is intentionally disabled for now:

```yaml
svd_affine:
  enabled: false
```

The current `svd_affine()` output is meant for `svdStack`, while the active composite sine models use `Stack`.

## Reduction Steps

Each candidate variant does:

1. Load the trained `doubleStack` from `last.ckpt`.
2. Evaluate the unreduced baseline on a dense grid.
3. Reduce `stack1` on a 100-point input grid.
4. Recompute the bottleneck values from the reduced `stack1`.
5. Reduce `stack2` on those bottleneck values.
6. Build a reduced `doubleStack` with the new hidden widths.
7. Evaluate, save metrics, save the reduced checkpoint, and optionally save figures.

The default 100-point reduction grid keeps the first sweeps small and comparable to the original notebook workflow. The experiment runner itself does not rely on a fixed sample count.

## Outputs

Each variant directory contains:

```text
reduction_config.yaml
metrics.json
reduced_checkpoint.pt
figures/
```

`metrics.json` is the complete record for one candidate.

`summary.csv` is the compact table used to compare candidates in one sweep. Start here when selecting a reduced model.

Reduced variant figures now also include geometry views for the hidden stacks:

- `stack1_kink_contributions.png`
- `stack2_kink_contributions.png` when the bottleneck is 1D
- `stack2_kink_boundaries.png` when the bottleneck is 2D

Each point uses `x = -b / v` and `y = v_k * abs(w_k)` by default. The exact quantity is controlled by `evaluation.kink_contribution_mode`.

The reduced-variant figures also include `reduced_bottleneck_representation.png`, which compares each bottleneck coordinate against `scaledSine(x)`. This is the intermediate learned function view, not just the final output prediction.

When `evaluation.save_stage_figures` is true, the runner also saves a whole-model trajectory under `figures/stages/`:

- `00_baseline/`: the unreduced source model
- `01_stack1_after_outside/` through `05_stack1_after_clustering/`
- `06_stack2_after_outside/` through `10_stack2_after_clustering/`

For 2D bottlenecks, `stack2_kink_boundaries.png` is a 2x2 view: the top row shows the stack-2 ReLU boundaries in bottleneck space, and the bottom row shows `bottleneck_0(x)` and `bottleneck_1(x)` with the boundary crossings marked along the 1D input trajectory.

Each stage directory contains:

- `predictions.png`
- `bottleneck_representation.png`
- `bottleneck_plane.png` when the bottleneck has at least 2 dimensions
- `stack1_kink_contributions.png` when `stack1` receives a 1D input
- `stack2_kink_contributions.png` when `stack2` receives a 1D input
- `stack2_kink_boundaries.png` when `stack2` receives a 2D bottleneck input
- `metrics.json`

If `evaluation.show_training_points` is true, the saved prediction plots overlay the dataset points, and the bottleneck figures overlay the same samples in input space or bottleneck space. Use `evaluation.training_points_split: train` or `val` to choose which split is shown.

`figures/stages/manifest.json` records the ordered stage metadata and metrics for the full reduction trajectory.

## Interpreting Metrics

Quality against the true target:

- `baseline_mse`: MSE of the original trained model against the target function.
- `reduced_mse`: MSE of the reduced model against the target function.
- `mse_delta`: `reduced_mse - baseline_mse`; lower is better.
- `relative_mse_delta`: fractional increase over baseline. For example, `0.05` means 5% worse than the baseline MSE.
- `reduced_component_mse`: per-output MSE, useful when one target component fails while the average still looks acceptable.

Fidelity to the trained model:

- `baseline_vs_reduced_mse`: MSE between the original model predictions and reduced model predictions.
- `baseline_vs_reduced_max_abs`: largest pointwise prediction change from reduction.

Compression:

- `baseline_params`: parameter count of the original trained model.
- `reduced_params`: parameter count of the reduced model.
- `compression_ratio`: `baseline_params / reduced_params`; higher means smaller reduced model.
- `reduced_stack1_neurons`, `reduced_stack2_neurons`: final hidden neurons in each stack.

Bottleneck diagnostics:

- `bottleneck_corr_with_scaled_sine`: correlation between each reduced bottleneck dimension and `scaledSine(x)`.
- `bottleneck_best_abs_corr_with_scaled_sine`: best absolute bottleneck correlation. Values near 1 suggest the bottleneck still represents the latent sine-like variable.

Reduction diagnostics:

- `stack1_diagnostics` and `stack2_diagnostics` report how many neurons existed initially, how many were removed by outside-neuron removal, how many weak neurons were discarded, and how many clusters were effectively used.
- `weak_threshold_removed`: neurons removed because their weak score was below `reduction.weak.min_score`.
- `weak_effective_discard`: neurons removed by the fixed-count weakest-neuron pass after threshold pruning.

## Choosing A Candidate

A practical first rule:

```text
choose the highest compression_ratio with relative_mse_delta <= 0.05
```

Then inspect:

- `baseline_vs_reduced_predictions.png`
- `reduced_residuals.png`
- `reduced_bottleneck_representation.png`
- `reduced_bottleneck_plane.png` when the bottleneck has at least 2 dimensions
- `stack1_kink_contributions.png`
- `stack2_kink_contributions.png` when available
- `stack2_kink_boundaries.png` for 2D bottlenecks
- `figures/stages/`
- `tradeoff.png`

If several candidates are close, prefer the simpler one with fewer neurons unless its residuals show structured errors.

## Notes For Future Work

Useful next additions:

- Add a post-reduction fine-tuning stage.
- Add weighted clustering with `sample_weight: cluster_strength` to the default sweep after the unweighted baseline is understood.
- Add Pareto-front selection and a `best.yaml` file.
- Add affine SVD using a model class that supports the six-weight `svd_affine()` output.
