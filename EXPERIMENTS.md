# Composite Sine Experiments

Each experiment is defined by one YAML file in `configs/experiments/`.

## Experiments

- `composite_sine_plus_bottleneck1.yaml`: the existing setup, using `compositeSinePlus` with a 1D bottleneck.
- `composite_sine_bottleneck1.yaml`: uses `compositeSine` with a 1D bottleneck.
- `composite_sine_plus_bottleneck2.yaml`: uses `compositeSinePlus` with a 2D bottleneck.

## Train

Commands below assume the project virtual environment is active. If it is not active, use `.venv\Scripts\python.exe` instead of `python`.

```powershell
python train_composite_sine.py --config configs/experiments/composite_sine_plus_bottleneck1.yaml
python train_composite_sine.py --config configs/experiments/composite_sine_bottleneck1.yaml
python train_composite_sine.py --config configs/experiments/composite_sine_plus_bottleneck2.yaml
```

Training writes each run under:

```text
runs/{experiment.name}/
```

Each run contains a copied `config.yaml`, Lightning logs, and checkpoints.

## Visualize

```powershell
python visualize_composite_sine.py --config configs/experiments/composite_sine_plus_bottleneck1.yaml --save --no-show
python visualize_composite_sine.py --config configs/experiments/composite_sine_bottleneck1.yaml --save --no-show
python visualize_composite_sine.py --config configs/experiments/composite_sine_plus_bottleneck2.yaml --save --no-show
```

Figures are saved under:

```text
runs/{experiment.name}/figures/
```

The visualization script also saves kink plots for every stack that receives a 1D input:

- `stack1_kink_contributions.png`
- `stack2_kink_contributions.png` when the bottleneck is 1D

These plots use `x = -b / v` and `y = v_k * abs(w_k)` by default, so each output view shows one point per hidden neuron at its ReLU kink.

For 2D bottlenecks, the visualization script also saves:

- `bottleneck_plane.png`
- `stack2_kink_boundaries.png`

In `stack2_kink_boundaries.png`, the top row shows the stack-2 ReLU boundaries in bottleneck space, and the bottom row shows `bottleneck_0(x)` and `bottleneck_1(x)` with the stack-2 boundary crossings marked back in 1D.

In the `bottleneck_representation.png` we are asking: "Can I read off scaledSine(x) from where the point lands in bottleneck space?", like "can I fit a line to the bottleneck representation of the data, and does it capture the underlying function (here scaledSine)?".

If you want to overlay the dataset samples on these plots, enable `evaluation.show_training_points: true` in a reduction sweep, or pass `--show-training-points --training-split train` to `visualize_composite_sine.py`.

## Reduce

Reduction sweeps are configured separately from training configs.

```powershell
python -m reduction_experiments.run --config configs/reductions/composite_sine_reduction_sweep.yaml
```

If the system Python imports the wrong PyTorch build, run the same command through the project venv:

```powershell
.venv\Scripts\python.exe -m reduction_experiments.run --config configs/reductions/composite_sine_reduction_sweep.yaml
```

The default reduction sweep uses `last.ckpt` for each trained model and writes outputs under:

```text
runs/{experiment.name}/reductions/{reduction.name}/
```

See `docs/reduction_experiments.md` for the directory layout, metrics, and interpretation guide.

You can also visualize a reduced checkpoint directly with the same script:

```powershell
python visualize_composite_sine.py --config configs/experiments/composite_sine_plus_bottleneck1.yaml --checkpoint runs/composite_sine_plus_bottleneck1/reductions/composite_sine_reduction_sweep/variants/s1_k0012_w0000_s2_k0012_w0000/reduced_checkpoint.pt --save --no-show
```

## Existing Checkpoint

If you want to visualize the checkpoint created before the `runs/` structure, use:

```powershell
python visualize_composite_sine.py --config configs/experiments/composite_sine_plus_bottleneck1.yaml --checkpoint checkpoints/composite_sine/last.ckpt --save --output-dir figures/composite_sine --no-show
```
