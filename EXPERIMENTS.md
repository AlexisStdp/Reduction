# Composite Sine Experiments

Each experiment is defined by one YAML file in `configs/experiments/`.

## Experiments

- `composite_sine_plus_bottleneck1.yaml`: the existing setup, using `compositeSinePlus` with a 1D bottleneck.
- `composite_sine_bottleneck1.yaml`: uses `compositeSine` with a 1D bottleneck.
- `composite_sine_plus_bottleneck2.yaml`: uses `compositeSinePlus` with a 2D bottleneck.

## Train

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

For 2D bottlenecks, the visualization script also saves `bottleneck_plane.png`.

In the `bottleneck_representation.png` we are asking: "Can I read off scaledSine(x) from where the point lands in bottleneck space?", like "can I fit a line to the bottleneck representation of the data, and does it capture the underlying function (here scaledSine)?".

## Existing Checkpoint

If you want to visualize the checkpoint created before the `runs/` structure, use:

```powershell
python visualize_composite_sine.py --config configs/experiments/composite_sine_plus_bottleneck1.yaml --checkpoint checkpoints/composite_sine/last.ckpt --save --output-dir figures/composite_sine --no-show
```
