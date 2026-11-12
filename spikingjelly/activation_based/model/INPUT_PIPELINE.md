# Modular input pipeline

Function-based encoders and their calibration helpers live in `input_encoder_fn.py`.
The old `training_script.py` imports them and keeps backbone construction and input-layer
adaptation. Module-based encoders live in `input_encoder_module.py` (renamed from
the previous plural filename), imported by `training_script_new.py`.

The function-based `training_script.py` also
supports the function-based population experiment described below. The new
`training_script_new.py` keeps the `SResNetTrainer` name and the original parser
and backbone-construction structure, with encoder selection and wrapping added.
It inherits from `train_classify_new.Trainer`, an independent copy of the original
trainer. Neither new file imports or inherits from either old training file.

```text
batch → preprocess_train_sample / preprocess_test_sample (identity)
      → ModelWrapper(input_encoder → backbone → decoder)
      → process_model_output (identity) → loss
```

The preprocessing hooks remain available for noise or other data transformations.
They do not perform encoding in the new path.

## Encoders and configuration

The available module encoders are `identity`, `repeat`, `latency`, `poisson`, `quantization`, `burst`,
`phase`, and `population`. Select one with `--encoder`. The default is `repeat`;
the default architecture is `spiking_resnet18`. `--T` retains the old parser's
default of 4, so phase coding needs an explicit compatible value such as 64.
Both fn and module interfaces infer P=T/4 when no phase period is supplied,
giving four complete cycles. T must then be divisible by 4, and the inferred
period must pass the existing tau-dependent calibration checks. An explicit
period overrides this rule; T must still contain complete cycles.

Examples, from the repository root:

```bash
python spikingjelly/activation_based/model/training_script_new.py \
  --arch sew_resnet18 --encoder repeat --T 64

python spikingjelly/activation_based/model/training_script_new.py \
  --arch spiking_resnet18 --encoder phase --T 64 --phase-period 16 --phase-tau 2

python spikingjelly/activation_based/model/training_script_new.py \
  --arch resnet18 --encoder identity
```

Add the dataset/device/training arguments appropriate to the experiment.
`-c` accepts an existing YAML file or a filename in `models/training_configs/`.
No YAML is selected implicitly. As in the old script, YAML values take precedence
over command-line values. Encoder YAML keys use underscores, for example:

```yaml
encoder: phase
T: 64
phase_period: 16
phase_tau: 2.0
```

Other options are `--encoder-mean`, `--encoder-std`, `--burst-max-spikes`,
`--burst-isi-min`, `--burst-isi-max`, and `--population-size` (default 3).
Both training paths support `--encoder quantization`, which distributes
round(x*T) spikes across T steps. Latency accepts `--latency-mode linear|log`
and `--latency-zero-silent`; defaults remain linear and zero_silent=False.
YAML uses `latency_mode` and `latency_zero_silent` (a boolean).

Module encoders call the functions in `input_encoder_fn.py` directly, sharing
normalization, arithmetic and calibration. Poisson equivalence requires the same
initial RNG state. Phase and population create fresh evaluation-mode LIF neurons
per call, so wrapper `.train()` / `.eval()` does not change encoding or carry
membrane state across batches. `extra_repr()` shows each encoder's configuration;
population also shows the calibrated tau and widths.

Population uses Gaussian currents followed by binary LIF spikes, with output
`[T,B,C*P,H,W]`, ordered by input channel then anchor. For example:

```bash
python spikingjelly/activation_based/model/training_script_new.py \
  --arch sew_resnet18 --encoder population --T 32 --population-size 3
```

The new path expands the first CNN convolution before optimizer creation, using
fresh weights. The old continuous-current prototype (`M`, `sigma`, stored anchors)
is replaced by the function's calibrated `population_size` API. Encoder modules
have no parameters or persistent buffers; select the configuration explicitly
when restoring checkpoints, including new wrapper checkpoints.

Sequence encoders return [T,B,C,H,W], followed by
a temporal-mean decoder. Ordinary ANN backbones process the flattened T/B batch;
multi-step SNN backbones receive the sequence directly. `identity` passes its input
through unchanged and uses an identity decoder; it is for ordinary ANN backbones.
Use `repeat` for a static image passed into a multi-step SNN.

## Old and new checkpoints

### Population experiment in the old script

Use `training_script.py --arch sew_resnet18 --encoder population --T 32`,
with your usual dataset and training arguments. By default three anchors (0, 0.5, 1)
generate Gaussian currents, followed by independent LIF neurons with fresh state
for each batch. RGB becomes binary spikes of shape `[T,B,9,H,W]`; the existing
input-layer adapter expands ResNet's stem before optimizer creation.

Calibration prefers tau=2 within (1,5], fixes peak current and threshold to one,
and uses subtractive reset with `decay_input=False`. At half of each anchor's
maximum distance, the target count is round(T/e); Gaussian widths are computed
per T from the midpoint of the admissible current interval. The preferred input
fires every step, while the weakest input may remain silent. T=16/32/64 all use
tau=2. Remove the former `population_sigma` config key; `population_size` defaults to 3 and accepts integers >=2. Anchors are evenly
spaced in [0,1]; each width scales with that anchor's farthest endpoint distance.
RGB then has 3*population_size channels. Changing P changes checkpoint shapes.
The notebook `notebooks/input_encoding_demo.ipynb` compares the function-based
encoders, including spike rasters, count curves, RGB channel expansion and
an independent NumPy recurrence comparison for phase coding.

The module-based path uses the same experiment. Its stem cannot load a
three-channel checkpoint or a different population size without explicit adaptation.

### Modular checkpoint compatibility

Use the existing `--resume /path/to/checkpoint.pth`, optionally with `--test-only`.
Explicit paths can point into the old output directories. `--resume latest`
searches the new run's directory, not an old run's directory.

Choose the same architecture, neuron settings, T and encoding as the old experiment.
Encoding is not inferred from old weights: it defaults to repeat, and other old
encodings must be selected explicitly. The new script also saves its configuration
in checkpoint `args`; the inherited loop does not automatically apply saved args.

The wrapper maps old backbone keys to `backbone.*` when loading, including the
backbone inside EMA. New checkpoints retain the wrapper structure. Missing keys,
unexpected keys and incompatible shapes still fail strict loading. The selected
fixed encoders add no optimizer parameters, preserving backbone parameter order
for optimizer restoration; optimizer type and parameter-group settings must match.

The original checkpoint format does not save RNG states or sampler position.
Resuming restores the states the old trainer saves; it does not promise a bitwise
reproduction of an uninterrupted stochastic training run.

## Scope and validation

Tested backbone factories: resnet18, spiking_resnet18, sew_resnet18, vgg11 and
spiking_vgg5. Transformer/Spikformer and tdBN factory adaptation are deferred and
rejected explicitly. The existing ordinary vgg5 factory has a feature/classifier
width mismatch; its source is left unchanged.

Run the CPU regression tests with:

```bash
python -m unittest discover -s spikingjelly/activation_based/model/tests -v
```

Tests compare old functions and modules exactly, check phase state isolation and
population binary output, layout, dtype and state isolation, exercise CNN forward/backward, and
compare a resumed optimizer step and EMA against the old path. A synthetic dataset
runs the old trainer through checkpoint creation and the independent new trainer
through resume and old/new checkpoint validation. A fresh-process test blocks imports
of both old training files and checks that the new path still constructs its model.
GPU, AMP and multi-process DDP execution require separate hardware validation.

## Function-based encoder selection in the old script

`training_script.py --encoder` accepts `repeat` (default), `poisson`, `quantization`, `latency`,
`burst`, `phase`, and `population`. Train and test hooks use the same settings;
Poisson still samples fresh spikes on each call. All these paths emit a time axis.
The shared options are `--T`, `--encoder-mean`, and `--encoder-std` (repeat keeps
its input unchanged before repetition).

- Phase: `--phase-period 16 --phase-tau 2`, for example with `--T 64`.
  Omitting `--phase-period` uses T/4, matching the module-based path.
  T must be a multiple of the period and the phase feasibility checks must pass.
  Default T=4 is unchanged, so explicitly set T when selecting phase.
- Burst: `--burst-isi-min 1 --burst-isi-max 8`, optionally `--burst-max-spikes 3`.
  Without a cap, the train continues throughout T. ISI >= T may give only one spike.
- Population: `--population-size 3`; widths and encoder tau are calibrated.
- Latency: `--latency-mode linear|log`, optionally `--latency-zero-silent`.
- Poisson and quantization need no additional encoder-specific options.

YAML uses the same names with underscores, e.g. `phase_period`, `phase_tau`,
`burst_max_spikes`. The existing YAML-over-command-line precedence is unchanged.
