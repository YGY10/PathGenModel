# V1 training

For the matched control/mixed-context experiment with TensorBoard and early
stopping, use [README_CONTEXT_TRAIN.md](README_CONTEXT_TRAIN.md). The prepared
configs are `configs/train_context_control.json` and
`configs/train_context_mixed.json`; original configs remain valid.

Run from the repository root, with Python, PyTorch and NumPy available:

```bash
python train.py --config configs/train_v1.json
```

For the repaired 300-world dataset (world-fixed guides with nonzero first points):

```bash
python train.py --config configs/train_v1_world300_fixedguide.json
```

This writes to `outputs/training_v1_world300_fixedguide/`. It retains the V1
architecture and loss settings; only data/output paths differ. Train from scratch:
old checkpoints were trained on per-pose reanchored guides. The model accepts a
nonzero guide start while keeping prediction index 0 at the ego origin. Do not
subtract the guide's first point in a custom loader or preprocessing step.

The supplied config uses the existing world-disjoint train/val JSONL files.
The script checks world overlap, duplicate sample IDs and empty splits before
training. It never re-splits data or uses the test set for model selection.
Config paths are relative to the current working directory. Change nested
`model` / `loss` JSON fields to configure architecture and supervision; omitted
fields use their dataclass defaults. Unknown fields fail rather than silently
being ignored. This version uses one CPU or CUDA device and FP32.

Useful overrides:

```bash
python train.py --config configs/train_v1.json --device cuda:0 --batch-size 16 --epochs 100
python train.py --resume outputs/training_v1/last.pt --epochs 200
python train.py --config configs/train_v1.json --output-dir outputs/another_run
```

`device=auto` selects CUDA when available, otherwise CPU. The default is zero
DataLoader workers for portability; increase `--num-workers` if data loading is
slow. Batch size can be lowered for GPU memory limits. Resume restores the saved
config unless `--config` is supplied. Epochs, device, num_workers, output_dir,
TensorBoard settings and early-stop patience may change during resume; other
training changes require a new run.
Load checkpoints created by this script from trusted sources only.

Outputs:

- `config.json`: resolved settings, including complete model/loss configuration.
- `data_manifest.json`: split world IDs, sample counts and SHA256 fingerprints.
- `history.jsonl`: per-epoch train/val losses, raw and weighted loss terms,
  ADE/FDE, distance-range ADE, guide baseline and per-navigation model ADE/FDE.
- `last.pt`: latest completed epoch, optimizer, scheduler and RNG states.
- `best.pt`: checkpoint with lowest validation ADE.

ADE excludes the fixed origin; FDE measures the final point. The distance
groups correspond to GT indices 1–20, 21–40 and 41–80 at the current 1 m
sampling. All distance metrics are in metres. Scheduler and optional early
stopping monitor validation ADE. A training interruption can be resumed from
the latest completed epoch; partially completed epochs are not saved. An
existing run is protected against accidental fresh-run overwrites.

Random seeds and shuffle RNG states are saved. CPU resume consistency is tested;
GPU runs may still differ due to nondeterministic kernels. Compare validation
ADE with guide_ADE and inspect FDE alongside it; lower loss alone does not prove
that the model uses BEV. Do not tune hyperparameters against the held-out test set.

An existing environment in this workspace can also be used directly:

```bash
/home/ygy/miniconda3/envs/toybev/bin/python train.py --config configs/train_v1.json
```
