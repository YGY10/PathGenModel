# Matched control vs mixed-context training

The architecture and loss are unchanged. Both configs use seed 42, the same base
train/val worlds, optimizer, batch size, scheduler and maximum epochs. A model
initialization fingerprint is saved in checkpoints for the paired evaluation.
Start both from scratch; do not resume an old model for the mixed arm.

## 1. Training-only supplementary cache

```bash
python -m Scenario.build_fov_supplements --input outputs/v4_world300_fixedguide/audit/train.jsonl --worlds-dir outputs/v4_world300_fixedguide/worlds --output-dir outputs/context_train300 --workers 4
```

If `outputs/context_train300/manifest.json` already exists and passes validation,
this step is complete. Cache generation uses source train worlds only, with two
new-region noise seeds 1031/1032, separate from evaluation seeds 724/725/726.
Original perception quality, GT, guide, nav and original observations remain
unchanged. New ranges: forward >=100m, lateral >=60m; rear unchanged. Only retained
new-region candidates that fit remaining 32-slot capacity are cached, ordered by
range without using GT or target IDs. JSONL generation is streamed by world with
bounded parallel work. A manifest is written only after all files complete.
Use a fresh output directory if an incomplete cache must be regenerated.

## 2. Train

```bash
python train.py --config configs/train_context_control.json
python train.py --config configs/train_context_mixed.json
```

Outputs: `outputs/context_comparison/control/` and `outputs/context_comparison/mixed/`.
The control has no augmentation. The mixed arm chooses per sample/epoch:

| Profile | Probability |
|---|---:|
| Original | 50% |
| Add up to 4 | 12.5% |
| Add up to 8 | 12.5% |
| Add up to 12 | 12.5% |
| Add all retained | 12.5% |

One of the two cached noise realizations is sampled for an augmented item. A random
subset is selected; it is never selected by GT/exit-lane association. Original
valid slots are preserved exactly, supplements only fill padding. Actual additions
may be fewer than requested; epoch logs include realized profile fractions and
mean added tokens. Augmentation depends on epoch/sample ID/seed, independent of
worker access order. Workers are not persistent; set_epoch propagates each epoch.
All base and cache fingerprints are checked at startup and checkpoint resume.

Validation always uses fixed original inputs, and best.pt is selected by original
validation ADE for both arms. Expanded-input evaluation is a separate fixed
comparison after training, not a stochastic validation target.

## 3. TensorBoard during or after training

```bash
tensorboard --logdir outputs/context_comparison --port 6006
```

Open the local URL printed by TensorBoard. Both runs write events to their own
`tensorboard/` directory; normal console epoch lines and history.jsonl are retained.
Tags: `train/*`, `val_original/*` (loss/ADE/FDE/guide baseline/navigation groups),
`train/augmentation/*`, `optimizer/lr`, `selection/best_val_ade_m`,
`early_stop/stale_epochs`, `early_stop/patience`. Event files flush each epoch and
close even on interruption. Resume purges event steps beyond the saved epoch.
Install the `tensorboard` package in the training environment if necessary;
`--no-tensorboard` disables it. The toybev environment was checked for availability.

## 4. Early stopping and resume

Both configs default to `early_stop_patience=20`: after 20 consecutive completed
epochs without a strictly lower original-validation ADE, training stops. An exact
tie counts as no improvement. Set n in JSON or override:

```bash
python train.py --config configs/train_context_mixed.json --early-stop-patience 15
python train.py --resume outputs/context_comparison/mixed/last.pt --epochs 150
```

`0` disables stopping. Saved stale count is restored on resume; a checkpoint that
already meets the threshold requires a larger patience or `0` to continue. Last
and best checkpoints are saved before exiting. CPU exact resume with multiworker
augmentation is tested; GPU kernels may remain nondeterministic. Early stopping
can select different epoch counts between arms; both share the same maximum budget
and stopping policy.

## 5. Compare fixed original and expanded validation inputs

```bash
python evaluate_context_comparison.py --control outputs/context_comparison/control/best.pt --mixed outputs/context_comparison/mixed/best.pt --supplement-dirs outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A_seed725 outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A_seed726 --worlds-dir outputs/v4_world300_fixedguide/worlds --routes-dir outputs/v4_world300_fixedguide/routes --output-dir outputs/context_comparison/evaluation
```

The comparison reuses actual saved validation supplements without resampling.
Reports original and each expanded seed for both models, including ADE/FDE,
wrong-direction and road/target-lane violation rates and world/navigation/pose/
quality groups. Checks matched initialization/config/base train membership and
fixed validation identity; the test set is not used. Outputs are summary.json,
per_sample.csv and aligned predictions.npz. A good mixed run should handle added
context better without materially degrading original-input performance.
