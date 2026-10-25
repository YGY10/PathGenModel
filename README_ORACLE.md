# Oracle lane-identity / exact-geometry experiment

This experiment uses the existing architecture and loss. It deliberately supplies
privileged lane identity by retaining only the selected incoming lane and target
outgoing lane boundaries. Numeric feature geometry comes exclusively from World
physical elements, rigidly transformed into the ego frame. No GT centerline,
GT residual or GT path coordinates construct the input. Coarse Guide and GT are
unchanged. These inputs are an Oracle diagnostic, not a deployable sensor policy.

Export requires the existing source `audit/train.jsonl`, `val.jsonl`, `test.jsonl`.
World split membership is inherited and checked without reshuffling. With current
300-World/7-pose data, the expected counts are 5040 / 630 / 630 samples. Only one
quality copy is retained (`clean` by default), since exact perception removes the
original clean/medium/hard differences. Quality is renamed `oracle` and the original
sample ID is retained in metadata. Source IDs are not numerical model features.

The input envelope is +/-100 m around ego, with no miss/break/noise. Required
boundary identities must all be represented; export fails if any are absent or
capacity would exceed 32 objects. Stop lines, crosswalks and other lane candidates
are omitted. The standard 32-point encoder remains unchanged; `export_report.json`
reports deviation between physical segments and the encoded polylines. This
deviation is a representation diagnostic, not a rigorous lower bound on path ADE.

## 1. Export

```bash
python -m Scenario.export_oracle_lanes \
  --source outputs/v4_world300_inside13 \
  --output-dir outputs/oracle_lane_v1
```

Use a new output directory. Export writes the inherited splits under `audit/`,
plus `samples.jsonl`, copied Worlds/routes, a preview and an export report. No
separate Dataset.audit command is needed to split again. The script also creates
`fit/train.jsonl` using four original training Worlds and `fit/val.jsonl` using
two original validation Worlds. These small splits remain disjoint. It does not
modify original datasets or previous checkpoints.

## 2. Small-data fitting diagnostic

```bash
python train.py --config configs/train_oracle_fit.json
```

Same model/loss/batch size/learning rate as control; no augmentation. Fit run has
1000 epochs, early stopping disabled and plateau scheduling effectively disabled
for that budget. It uses the existing disjoint validation mechanism, but validation
is not the criterion for whether the model can fit training samples.

Training logs have dropout enabled and therefore are not the final fitting score.
Evaluate `last.pt` on its actual training set, in eval mode (dropout disabled):

```bash
python evaluate.py \
  --checkpoint outputs/training_oracle_v1/fit/last.pt \
  --split train \
  --no-empty-bev \
  --output-dir outputs/training_oracle_v1/fit/eval_train
```

This is an intentional training-set fit check, not a held-out result. Do not use
validation-selected best.pt as the sole fitting-capacity diagnostic. If train ADE
is above 0.05m, inspect pose/FDE, input representation, output parameterization and
optimization. Failure of one configuration does not prove an architecture's
absolute precision limit or information-theoretic impossibility.

## 3. Full Oracle training

```bash
python train.py --config configs/train_oracle_full.json
```

Maximum 300 epochs, early stopping after 50 epochs without a best original-input
validation ADE; TensorBoard and console logs retained. The existing logger name
`val_original` here refers to the fixed Oracle inputs (no stochastic augmentation).
Start from scratch; do not resume a mixed-input checkpoint.

```bash
python evaluate.py \
  --checkpoint outputs/training_oracle_v1/full/best.pt \
  --no-empty-bev \
  --output-dir outputs/training_oracle_v1/full/eval_val

python evaluate.py \
  --checkpoint outputs/training_oracle_v1/full/best.pt \
  --split train \
  --no-empty-bev \
  --output-dir outputs/training_oracle_v1/full/eval_train
```

Full 80m ADE remains the main score, origin excluded. Inspect training/validation
ADE, FDE, pose, turn and tail metrics. Train <=5cm but validation above 5cm suggests
generalization/coverage or optimization concerns; both <=5cm supports progressively
restoring uncertainty. Do not compare Oracle scores directly as a production
improvement over the realistic input setup.

## TensorBoard

```bash
tensorboard --logdir outputs/training_oracle_v1 --port 6006
```

Scripts/configurations were validated with synthetic export tests covering World
isolation, removal of duplicate qualities and feature independence from GT values.
No full user dataset export or training is performed as part of preparing files.
