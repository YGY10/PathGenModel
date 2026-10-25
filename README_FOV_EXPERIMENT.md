# Experiment A: supplement observations outside original FOV

```bash
python experiment_fov.py --evaluation-dir outputs/training_v1_world300_fixedguide/evaluation_best_val --worlds-dir outputs/v4_world300_fixedguide/worlds --routes-dir outputs/v4_world300_fixedguide/routes
```

Parameters are in `configs/expanded_fov_A.json`. Default forward extent is at
least 100 m and both lateral extents at least 60 m; rear extent remains unchanged.
This is a diagnostic range, not a physical sensor specification. The original
perception config, training data and model checkpoint are never modified.

The script reconstructs the original 32-slot model input and freezes every valid
point/attribute/mask slot. World geometry is observed with exactly the original
quality's miss, break, offset, point-noise and angle-noise parameters. A separate
seed controls new-region randomness. Only fragments outside the old FOV and within
the expanded FOV survive; full boxes overlapping old FOV are excluded. Selection
uses distance, not target exit identity, navigation relevance or GT proximity.
Supplementary objects fill only padding slots, keeping the capacity at 32x32.
Candidate generation has no intermediate 32-line cap; final model input still does.

This is a *supplementation intervention*, not the output of a fully consistent
large-FOV sensor simulation. Old misses are not repaired. Noise realizations in
new regions are sampled separately; the original observations are bit-identical.
Synthetic source IDs are used only for evaluation, never as model input features.

Inference runs the same checkpoint in eval mode on each original/augmented pair.
Baseline predictions must reproduce the saved evaluation within floating-point
tolerance. Only held-out worlds are accepted. This experiment performs no training,
no quality improvement, no token increase and no origin/guide preprocessing change.

Default output is `expanded_fov_A/` within the evaluation directory:

- `summary.json`: paired original/expanded ADE/FDE, wrong-direction and geometry
  violation rates, broken down by navigation, pose, quality and intervention subsets.
- `per_sample.csv`: prediction changes, free/used capacity, added/truncated object
  counts, and target-exit coverage before/after.
- `predictions.npz`: aligned IDs, baseline, expanded, GT and guide arrays.
- `added_observations.jsonl`: actual retained supplementary geometries, in appended
  slot order, so original plus supplement inputs can be reconstructed.
- `samples/`: largest ADE improvements and deteriorations with observed inputs.

Subsets: all samples; samples with added tokens; samples gaining support at >=5
exit stations; samples with unchanged input. Exit coverage counts 23 route stations
(3–25 m after exit) where BOTH exact target boundary source IDs have a fragment
within 1.25 m of their true boundary projection, using the encoded 32-point lines.
This is geometric support, not sensor recall. No center-FOV mask is applied, since
actual added observations are being measured. Positive ADE gain means improvement.
ADE changes within 1 cm are reported as approximately unchanged.

Use `--config other.json` for a separate seed/range, and `--output-dir new_dir`
to avoid overwriting a completed experiment. This first run uses a single seed;
causal claims should also consider input shift, coverage selection and world-level
variation. If improvements are absent, that does not prove extra information is
useless: the checkpoint may not know how to use the expanded context.
