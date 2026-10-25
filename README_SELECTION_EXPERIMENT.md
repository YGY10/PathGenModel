# Near/Far/Random × added-object count

```bash
python experiment_selection.py --supplement-dir outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A --worlds-dir outputs/v4_world300_fixedguide/worlds --routes-dir outputs/v4_world300_fixedguide/routes
```

Reuses saved observations/noise from one Experiment A run; nothing is regenerated
inside the selection ablation. All arms use the same checkpoint and preserve every
original encoded slot. Capacity is fixed at 32 objects x32 points. Default arms:
baseline, Full, Near/Far/Random with K=4/8/12. Effective K is min(K, retained pool
size); because this pool already fits original padding, policies always add the
same number for each sample at a given K. All subsets restore original pool order.
If K reaches pool size, all policies become identical Full inputs.

Near and Far rank by exact segment-to-guide distance on the encoded representation;
mean point-to-guide distance breaks ties. Closed boxes use their perimeter, not
filled area. This includes all object types. Source IDs, GT and target-exit IDs
play no role in ranking. Random uses a fixed seed derived from sample ID; K arms
use prefixes of the same ordering. Candidate length, type, confidence and local
direction alignment are retained in candidates.csv to expose confounders.

The candidate pool is limited to objects RETAINED in the original A run. Objects
previously discarded by range/capacity selection are not recovered. Therefore Far
means far from guide *within this pool*, not globally farthest within expanded FOV.
This restriction enables exact Full reproduction and avoids any new observation
randomness. Additional perception seeds require separate experiment_fov runs first.

Output: `selection_ablation/` within the supplement directory:

- summary.json: all-sample, per-arm full-budget and common-Kmax-eligible summaries;
  world/navigation/pose/quality groups; paired world-equal ADE gains.
- per_sample.csv: variant, ID, actual K, boundary coverage, metrics and paired gain.
- candidates.csv: geometry descriptors for each candidate pool index.
- selections.jsonl: actual chosen pool indices and object-type counts per variant.
- predictions.npz: aligned baseline/full/Near/Far/Random predictions, GT and guide.

Baseline and Full predictions must reproduce their saved experiment A counterparts
within floating-point tolerance. Exit coverage is an evaluation-only measure using
both target boundaries at 23 stations (3–25 m after exit), tolerance 1.25 m. Common
budget subset restricts all arms to samples with at least max(K) candidates; use it
when comparing count-response curves. Full-budget subsets differ across K, so
compare Near/Far/Random within K on those subsets, not across K without qualification.

Use `--selection-seed`, `--budgets`, `--batch-size`, `--device` and a new
`--output-dir` for controlled variants. Do not pick a per-sample winner using GT.
This measures inference sensitivity, not retrained performance. A Near advantage
supports studying geometry relevance but does not identify whether the spatial
encoder or decoder cross-attention is responsible. Guide proximity need not imply
correct-lane relevance, especially when the guide itself is biased.

## Repeated supplement seeds

The first screen reuses supplement seed 724. Independent observation realizations
with identical FOV and quality settings can be prepared with
`configs/expanded_fov_A_seed725.json` and `configs/expanded_fov_A_seed726.json`
via experiment_fov.py, each using a distinct output directory, then evaluated by
experiment_selection.py. Selection seed stays fixed at 42. Repetition changes
new-region sensing randomness, not original observations or model parameters.

After these runs, summarize with:

```bash
python summarize_selection_experiments.py --runs outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A/selection_ablation outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A_seed725/selection_ablation outputs/training_v1_world300_fixedguide/evaluation_best_val/expanded_fov_A_seed726/selection_ablation --output-dir outputs/training_v1_world300_fixedguide/evaluation_best_val/selection_ablation_multiseed
```

Outputs include per-seed trials.csv, aggregate summary.json and comparison.png.
The plotted shade is min/max across supplement seeds, not a confidence interval.
For world-level behavior, consult each trial's groups/world_id results. These
trials share the same validation worlds, so they do not multiply independent
world count. All settings are fixed before inspecting additional-seed outcomes.
