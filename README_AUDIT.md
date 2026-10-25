# Intersection V4 + dataset audit layer

Core generator remains `python -m Scenario.build_dataset_v4`. Audit is separate:

```bash
python -m Scenario.build_dataset_v4 --worlds 30 --seed 20261023 --output-dir outputs/v4_world30
python -m Dataset.audit --input outputs/v4_world30/samples.jsonl --worlds-dir outputs/v4_world30/worlds --output-dir outputs/v4_world30/audit --seed 42
python -m pytest -q tests/test_audit.py
```

Outputs: `train.jsonl`, `val.jsonl`, `test.jsonl`, `split_manifest.json`, `per_sample.csv`, `visibility_by_group.csv`, `issues.csv`, `summary.json`.

## Interpretation

- Split unit is **world_id** (stable hash by seed, largest remainder world counts). `world_id` repeats only within one split.
- `two_sided_proxy` is *not* a strict target-exit-lane visibility test. It finds lines or curbs near either side of GT in the future half of the path; some may belong to unrelated lanes. It is an exploratory statistic only.
- `detected_box_counts` counts returned detections, **not** detection probability; the number of boxes within FOV varies with ego pose. Exact recall requires matched truth boxes and opportunity eligibility.
- Stop/crosswalk `box_miss_rate` is separate from `line_miss_rate`, `curb_miss_rate`, and `break_rate` in V4. Boxes are constrained to four corners by the estimator.
- `gt_lane_violation` and world-family grouping are not yet available: world truth lacks per-sample target-lane polygon identity and explicit `world_family_id`; do not treat a green audit summary as proof of full geometry validity.
- A single World can only belong to one split. Use multiple Worlds for meaningful train/val/test outputs.

The script does not modify V4 generator or model inputs. It can be run on already generated `samples.jsonl`.
