# PathGenModel: V4 generator + V5 audit layer

## Continue research on another computer

Start with [README_MIGRATION.md](README_MIGRATION.md) and
[RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md). This repository includes checksum-verified
research snapshot parts under `research_artifacts/`; run
`python scripts/restore_research.py` after cloning to restore the current datasets,
Oracle checkpoints, evaluations and recent control/mixed experiments.
Large duplicate historical datasets and reproducible training caches are omitted.

## V4 ego poses

New V4 generation visits `approach_30`, `approach_20`, `approach_10`, `entry`,
`inside_8`, `inside_13`, `exit_5`. `inside_13` means 13 m of GT arc length after
the route's entry, and is omitted if that station is at or beyond its exit.
Consequently, with three perception qualities, a World has up to 63 samples;
the exact count depends on which routes still contain `inside_13`.
Preview shows seven columns and marks skipped positions. Original six poses keep
their perception RNG IDs; adding the new pose does not resample their observations.
FOV supplement generation supports the new pose as well. Existing datasets are
not rewritten: use a new output directory for generation and run the usual audits
to produce world-disjoint train/validation/test files.

## Fixed world guide convention

V4 now preserves a single `guide_world` per route across all ego poses. Samples
only crop that route and transform it into ego coordinates; they do not translate
the guide to make its first point zero. `reference_line[0]` may therefore be
nonzero. GT and predicted path still start at the ego origin. The convention is
recorded as `world_fixed_ego_transform_v1` with `guide_start_s_m` in metadata.
This preserves the existing world-guide geometry; it does not impose a constant
lateral offset along every part of the route or change nearest-vertex cropping.

Repair a previously generated V4 dataset without resampling worlds, GT or perception:

```bash
python -m Scenario.reexport_guides --source outputs/v4_world300 --output-dir outputs/v4_world300_fixedguide
python -m Dataset.audit --input outputs/v4_world300_fixedguide/samples.jsonl --worlds-dir outputs/v4_world300_fixedguide/worlds --output-dir outputs/v4_world300_fixedguide/audit --seed 42
python -m Dataset.audit_extended --input outputs/v4_world300_fixedguide/samples.jsonl --worlds-dir outputs/v4_world300_fixedguide/worlds --routes-dir outputs/v4_world300_fixedguide/routes --output-dir outputs/v4_world300_fixedguide/audit_extended
```

Use a new output directory. The reexport report checks non-guide fields are
identical, reconstructs guides against saved world routes and regenerates preview.
The extended audit checks fixed-guide world consistency as well as GT closure.
Use the original split seed to preserve world assignments. Old checkpoints and
their evaluation results refer to the old anchored guides; retrain for comparisons
under this convention. Legacy `Scenario.build_dataset` retains its old behavior;
use `Scenario.build_dataset_v4` for new fixed-world-guide datasets.

This package **does not modify** V4 geometry generation. It adds `Dataset/audit.py` to create disjoint-world splits and audit training distributions.

## Commands

```bash
pip install numpy matplotlib
python -m Scenario.build_dataset_v4 --worlds 20 --seed 20261023 --perception-config perception_v4.example.json --output-dir outputs/generation
python -m Dataset.audit --input outputs/generation/samples.jsonl --output-dir outputs/audit --seed 42
```

Results: `train.jsonl`, `val.jsonl`, `test.jsonl`, `split_manifest.json`, `summary.json`, `distribution.csv`, `per_sample.csv`.

**Do not perform row-level splitting again in `BEVDataset`.** Use these split-specific JSONL files with `split='all'` to preserve world-disjoint partitions.

## Box perception in V4

Stop line and crosswalk use `box_miss_rate`, `forward_m`, `rear_m`, `left_m`, `right_m`, `angle_sigma_deg`, `offset_sigma_m`, `sigma_m` (via the common rectangle estimator). They **do not** use `line_miss_rate` or `break_rate`. Perception emits complete four-corner boxes or misses them. FOV masking is asymmetric; the legacy rectangle estimator uses a conservative symmetric lateral range internally, so the border case is a heuristic, not exact sensor visibility.

## Audit limitations

- `visibility_proxy` estimates two-sided GT-adjacent geometric support from observed line pieces. It is **not** a confirmed target-lane two-boundary test; it can classify irrelevant nearby lines. Treat as provisional only.
- `world_id` separation is enforced. Same-template or near-identical worlds may remain across splits; `world_family_id` support and geometry-family holding-out are still future work.
- The original V4 generator can reject invalid geometries for some seeds. Run generation and audit before training; no assurance of every seed succeeding.
- The generator still chooses one route per direction/world. Real-world lane restrictions are not modeled fully.
- A 1-world smoke test can produce zero-world validation or test partitions; use many worlds for real splits.
