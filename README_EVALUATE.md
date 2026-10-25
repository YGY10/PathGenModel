# Evaluation and visualization

From the repository root, using the toybev environment:

```bash
python evaluate.py --checkpoint outputs/training_v1/best.pt
```

Loads the checkpoint's architecture, max_lines/max_points and validation path.
Default output: `outputs/training_v1/evaluation_best_val/`. An existing completed
evaluation is protected against overwrite; use a new `--output-dir` to repeat.
Requires PyTorch and NumPy; PNG rendering additionally requires Matplotlib.
All inference is single-device FP32. Checkpoints must be from trusted sources.

Outputs:

- `summary.json`: checkpoint epoch, data fingerprint, overall and per-navigation,
  quality, pose, world metrics. Includes mean, per-sample ADE/FDE percentiles,
  guide improvement and fraction of samples beating guide.
- `per_sample.csv`: stable input row index, ID, metadata, per-sample metrics.
- `predictions.npz`: aligned IDs, pred/gt/guide and optional empty_bev arrays.
- `overview.png`: mean position error by station and empirical ADE CDF.
- `samples/*.png`: full-horizon and first-30m views, observed model-input geometry,
  GT, guide, prediction, optional empty-BEV prediction and ego/FOV.
- `plot_manifest.json`: sample image mapping and selection tags.

Best/worst/random samples are selected by model ADE within each navigation;
selection is seeded and deduplicated. Default count is two per selection type
per navigation (at most 18 images). Selection is not quality/pose-stratified;
inspect group tables or use per_sample.csv to diagnose those dimensions.
Plot axes show forward upward and left toward the left; geometry is taken from
the actual encoded/truncated inputs, not the complete truth world.

```bash
# Specify dataset/device/output and increase example count
python evaluate.py --checkpoint outputs/training_v1/best.pt --split val --device cuda:0 --examples-per-group 4 --output-dir outputs/eval_v1_detailed

# Training-set diagnostics
python evaluate.py --checkpoint outputs/training_v1/best.pt --split train

# Final held-out evaluation, after choices are fixed
python evaluate.py --checkpoint outputs/training_v1/best.pt --split test --input outputs/v4_world30_retry/audit/test.jsonl

# Metrics only, without empty-BEV inference or plotting
python evaluate.py --checkpoint outputs/training_v1/best.pt --no-plots --no-empty-bev --output-dir outputs/eval_v1_metrics
```

ADE excludes fixed index 0. Distance ranges use indices 1–20, 21–40, 41–80
under the current uniform 1 m GT sampling. Positive `ade_gain_m` means the model
beats guide; positive `empty_bev_ade_delta_m` means removing BEV worsens ADE.
The empty-BEV experiment clears points, masks AND attributes while preserving
guide/navigation. It measures inference-time sensitivity, not the performance
of a separately trained guide-only model. It may introduce a distribution shift.

Validation/test reject training-world overlap. Training diagnostics explicitly
use `--split train`. Default split files are fingerprint-checked against the
checkpoint; `--input` explicitly evaluates a new file. Rows must have unique IDs
and nonempty world IDs. Metadata is used only for reporting, never model inputs.
Samples share worlds/routes, so percentiles are descriptive, not confidence
intervals. Drivable legality and lane correctness are not computed here.

Extension points in evaluate.py: `load_model` handles architecture construction,
`predict` handles inference/ablations, `sample_records` computes aligned metrics,
`grouped_summaries` controls report dimensions, and `plot_sample` controls visuals.

## Corrections that worsen the guide

After evaluation, render the largest regressions without running inference again:

```bash
python visualize_regressions.py --evaluation-dir outputs/training_v1/evaluation_best_val
```

Open `regressions_ade/index.html` inside that evaluation directory for the top 12
regressions, with path overlays and error-versus-station curves. `ranked.csv`
contains every regressed sample, ordered by prediction ADE minus guide ADE.
Use `--top-k 58` for more plots, `--metric fde` to rank by final-point regression,
or `--tolerance-m 0.05` to ignore deterioration of at most 5 cm. Use a new
`--output-dir` to repeat an existing report. Saved predictions are aligned and
fingerprint-checked against the source JSONL; the original trusted checkpoint
must remain available to recover its encoding configuration. Positive
regression means worse; it has the opposite sign to evaluation's `ade_gain_m`.

## Aggregate correction types

```bash
python analyze_corrections.py --evaluation-dir outputs/training_v1/evaluation_best_val --worlds-dir outputs/v4_world30_retry/worlds --routes-dir outputs/v4_world30_retry/routes
```

Writes `correction_statistics/summary.json`, `per_sample.csv` and `grouped.csv`.
Default station tolerance is 10 cm; a sample flag requires at least 25% of future
stations. Change with `--tolerance-m` and `--min-station-fraction`. The JSON also
reports sensitivity at 5/10/20 cm and group results on all vs regressed samples.
Flags can overlap and under-correction can still improve guide. Geometry is V4
specific. Use a new `--output-dir` to rerun a completed report.

## Truth and residual diagnostics

```bash
python diagnose_paths.py --evaluation-dir outputs/training_v1/evaluation_best_val --worlds-dir outputs/v4_world30_retry/worlds --routes-dir outputs/v4_world30_retry/routes --indices 48 51 44
```

This reads saved predictions only. Outputs in `truth_diagnostics/`: each sample's
full truth/observed-input overlay, tangent/left-normal residual plots, residual
arrays and `report.json`. Truth target boundaries, reconstructed target lane and
drivable surface are used exclusively for diagnostics. The report checks GT
route closure, observed source IDs, exit-center FOV eligibility, correction
direction/magnitude, and centerline geometry legality with 8 cm tolerance.
Residuals compare equal point indices in the guide frame, rather than an exact
Frenet reprojection. Distinguish longitudinal sampling errors from lateral shape
errors. First-point clamping can contribute a heading change at the first segment.
Boundary source presence alone does not establish full station-level visibility.
The road/lane reconstruction is specific to this V4 generator; do not use it for
other geometry schemas without a matching adapter. Source JSONL fingerprints and
prediction IDs are checked; supplied world/route files should be from that same
generation run. Existing reports require a different `--output-dir` to rerun.
