# 路口 + 增道/减道联合训练

在项目根目录激活 toybev 后，按顺序执行下面三步。脚本已准备好；完整数据生成和训练由你执行。

## 1. 生成 600 个分流 World

```bash
python -m Scenario.build_lane_transition \
  --worlds 600 --seed 42000000 \
  --guide-shape piecewise_linear \
  --perception-config perception_v4.example.json \
  --preview-worlds 5 \
  --output-dir outputs/lane_transition_world600_v1
```

生成约 17280 条样本，480/60/60 个 World 划分为 train/val/test。
循环覆盖 1–4 条车道、变化 1–2 条的所有数量组合和左右侧。每条连接有三 pose、三种感知质量。
仅绘制前 5 个 World，避免批量生成大量预览。已有目录不会被覆盖。
低质量输入可能全部漏检，仍保留为空 BEV 样本；模型支持此情况，数量记录在 generation_report.json。
过渡长度 15–25m、无内部标线、路沿+内侧白色实线、固定的分段一次 Guide 均沿用已确认版本。

## 2. 混合，并统一路口感知断裂规则

```bash
python -m Scenario.prepare_mixed_dataset \
  --source intersection=outputs/v4_world300_inside13 \
  --source lane_transition=outputs/lane_transition_world600_v1 \
  --refresh-perception intersection \
  --perception-config perception_v4.example.json \
  --seed 7301 \
  --output-dir outputs/road_mixed_v1
```

读取原路口 World、ego pose，重新导出三档感知，使路口/分流都采用 VCS 可见段内断裂。
不修改原数据目录，GT、Guide、自车 pose、World 的 train/val/test 归属保持原样。
旧感知 support 诊断字段会清空，避免将旧输入支持信息用于新观测。
所有输入逐行检查形状、非有限值、重复 ID 和跨集合/来源 World 冲突；发现问题就报错。
不重复采样或补齐样本数量。预计混合后约 36180 条，分流约 48%，路口约 52%。
最终训练实际比例见 `mix_manifest.json`，它记录逐来源的数量和数据 SHA-256。

可重复 `--source 名字=目录` 扩展来源；每个来源需有 `audit/train.jsonl`、`val.jsonl`、`test.jsonl`。
`--refresh-perception` 可重复指定来源。重导出所需的 World 支持 `worlds/<id>.json` 或 `worlds/<id>/world.json`。
混合目录保留原始 World 路径引用；不要删除来源数据。

## 3. 从头训练

```bash
python train.py --config configs/train_road_mixed_v1.json
```

当前架构、32×32 编码、Loss 保持原配置；从头初始化，不加载 Oracle 或旧 mixed checkpoint。
最多 150 epochs，连续 25 epochs 没有更好的混合验证 ADE 则早停，保留原终端日志。
不启用扩展 FOV 增强，避免在增加新场景时同时改变感知范围；现有扩展缓存不适用于新 pose。

每轮额外输出路口/分流验证 ADE，写入 history.jsonl 和 TensorBoard：

```bash
tensorboard --logdir outputs/training_road_mixed_v1/tensorboard
```

关注 `val_original/ade_m`、`val_groups/intersection/ade_m`、`val_groups/lane_transition/ade_m`。
best.pt 和学习率调度仍依据混合验证集的全 80m、排除起点的 ADE，两个子集用于诊断。
新训练输出为 `outputs/training_road_mixed_v1`。

## 训练后评估

```bash
python evaluate.py --checkpoint outputs/training_road_mixed_v1/best.pt

python evaluate.py \
  --checkpoint outputs/training_road_mixed_v1/best.pt \
  --input outputs/road_mixed_v1/subsets/intersection/val.jsonl \
  --output-dir outputs/training_road_mixed_v1/eval_intersection

python evaluate.py \
  --checkpoint outputs/training_road_mixed_v1/best.pt \
  --input outputs/road_mixed_v1/subsets/lane_transition/val.jsonl \
  --output-dir outputs/training_road_mixed_v1/eval_lane_transition
```

总体评估也会按 scene_type 分组。分别看两类 ADE、pose/quality、是否比 Guide 改善以及 Empty BEV 对照。
由于路口感知已重导出，不能把旧 checkpoint 的旧数据成绩直接视为同输入公平对照；
如需比较模型，要对旧模型评估同一份 `subsets/intersection/val.jsonl`。
test.jsonl 暂留，避免用测试集反复调参。
