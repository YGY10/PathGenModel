# 新版城区分流与原路口联合训练

已确认数据：路口 outputs/v4_world300_inside13（含原Guide越界/路沿冲突），分流 outputs/urban_lane_transition_world600_v1（600 World、45360感知样本，validation_report passed=true）。
本轮v2为实验编号，模型仍是 Model/pathgen_model_v1.py 的XY residual及原Loss。配置 configs/train_road_mixed_v2.json，seed42，从头训练；不续训、不加载旧Guide增强缓存、不启用扩展FOV。

1. 在项目根目录，使用已安装依赖的Python环境，先混合数据：

```bash
python -m Scenario.prepare_mixed_dataset \
  --source intersection=outputs/v4_world300_inside13 \
  --source lane_transition=outputs/urban_lane_transition_world600_v1 \
  --refresh-perception intersection \
  --perception-config perception_v4.example.json \
  --seed 7301 \
  --output-dir outputs/road_mixed_v2
```

重导出路口感知，统一可见段内断裂约定，保持GT、Guide、自车pose和train/val/test的World归属。
分流直接使用已叠加的三档感知，不再次叠加噪声；World物理线不直接作为模型输入。
保留两个来源目录：混合数据的元数据仍引用原World路径。
原路口样本包含Guide冲突，但此次不增加Guide变体；旧Guide增强缓存的指纹不适用于混合数据。

预计划分：

| split | 路口 | 分流 | 合计 |
|---|---:|---:|---:|
| train |15120|36342|51462|
| val |1890|4401|6291|
| test |1890|4617|6507|

训练按自然样本比例约路口29.4%、分流70.6%，不额外重复采样。最终以mix_manifest.json为准。
本轮只循环采用已确认的五类分流模板，不能将600World解释为全部城区道路拓扑覆盖。

2. 训练：

```bash
python train.py --config configs/train_road_mixed_v2.json --no-tensorboard
```

最多150轮，25轮无best早停；按混合验证ADE选择best，并单独记录两场景验证ADE。输出 outputs/training_road_mixed_v2。
当前torch310环境缺TensorBoard，因此关闭其日志，history.jsonl仍保存指标。安装有TensorBoard时可移除--no-tensorboard。

3. 训练后评估（默认保留EmptyBEV对照）：

```bash
python evaluate.py \
  --checkpoint outputs/training_road_mixed_v2/best.pt \
  --output-dir outputs/training_road_mixed_v2/eval_val
```

关注分场景与分pose指标和EmptyBEV对照，不只看混合ADE。路口感知已重导出，旧模型应在同份新路口子集评估才能公平比较。
上述新混合目录、训练目录和评估目录必须尚不存在；已有输出请换新目录并同步修改配置，避免覆盖。
