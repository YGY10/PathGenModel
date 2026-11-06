# 标准化自定义诊断场景

预览：`outputs/custom_standardized/20261105_v2/gallery.html`；总览 `overview.png`。
这组是按用户自定义测试的意图重建的受控样例，不是给原图自动推断GT，也不是四臂真实路口的完整复刻。不替代现有真实场景或验证集。

|目录|连接与导航|Guide问题|
|---|---|---|
|standard_straight|两车道，当前左车道保持直行；nav=straight|逐渐偏右0.8m|
|standard_left_turn|简化双车道左转连接：入口左车道到出口内侧，80m路径内实际左转；nav=left|提前2.5m转弯，出口向内偏2.5m，故意保留路沿冲突|
|standard_right_split|右侧2→3增道，入口右车道自然分出新增右出口；nav=right表示下一路口准备|横移滞后8m，出口偏左0.8m|

车道基准宽3.5m，边缘标线与路沿间额外留0.25m。自车与目标起点为(0,0)，朝向+x；Guide也从(0,0)起步。GT与Guide各81点、按各自弧长1m采样；曲线上的相邻点直线距离略小于1m是正常弦长效应。
分流过渡x12–52m，内部公共无标线区x16–46m，片段不跨空白连接。左转连接路口区域不画内部车道线；绿色走廊是已声明的虚拟连接，不是给模型的感知标线。

完整道路独立保存。模型BEV严格裁剪到训练clean档几何FOV：x∈[-18,85]，y∈[-22,22]（米）。这是确定性的无噪声基准，不宣称等同于完整训练clean感知分布；不额外抽样漏检/断裂，置信度0.99，区域框只在完整可见时保留。Guide和参考路径是规划80m范围，不按感知FOV截短。

每个目录包含：

- `input.json`：可直接infer.py，包含可见BEV、Guide、导航和人工参考`gt_path`。
- `annotation.json`：入口出口车道名、连接、Guide误差、目标走廊、多边形道路范围、标签来源。
- `world.json`：裁剪前几何，用于审计；不要把它直接作为感知输入。
- `scene.json`：编辑器可导入工程，Guide保留完整密集几何，允许继续编辑和重新导出；绿色路径仅供参考，编辑后标签需重审。
- `preview.png`：左图完整道路，右图实际可见BEV；绿色路径和浅绿色走廊都是标注，不输入模型。

参考路径是人工构造的标准答案，不是真实采集GT。模型不接收目标车道ID、走廊多边形或GT；此标注消除的是评判歧义，不是给模型额外Oracle输入。若远侧感知缺失仍不能唯一识别目标，应明确记录，不能认为外部标注自动解决了模型可观测性。

已自动检查：Guide/参考起点、采样长度、目标路径及完整走廊在道路内、所有感知坐标在FOV内、没有元素被容量丢弃、带/不带gt_path的模型特征完全一致、scene.json重新导出的Guide与input逐点一致。报告`validation.json`。只创建样例，不启动模型推理或训练。

重建到新目录：

```bash
python scripts/build_standardized_custom_cases.py \
  --output-dir outputs/custom_standardized/my_new_version
```

例如评估混合增强模型（人工参考ADE仅用于本组诊断）：

```bash
python infer.py \
  --checkpoint outputs/training_road_mixed_guide_aug_v1/best.pt \
  --input outputs/custom_standardized/20261105_v2/standard_left_turn/input.json \
  --device cuda
```

比较单路口模型时只替换checkpoint为`outputs/guide_training_v1/aug/best.pt`，其他输入保持一致。
