# 固定道路、多 Guide 冲突专项评测

本实验不训练模型、不重建World、不修改原始数据。以现有验证World为单位，固定道路、GT、目标车道、导航、pose与全部感知元素，只改变Guide。用于衡量既有模型在Guide与道路冲突时的误差和几何可行性。测试集保持未使用。

## 范围与干预

默认选取验证集30个World，左右转，approach_10 / entry / inside_8，clean / medium / hard，共540条原始输入。每条设置13种Guide条件，最多7020条案例：

- original：直接复用原输入81点Guide，作为成对比较基准；不表示Guide正确、无偏或与GT重合。
- timing：-6/-3/+3/+6m。沿入口行驶方向平移Guide转弯附近，入弯前30..5m平滑升至完整位移，出口后5..35m平滑恢复；远处入口和出口保持原样。负值表示转弯附近向自车方向移动，正值表示向前移动。这里是变形前地图中的空间位移，不是严格的时间或弧长变化。
- lateral：-3/-1.5/+1.5/+3m。沿原Guide局部左法向整体偏移，然后应用原World的公共形变；正值为Guide左侧。不同转向下，该方向不等于统一的向路口内侧/外侧。
- joint：timing=±6m与lateral=±3m的四种组合。

每个World/nav的Guide首先在世界地图中生成，再在各pose截取81个1m弧长站点并刚体变换到自车坐标；不平移Guide使其首点为零。GT始终为原始未来80m站点，预测首点仍固定为零。

重建原Guide仅使用原World参数、入口/出口lane身份及原Guide参数，调用原生成器同一几何函数；不使用GT路径点构造Guide。GT、物理道路与车道面仅用于诊断指标。非原始变体在固定World地图上合成，含刻意超出训练范围的情况；不能把该集合平均ADE当成原验证ADE的新版本。

若完整Guide变体出现自交，显式拒绝并记录；不得根据预测好坏过滤。不同World可能保留不同变体数，比较时提供匹配原输入的成对增量。完整道路用于评价，模型仍只接收原始FOV内的感知输入，不额外添加真值几何。

## 运行

在具有numpy、torch、shapely>=2与matplotlib的项目环境中：

```bash
python -m pytest -q tests/test_guide_conflict_benchmark.py
python scripts/evaluate_guide_conflicts.py \
  --limit-worlds 2 --qualities clean --poses entry \
  --device cpu --output-dir outputs/guide_conflict_benchmark/smoke_new
python scripts/evaluate_guide_conflicts.py \
  --checkpoint outputs/context_comparison_inside13/mixed/best.pt \
  --device cpu --output-dir outputs/guide_conflict_benchmark/val30_mixed_new
```

输出目录必须不存在。默认batch-size=32，CPU线程4；默认device=auto。
原输入预测会与已保存Mixed原始验证预测核对；`--baseline-predictions`须与checkpoint匹配。此处脚本针对该Mixed基线，替换权重时必须先生成对应原始预测档案。

## 指标口径

- 全程ADE/FDE：按原GT等站点标签，ADE排除固定起点。
- 转弯窗口ADE：原GT世界站点entry-5m至exit+3m。
- 路沿穿越：整条预测折线与独立World物理路沿crosses，不只检查81个点。相切或贴线不计穿越。
- 明显出道路：中心折线在重建道路8cm缓冲区之外的长度超过0.25m；保留该长度原值。
- 最大侵入深度：到未加缓冲的真实道路区域的最大外部距离；沿预测折线每0.1m采样，最大可能低估不超过0.05m。只在道路外计算，不把道路内到边界的距离算作侵入。
- 可见路沿穿越：针对真正保留并编码为32点的感知路沿。原始版本包含所有片段；GT-clear版本排除同时与GT相交的片段，以免把噪声观测当成GT也应遵守的障碍。报告Guide与可见路沿冲突组的修复率。
- 出口横向误差：相同World路线出口后3..30m的共同GT站点，以GT局部法向分解预测误差，至少5个可用站点。各Guide条件查询站点完全相同。
- 出口车道偏离：出口后3m起的预测中心线超出目标车道8cm缓冲区域的长度超过0.25m。它不是道路路沿碰撞，单独统计。
- Guide稳定性：同一base输入不同Guide下，出口平均有符号横向误差的max-min范围。它不是单个站点的最大跨度。另报告相对原Guide的预测位移与ADE增量。

冲突严重程度按Guide最大侵入深度分组：<=8cm、8cm..0.5m、0.5..1m、1..2m、>2m。固定输入意味着GT始终相同，但并不意味着任何错误Guide仍包含足以唯一识别目标车道的信息。

## 输出

- summary.json：配置、权重哈希、分组结果、成对增量、数据与复现检查。
- per_case.csv：逐案例指标与Guide参数；per_base_stability.csv：固定原输入的Guide间出口跨度。
- cases.jsonl：原样本ID、Guide数值、参数及非Guide内容哈希；原道路输入保留在原验证JSONL中，避免重复数百MB道路几何。
- predictions.npz：与case_id一一对应的预测。
- variant_metrics.png、examples.png：分条件指标与成功/失败案例。
- examples/*.json：少量完整、可直接用infer.py重放的输入，index.json包含对应指标。

```bash
python infer.py --checkpoint outputs/context_comparison_inside13/mixed/best.pt \
  --input outputs/guide_conflict_benchmark/<本次目录>/examples/<某案例>.json
```

重复pose、质量、Guide不是独立World，不能据7020条案例宣称7020条独立道路。当前评价仅为中心线，不包含车身宽度/车辆运动学，也没有将路沿约束加入模型。
