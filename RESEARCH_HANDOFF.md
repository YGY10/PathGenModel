# PathGenModel 研究交接

更新：2026-10-25。这是研究状态记录；文件内路径均相对项目根目录，换电脑后以实际文件为准。

## 当前目标和下一步

任务：离散 BEV polylines + 粗 Guide + Navigation，预测未来 80 m、81 个点的 GT path。
研究目标：探索能否把全程验证 ADE 降至 0.05 m。全程指标不取消，观测支持分区和同世界路段误差只作辅助诊断。

目前正在研究 Oracle（目标车道明确、相关物理几何无感知噪声）条件下的精度。完整训练已结束：训练 ADE 0.10514 m、验证 ADE 0.11923 m。下一步尚未执行：

1. 检查 32 点编码后的几何损失和高偏差样本。
2. 分解纵向/横向误差，以及按站点误差与到 GT 折线距离的差异。
3. 按 pose、转向、转弯段/出口段分组，定位误差集中在哪里。
4. 根据证据一次只改一个因素；暂未决定修改模型、输出参数化或 Loss。

不要将当前结果解释为模型精度的理论下限，也不要把 Oracle 与真实感知的差异全部归因于某一个因素。

## 沟通与运行约定

- 用中文，解释方法与证据，区分观察、关联和因果结论。
- 最近用户选择自己执行完整数据生成和训练；给出确实存在的脚本/配置和可执行命令。
- 不因交接而自动启动长训练或重建大数据；用户明确要求的评估可以执行。
- 已有数据、checkpoint、评估结果保留；新实验用新目录。
- 仓库若存在 .codegraph/，按照用户要求先用 CodeGraph 定位代码；此前检查没有该目录。

## 环境

原机器项目：/home/ygy/PathGenModel。
原 Python：/home/ygy/miniconda3/envs/toybev/bin/python；另一台机器需换为其 Python/conda 环境。
主要依赖：Python 3.10+、numpy、torch、shapely、matplotlib、pytest、tensorboard。
配置里的数据路径通常相对项目根目录，命令应在根目录运行。
一些旧 summary.json 记录原机器绝对路径；若脚本依据它们定位文件，换电脑后要核实路径，不能盲目使用。

## 数据及划分

当前常规数据：outputs/v4_world300_inside13。
300 World，18900 样本；7 pose × 3 转向 × 3 感知质量 = 每 World 63 条。
pose：approach_30、approach_20、approach_10、entry、inside_8、inside_13、exit_5。
inside_13 为从 GT entry 沿路线前进 13 m；若已经到达/越过 exit，则跳过。当前 300 World 均生成了该位置。
生成器：Scenario/build_dataset_v4.py；共享 pose/稳定 RNG ID：Scenario/build_dataset.py。
新增 pose RNG ID 为 6，旧 pose 编号没有改变；补充感知脚本也支持它。

划分位于 audit/train.jsonl、val.jsonl、test.jsonl，以 World 为单位，240/30/30 World；不要按行重新拆分。
GT 与 Guide 在同一 route 的不同感知质量下保持一致。

Guide 修过一个重要 bug：原来每个 pose 做 ref -= ref[0] 会抹掉出口常量偏差。
现为同一 guide_world 只截取并刚体变换；Guide 起点允许非零，GT 和预测起点仍固定为零。
约定名 world_fixed_ego_transform_v1。相关脚本 Scenario/reexport_guides.py。
旧数据 outputs/v4_world300_fixedguide 为六 pose 版本，不要与 inside13 版本混淆。

## 当前模型与 Loss

Model/pathgen_model_v1.py：point feature 6 维，polyline attr 10 维，最多 32 条线，每条 32 点。
D=128，4 heads，spatial encoder 2 层，decoder 2 层，dropout=0.1。
Polyline 点编码/池化成 token，Guide 点编码为 query，融合 Navigation，输出 Guide + XY residual。
已有 query—polyline 最近线段相对几何，进入 per-head attention additive bias；不是局部几何 value 的直接聚合。
没有做过坐标归一化改造、Frenet 输出改造或 Residual Gate。
GT/预测共 81 点，间隔 1 m，起点强制为零。
Model/losses.py：位置 SmoothL1 + 0.1 × 二阶差分对 GT 的匹配；位置项排除起点。
停止线/斑马线也由 polyline encoder 处理，类型由 attributes 区分。

## Oracle 实验：现在最重要

说明：README_ORACLE.md。
导出脚本：Scenario/export_oracle_lanes.py。
数据：outputs/oracle_lane_v1。

Oracle 用真实入口/目标出口 lane ID 筛选物理边界，只保留两侧相关边界，隐式明确目标身份。
坐标来自独立 World physical 元素；不把 GT path、GT 中心线或 GT residual 做输入。
保留原粗 Guide、GT、Navigation；无感知噪声、断裂或漏检，ego 四向范围 100 m。
仍保持原模型的 32-token / 32-point 表达。无其他车道候选、停止线或斑马线。
原三种感知质量变成相同输入，因此只保留原 clean 一份并标为 oracle，避免重复三份。
World 划分继承原数据：train 5040、val 630、test 630。

小拟合实验：
- 配置 configs/train_oracle_fit.json。
- 4 个训练 World（84 样本），2 个独立验证 World。
- 1000 epochs，无 early stop，LR 1e-4 保持不变。
- outputs/training_oracle_v1/fit/last.pt 在训练集 eval 模式 ADE 0.0579976 m，FDE 0.0609065 m。
- 训练时 dropout 开启，日志 train ADE 不能直接当作拟合极限；拟合诊断要评估 last.pt 的训练集。

完整 Oracle 实验：
- configs/train_oracle_full.json；最多 300 epochs，50 epochs 无 best 早停。
- 第 138 个 epoch 早停，best.pt 为 epoch 88。
- outputs/training_oracle_v1/full/best.pt。
- eval_train/summary.json：ADE 0.1051433 m，FDE 0.1220106 m。
- eval_val/summary.json：ADE 0.1192326 m，FDE 0.1415428 m，ADE P90 0.2257340 m。
- val 分段 ADE：1–20 m 0.0816188，21–40 m 0.1179912，41–80 m 0.1386602。
- 训练/验证 ADE 差约 1.4 cm，尚不能把剩余问题主要归因于明显过拟合。

已读取 outputs/oracle_lane_v1/export_report.json：32 点编码边界相对原物理折线的最大偏差 0.080504 m，P90 0.035158 m，134 条样本最大偏差超过 5 cm。
这尚未做深入分组，且边界偏差不是 path ADE 的严格下限！这是下一步分析的重要线索。

评估命令示例（输出已存在时需换目录）：

```bash
python evaluate.py --checkpoint outputs/training_oracle_v1/full/best.pt --no-empty-bev --output-dir outputs/training_oracle_v1/full/eval_val_new
python evaluate.py --checkpoint outputs/training_oracle_v1/full/best.pt --split train --no-empty-bev --output-dir outputs/training_oracle_v1/full/eval_train_new
```

## 已做研究与证据

### Mixed 与扩展感知

configs/train_context_control_inside13.json、train_context_mixed_inside13.json。
输出 outputs/context_comparison_inside13/control 和 mixed。
Mixed 按 epoch/sample 确定性抽样：50% 原输入，其他各 12.5% 补至多 4/8/12/全部 retained tokens。
训练 cache：outputs/context_train300_inside13，seeds 1031/1032。
扩展感知不重新生成旧观测，只在原 FOV 外补充，保持质量参数，最多填充 32-token 空位。
验证 cache：outputs/context_comparison_inside13/fixed_val_supplements，seeds 724/725/726。
不得按 GT 选择正式训练/推理的补充 token；Oracle 是明确标记的特权诊断例外。

Inside13 Mixed best 为 epoch 38，第 58 轮早停。全量 val 1890 样本：
- 原始 ADE/FDE：0.597270 / 0.905977 m。
- 三扩展种子均值：0.585358 / 0.885791 m。
- pose_context_evaluation/by_pose.csv 含完整按 pose 原始/扩展结果。
- entry 最难，扩展 ADE 0.9863；inside_8 0.4737；inside_13 0.2954；exit_5 0.1065。
- 单看原六 pose，新 Mixed 原始 ADE 0.6449（上一版 0.6552），扩展 0.6337（上一版 0.6236）；新增容易 pose 会改变整体平均，不能直接归因比较。
- 当时用户说本轮 Control 仍在训练；其最新状态需在新会话重新检查。

上一轮六 pose 公平四条件结果：outputs/context_comparison/evaluation/summary.json。
Control 原始/扩展 ADE 0.6437/0.7591；Mixed 原始/扩展 0.6552/0.6236（三扩展种子均值）。
排除 Guide lane 歧义的 738 条子集：control 0.6331/0.7115；mixed 0.6136/0.5928。
相关目录 outputs/context_comparison/clear_guide_evaluation；没有证明扩展信息对每条样本有益。

### Guide 与车道身份歧义

audit_guide_lane_ambiguity.py，报告 outputs/context_comparison/guide_lane_audit/findings.md。
旧 90 条路线中，61 多车道：44 target_clear，6 near-tie/mixed，11 更靠近其他车道。
在目标明确组仍存在选错车道问题；不能全部归罪于 Guide。
重要案例 World 32019345 left entry：Guide 几乎在两车道分隔线上，GT 与预测各在一个车道。
生成器将入口/出口车道组合打乱，选可行组合，没有固定“左转一定去某出口 lane”的规则。
补入理想出口边界（不显式标注目标身份）只小幅改善稳定反向案例，报告 outputs/context_comparison/exit_boundary_oracle。

### 同一世界出口路段跨 Pose 对照

diagnose_pose_progression.py，outputs/context_comparison_inside13/mixed/pose_progression/findings.md。
270 组相同 World/nav/quality，entry/inside8/inside13 均评价出口后 3–30 m 的相同 28 个世界站点。
原始同段误差 1.261 → 0.559 → 0.327 m；扩展 1.230 → 0.541 → 0.312 m。
不是仅因困难转弯退出移动 80 m 窗口造成的下降，但前瞻距离也缩短了。
左转扩展两侧边界支持 44.8% → 36.2% → 48.0%，误差 1.584 → 0.743 → 0.438 m。
32 条左转三元组在共同窗口内都没有任一侧目标边界支持，仍平均从 1.785 降至 0.503 m。
因此不能把进入路口后的改善只归因于新增出口边界数量；姿态/位置、候选 lane 关联、Guide 截取等共同变化。
支持阈值为 1.25m，基于实际 retained / 32点编码几何与诊断 source IDs，不等于严格信息可观测性。

## 迁移文件

最省事是复制整个项目（含 outputs），再在新机器重建 Python 环境。
若只继续当前 Oracle 研究，至少携带：
- 所有代码、configs、tests、README 文档及本文件。
- outputs/oracle_lane_v1（数据/World/routes/划分/export_report）。
- outputs/training_oracle_v1（checkpoint/config/history/评估结果）。
- 若要继续真实感知对照：outputs/v4_world300_inside13、context_train300_inside13、context_comparison_inside13。
- 若要复核旧歧义结论：outputs/context_comparison 及其引用的数据/预测；可按报告路径补充迁移。

代码和 Markdown 可用 Git 管理；大数据、checkpoint 和未跟踪的 outputs 需确认实际同步，不能仅复制代码。
新会话启动后先核实上述文件是否存在，不应根据本记录假定所有文件已经迁移。
