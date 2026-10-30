# PathGenModel 研究交接

## 2026-10-30：联合模型 best 评估与分流过渡段诊断

用户已运行 `outputs/training_road_mixed_v1/eval_best_val`。总ADE0.360673m、EmptyBEV0.650670m。
路口正常/EmptyBEV ADE=0.593469/1.13216m；分流=0.107372/0.12677m。
路口entry ADE0.99896m、approach10 0.87494m；分流before10/during/after全程ADE=0.11018/0.13995/0.07199m。
新增 transition_region_summary.json 和 transition_region_per_sample.csv：按GT世界x落在[0,length_m]筛选过渡段，排除起点，
对有可评价点的样本等权。before10过渡段正常/EmptyBEV/Guide ADE=0.204670/0.262075/0.478282m；
during_mid=0.130967/0.122518/0.521814m，每组579样本。after5未来不含过渡段。
分流Guide自身可支持大部分修正，清空BEV仍有较高精度；不能将0.107m解释为已充分学习车道几何。
各场景最差样本另存 diagnostic_worst_intersection.png、diagnostic_worst_lane_transition.png。

## 2026-10-30：联合训练首轮已完成

已核实 `outputs/training_road_mixed_v1/history.jsonl`：epoch69早停，best为epoch44。
best混合验证3627样本（路口1890、分流1737），ADE/FDE=0.360673/0.511606m。
路口ADE/FDE=0.593469/0.889683m，Guide ADE=1.460471m；
分流ADE/FDE=0.107372/0.100228m，Guide ADE=0.496791m。
路口straight/left/right ADE=0.462179/0.702033/0.616197m，41–80m ADE=0.846790m。
最后epoch69路口ADE0.613956m、分流0.100197m，整体未刷新best，不建议直接延长。
本次尚未运行 evaluate.py，目录下未发现评估summary。下一步应评估best，并检查分流before/during/after及过渡段局部误差。
混合平均不能代表路口也达到0.36m；不能与旧输入上的旧成绩直接比较混合收益。

## 2026-10-30：准备批量分流 + 路口联合训练

新增 `Scenario.prepare_mixed_dataset`（重复 --source name=目录），保留来源 train/val/test World 归属，
拒绝重复 ID / World 冲突；可用 --refresh-perception intersection 从原物理 World 重导出路口感知，
统一 VCS 可见段内断裂，但保留 GT/Guide/pose/划分。旧支持诊断会清空。
新增 `configs/train_road_mixed_v1.json`，准备600分流 World（预计17280样本）+300已有路口（18900样本），
新数据输出 `outputs/road_mixed_v1`，新训练 `outputs/training_road_mixed_v1`。
架构与 Loss 沿用 V1，标准 FOV，无额外 FOV augmentation，从头训练；最多150轮、25轮无best早停、TensorBoard。
train.py 新增可选 validation_groups，每轮单独记录两类验证 ADE；选择仍按混合 val ADE。
evaluate.py 新增 scene_type 分组。生成器新增 --preview-worlds（默认5），避免批量画图。
完整执行命令见 `README_MIXED_ROADS.md`。完整600 World生成和正式训练尚未启动，由用户自行运行。
小样本流水线已实际跑通：3个分流World+6条真实路口样本，96条均可编码；37项相关测试通过。
批量生成允许全部漏检的空BEV样本，避免低质量场景在大批量时因非空断言中断。

## 2026-10-30：路沿+边线、可见段内感知断裂

用户澄清最外侧同时有路沿与车道线。生成器保留两侧 curb，并在各自内侧横向 0.25m
增加白色实线 lane_marking，过渡区仍无内部车道分隔线。
共享 `Scenario.build_dataset_v4.observe` 现先按 VCS 裁剪连续段，再在段内按原 break_rate
采样缺口；标记 `perception_convention=vcs_visible_segment_breaks_v2`。这也影响后续路口数据生成，
旧数据未重写。新批次 `outputs/lane_transition_preview_v4_visible_breaks`：5 World、162 样本，
GT/Guide 与 v3 逐条一致，感知 RNG 调用变化，不作为同噪声严格配对。
相关回归测试 35 项通过（含可见段断裂、VCS 旋转、固定 Guide、扩展感知及多进程训练流程）。

## 2026-10-30：分流 Guide 改为分段一次

用户要求降低分流 Guide 的平滑程度。`build_lane_transition.py` 新增
`--guide-shape piecewise_linear|smooth`，默认分段一次：入口/过渡直连/出口三段、四个世界锚点。
保留偏移和过渡位置扰动，在世界坐标固定后截取，GT/道路/感知不随该模式改变。
新预览 `outputs/lane_transition_preview_v3_linear`，5 World、162 样本；旧 v2 保留。
生成时显式读取 `perception_v4.example.json`。尚未训练。

## 2026-10-29 新增：增道/减道预览数据

用户确认场景为沿行驶方向从 x 条车道变成 x±n 条，而不限定为两条道路分叉。
新增 `Scenario/build_lane_transition.py`，说明 `README_LANE_TRANSITION.md`。
用户指出初版形态不符，已修订并生成 `outputs/lane_transition_preview_v2`：5 World、162 样本。
车道数前后均 1–4，变化 1–2；过渡长度 15–25m；过渡区内部无标线，只有道路边界。
每条连接 3 pose（before_10、during_mid、after_5）× clean/medium/hard，复用 V4 感知参数与函数。
格式兼容现有模型编码，固定 World Guide，nav=straight；World 隔离 3/1/1，未启动训练。
预览在 `previews/`，3×3 行质量列 pose，使用路口相同坐标与真值/感知配色，额外画 FOV。
当前为单侧增减道、平滑连接、GT 仅验证道路包络；
不代表已完成所有车道连接规则。先与用户确认形态，再扩充和制定路口+增减道混合训练。
`tests/test_lane_transition.py` 21 项测试通过，检查所有合法数量组合、无标线区域、输入编码、采样、固定 Guide 和确定性。
旧 `outputs/lane_transition_preview_v1` 保留，但不作为当前方案使用。

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

## 2026-10-26：自定义冲突测试后的数据覆盖审计

新报告：outputs/guide_boundary_audit/20261026_132823/findings.md。
脚本：scripts/audit_guide_boundary_conflicts.py。仅分析现有train/val与保存的Mixed预测，没有启动训练。
训练去quality重复后的转弯窗口：右转374/1440个pose输入Guide明显出物理道路（25.97%，来自63条独立路线）；不是完全缺乏冲突。
验证全80m三quality：Guide出道路255条，预测仍出道路19条；右转150条中5条仍出道路。clean右转转弯窗口47条冲突，预测明显出道路0条。
因此原分布上多数能修正；自定义失败不能解释成模型普遍不会规避。冲突/非冲突验证ADE约0.598/0.597m，尚不能把整体误差主要归因于冲突。
每World/nav仅一条Guide，Mixed只补tokens，没有同道路多Guide或后方元素干预的一致性训练。
同世界出口后3–30m跨entry/inside8/inside13：entry冲突39组三元组（13 Worlds）平均跨pose横向max-min为0.815m；信息/姿态/噪声同时变化，不能当作单因素因果结果。原Oracle精度诊断目标仍未完成。

## 2026-10-26：已执行固定道路、多Guide专项评测（第一步）

说明：README_GUIDE_CONFLICT_BENCHMARK.md；脚本scripts/evaluate_guide_conflicts.py；完整报告outputs/guide_conflict_benchmark/val30_mixed_v1_20261026/findings.md。
30验证World/60左右转路线，3pose×3quality=540基础输入，每个13种Guide；排除54条自交Guide案例后有效6966条。道路/GT/感知输入固定，世界Guide变形后跨pose截取；未训练、未重建World，test未用。
Mixed专项基准原始Guide ADE0.891m、明显出道路3.33%；Guide转弯附近提前/延后3m后ADE1.766/1.975m、出道路20.74%/21.48%。这是偏困难子集与合成干预，不能替换全验证0.597m。
可见编码路沿冲突子集：提前3m为41/141仍穿越，延后3m为19/62；clean输入也显著退化。支持Guide误差泛化/几何约束不稳，不能单独认定网络或Loss原因。
仅原始与±3m三Guide，出口平均有符号横向误差跨Guide范围均值1.668m。更极端干预含车道身份歧义，须结合报告解释。
6项测试通过；原始预测复现最大坐标差7.63e-6m；GT无出道路、非Guide哈希一致、train/val World无交集。配方、预测、逐例指标与可infer重放示例均已保存。尚未执行第二步训练。

## 2026-10-26：下一轮训练入口已准备，尚未启动正式训练

README_GUIDE_TRAINING.md给出完整命令。用户自行运行：先python scripts/build_training_guide_cache.py，再python train.py --config configs/train_guide_aug_v1.json --no-tensorboard。
新模块Dataset/guide_augmentation.py：左右转50%原Guide、50%从每World/nav八条连续变体采样，直行不变；每epoch仍15120条，不增训练步数。只替换Guide张量，BEV/GT/nav不变；跨pose共享世界Guide。缓存仅使用train240Worlds，输出outputs/guide_train_cache_v1。
新增TrainConfig.guide_augmentation，缓存指纹进入checkpoint splits，验证不增强；与FOV增强互斥，保持单因素。既有训练配置默认行为保持不变。
configs/train_guide_aug_v1.json从随机seed42开始，模型/Loss/优化/最大100epoch与原Control一致，输出outputs/guide_training_v1/aug；严格重跑Control配置train_guide_control_v1.json输出独立control目录。旧Mixed不作为无缝续训起点；主要单因素对照是Control，旧Mixed只作参考。
训练后先evaluate.py导出新模型原验证predictions.npz，再向scripts/evaluate_guide_conflicts.py传新checkpoint和对应--baseline-predictions，复用固定压力测试条件。不能用旧Mixed的baseline档案验证新权重。
相关测试15通过、1项因缺TensorBoard跳过；双worker与精确续训测试在允许本地IPC的执行环境通过。没有生成正式完整缓存或启动完整训练。

## 2026-10-26：Guide增强第一轮训练已由用户完成并复核

增强outputs/guide_training_v1/aug/best.pt为epoch78，第98轮早停。原验证ADE0.624594m/FDE0.946486m。新增同条件Control评估outputs/guide_training_v1/control_reference_eval_val_20261026（ADE0.607839m），及Control压力评测outputs/guide_conflict_benchmark/control_reference_v1。
比较报告outputs/guide_training_v1/comparison_20261026/findings.md；两组初始模型SHA相同，train/val指纹及主要超参数相同，FOV增强都为null，仅Guide增强、日志开关和输出目录不同。
固定6966压力案例：Control/增强 ADE3.0170/1.3283m，明显出道路32.29%/13.71%，可见路沿穿越15.50%/1.92%，可见冲突修复57.85%/96.66%。常规ADE退化约1.68cm（2.76%）；exit_5由0.0838到0.1724m，是主要净退化贡献，inside8/13亦退化。右转整体亦退化，不能仅引用整体approach10改善。
自定义81.85m Guide、固定停止线/斑马线、同时延长右侧车道线/路沿：路沿终点7/10/13/16/19m，Control后三次穿越，增强五次无穿越。结果在outputs/guide_training_v1/custom_curb_followup_20261026；增强有先左偏再右转绕行，无GT不能认定正确路径。
待验证下一轮：仅在approach30/20/10及entry启用Guide增强，inside8/13/exit5保留原Guide，其余条件不改，以检验出口精度恢复。该方案尚未实现/训练。Oracle精度诊断仍待执行。

## 2026-10-27：Δs/Δl与纯横向输出诊断完成，未改模型

报告outputs/lateral_output_diagnostic/20261027_v1/findings.md；脚本scripts/diagnose_lateral_output.py。原训练5040/验证630唯一几何pose，核验真实感知全部15120/1890条GT/Guide相同。
保持现有同序号输出p_i=Guide_i+Δl_i*n_i时，GT理想Δl的全80m ADE下限=train0.6761m、val0.5889m；val左/右/直行0.8647/0.7626/0.1394m。故不能直接删切向自由度达到5cm。原Oracle0.11923m，删去其预测切向分量后ADE0.61269m。
val mean abs GT所需Δs/Δl=0.5889/1.1945m；纵向占残差平方和31.33%。增强训练池纯横向下限2.0735m，当前训练混合期望1.0581m。
重要区别：GT理想横向修正曲线与GT的双向最近折线平均距离仅0.02525m，沿程对应是重要因素。投影曲线长度足80m的337/630子集上，按弧长重采样ADE从0.7207降到约0.05756m；另293条长度不足，不能当全验证性能或理论最优。
建议当前逐点任务保留Δs；l(s)+重采样是值得单独研究的不同方案，需先解决参考曲线覆盖与起点等。5项诊断测试通过，未训练/未改变任何模型、Loss或策略。
