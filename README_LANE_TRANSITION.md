# 当前预览：城区增减道 + 导航选道（2026-11-04）

## 批量数据与感知输入

600 World正式批次已完成：`outputs/urban_lane_transition_world600_v1`，`validation_report.json` 的 passed=true。
合计45,360行：训练480 World/36,342行，验证60/4,401，测试60/4,617。左/直/右导航各15,120行，三档感知质量各15,120行；28条hard空BEV样本按漏检设定保留。
使用独立seed 44000000，均采用35–50m过渡、内部无标线、原有三档感知参数。
World是物理真值；生成时已按路线/导航、自车pose和clean/medium/hard叠加可见范围裁剪、漏检、片段断裂及位置噪声。
模型训练读取 `audit/train.jsonl` / `val.jsonl` 中感知后的 `lane_markings`、`road_boundaries`，以及Guide、navigation，GT仅监督；不能直接用world物理线作为模型输入。
World隔离480/60/60；当前批次循环覆盖已确认的5个城区增减道模板，尚非全部可能的车道数量和连接形态。
数据生成器改为逐World写盘，5-World的样本及三个划分与改造前逐字节一致。并行批次分4片生成，最后统一按全局World种子划分，分片自己的临时划分不用于训练。
旧串行未完成目录 `outputs/urban_lane_transition_world600_v1_partial_serial` 不能用于训练。

## 2026-11-04 当前规则

用户确认：道路过渡长度在 **35–50m** 内均匀随机；内部无标线区起止位置按过渡长度比例随机，每条线端点稍有差异。
默认标线模式改为 `unmarked`，所有World都有共同的内部空白区，两侧路沿、外侧实线保留。
空白区名义起点为长度的10–20%，名义终点为76–88%；各分隔线端点另加±3.5%长度的扰动。
新预览：`outputs/urban_lane_transition_preview_v4_length35_50`，5 World、378行，未训练。旧输出保留。

```bash
python -m Scenario.build_lane_transition --worlds 5 --seed 43000000 \
  --scene-mode urban --marking-pattern unmarked \
  --perception-config perception_v4.example.json \
  --output-dir outputs/urban_lane_transition_preview_length35_50_new
```

## 2026-11-03 历史更新：内部无标线区

新版预览：`outputs/urban_lane_transition_preview_v3_marking_gaps`，仍为5 World、378行。
上一版 `--marking-pattern mixed` 为默认：前5个World依次为 unmarked / partial / continuous / unmarked / partial。
可指定 `--marking-pattern unmarked` 让所有World都有完整的内部无标线区。

- unmarked：各内部标线在过渡区前部终止、后部恢复，各线端点略有差异，保证一段共同的内部完全空白区。
- partial：保留一条持续分隔线，其余内部线局部缺失。
- continuous：保留上一版的连续标线，作为对照。
- 两侧路沿和道路外侧实线均保留。空白是物理道路中没有标线，与随机感知漏检/断裂分别建模。
- 缺口两侧独立保存为折线，禁止编码时连接穿过空白；同seed改变标线模式不改变GT、Guide、导航目标和道路边界，但感知随机调用变化，不构成同噪声严格配对。

预览浅黄色带标记保证内部完全无标线的公共区域（仅unmarked模式）。`world.json`和样本metadata中记录各线缺口、公共区间及标线模式。

```bash
python -m Scenario.build_lane_transition --worlds 5 --seed 43000000 \
  --scene-mode urban --marking-pattern mixed \
  --perception-config perception_v4.example.json \
  --output-dir outputs/urban_lane_transition_preview_marking_gaps_new
```

CLI 现在默认 `--scene-mode urban`，新增 `make_urban_world`；原 `make_world` 和 `--scene-mode legacy` 保留旧数据生成行为。

```bash
python -m Scenario.build_lane_transition \
  --worlds 5 --seed 43000000 --scene-mode urban \
  --perception-config perception_v4.example.json \
  --output-dir outputs/urban_lane_transition_preview_new
```

已生成 `outputs/urban_lane_transition_preview_v2`：5 World、378 行（多入口 × 三导航 × 三 pose × 三质量），用于形态确认，尚未训练。
`previews/` 每张图三行是 left/straight/right，三列为 before_10/during_mid/after_5，均展示同一入口的 clean 输入；全部质量在 JSONL 中。
`overview.png` 对比每个 World 同一入口、同一自车位置的三导航 GT/Guide。

规则：导航表示后续路口意图，当前路段进行提前选道，并非立即转弯。目标限制为原槽位或相邻槽位内的合法出口，最多调整一个车道；left/right 选择可达范围物理最左/最右目标，straight 选择横向变化最小目标。槽位索引从固定侧边缘向变化侧计数，不能直接把小索引当作左侧。某些导航可得到同一个近期目标。没有一车道内可达出口的入口会被省略，不伪造跨多车道路径。

过渡长度 35–80m，保留持续车道的虚线；新增分隔线在扩宽提供空间后出现，消失分隔线在收窄时终止。GT 允许有意跨越虚线调整到相邻车道，检查道路包络与最大几何斜率；这不是车速相关的动力学可行性验证。Guide 依目标端点独立生成，保持世界固定，具有 ±7m 起止位置扰动、0.3–0.9m 横向误差，出口不改变最近车道身份。尚未实现同路多 Guide 增强。

下文保留旧版形态说明，显式加 `--scene-mode legacy` 才对应这些旧约定。

---

# 增道 / 减道小批量预览

```bash
python -m Scenario.build_lane_transition \
  --worlds 5 --seed 41000000 --scene-mode legacy \
  --guide-shape piecewise_linear \
  --perception-config perception_v4.example.json \
  --output-dir outputs/lane_transition_preview_v4_visible_breaks
```

输出目录必须不存在。当前修订版已生成在上述目录，查看 `previews/` 的五张 PNG。
每张 3×3：行是 clean/medium/hard，列是过渡前 10m `before_10`、
过渡中点 `during_mid`、过渡结束后 5m `after_5`。距离沿 GT 弧长计算。
预览选取横移最大的同一连接，其他连接也全部保存在数据里。
沿用路口画法：横轴 x 向前、纵轴 y 向左，等比例；灰蓝色世界边界、浅金色世界标线、
紫色实际感知、绿色 GT、橙色虚线 Guide；另用蓝色点线标注 FOV。
世界真值只用于对照，模型输入中只有带噪声的感知对象。

前后车道数均为 1–4，数量变化为 1 或 2；过渡长度 15–25m。
首批：1→3 右侧增道、3→1 左侧减道、2→4 左侧增道、
4→2 右侧减道、3→4 右侧增道。生成更大批次时循环覆盖所有合法数量变化和左右侧。
车道宽度 3.3–3.8m，
附带小幅道路横移。单向道路，索引从固定侧道路边缘向变化侧递增。
固定侧车道维持连接；最外侧入口可连接多个新增出口，多个收窄入口可汇入最外侧保留出口。
**过渡区内没有内部车道分隔线**，两侧各有路沿和白色实线。
白色实线在路沿内侧横向 0.25m 处，独立编码为 lane_marking；路沿仍为 road_boundary。
两者覆盖过渡区，感知漏检分别使用 line_miss_rate 和 curb_miss_rate。
所有入口内部标线在过渡起点前终止，
所有出口标线从过渡终点开始，每条折线独立保存，避免编码时连接穿过空白区。
连接使用五次平滑函数，当前只验证完整道路可行驶范围，
并不声称已模拟所有车道级连接规范或真实道路设计标准。

每个 World 生成全部上述连接，每条连接 3 pose × 3 种现有感知质量。
当前 5 World 共 162 条样本。感知参数与 V4 完全一致，不为了预览修改漏检率或噪声。
复用 V4 FOV 裁剪、漏检、断裂、噪声及模型编码，不添加无依据的停止线或斑马线。
`--perception-config` 可指定与 V4 相同格式的配置文件。

共享 V4 observe 已改为 `vcs_visible_segment_breaks_v2`：先在 VCS 裁剪得到可见连续段，
再按原 break_rate 对每条足够长的可见段采样至多一个内部缺口。
缺口宽度为可见弧长的 4–13%（至少 0.8m），两端各留至少 2m 用于形成有效片段；
可见段不超过 5m 时跳过断裂。路口和分流之后生成的数据都采用此实现，已有输出不变。
随机数调用顺序会改变，因此新旧感知结果不应当作逐对象相同噪声的严格配对；
当前批次已验证 GT/Guide 与 v3 完全相同。

GT 与 Guide 分别从连接端点、道路参数生成。GT 使用五次平滑曲线；
默认 Guide 为分段一次折线：入口段、过渡区直连段、出口段，四个世界坐标锚点，
锚点保留入口/出口偏移及少量过渡位置偏移。每段均在最终世界坐标中保持直线，
不对拐角做样条平滑。仍按自身弧长取 81 点；跨拐角的相邻采样点弦长可能略小于 1m。
`--guide-shape smooth` 可使用旧版平滑 Guide，以相同 seed 对照；道路、GT、感知不受此开关影响。
它在 World 中固定，所有 pose 仅截取和刚体变换，保留出口偏差；不重新归零 Guide 起点。
导航统一 `straight`：这里的增道/减道不等于交叉口左转/右转，目标连接由 Guide 提供。

输出：

- `samples.jsonl`：现有 `PathJSONLDataset` 可直接读取的训练格式，GT/Guide 均为 81 点、1m 采样。
- `worlds/<world_id>/world.json`：独立道路物理几何和生成参数。
- `worlds/<world_id>/routes.json`：全部连接及固定的 GT/Guide 世界折线。
- `audit/train.jsonl`、`val.jsonl`、`test.jsonl`：按 World 隔离，5 World 分成 3/1/1。
- `generation_report.json`：参数、质量配置、连接数及划分。

这五个 World 用于检查场景理解与几何预览，不足以评估泛化或开始正式混合训练。
确认生成形态后再扩展数量与拓扑，建立独立的路口/增减道验证子集和混合训练配置。

批量生成与联合训练命令已准备在 [README_MIXED_ROADS.md](README_MIXED_ROADS.md)。
