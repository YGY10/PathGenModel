# 自定义输入推理

在项目根目录、激活 toybev 环境后运行：

```bash
python infer.py \
  --checkpoint outputs/context_comparison_inside13/mixed/best.pt \
  --input examples/custom_input.json
```

输出在新的 `outputs/custom_inference/<时间>/`：

- `prediction.json`：81 个预测点、相对 Guide 的残差、输入/保留元素数量。
- `prediction.png`：实际编码后进入模型的 BEV、Guide、Prediction；提供 GT 时也绘制 GT。
- `input.json`：本次使用的输入，便于复现。

也可指定 `--device cpu`、`--output-dir outputs/my_custom_case`。输出目录必须尚不存在。
脚本仅做推理，不修改 checkpoint、训练数据或原始输入。

## 修改示例 JSON

坐标单位为米，采用统一的自车坐标系：原点为自车当前位置，**x 向前，y 向左**。
例如 `[20, -3]` 表示前方 20 m、右侧 3 m。图片横轴取 `-y`，纵轴取 `x`。
输入中不应包含世界坐标、像素坐标或未经转换的其它坐标系。

| 字段 | 格式 |
| --- | --- |
| `guide`（或 `reference_line`） | 81 个 `[x,y]`，通常按前方路径每 1 m 一个点，共 80 m |
| `navigation`（或 `nav`） | `straight` / `left` / `right` |
| `lane_markings` | 车道标线对象列表 |
| `road_boundaries` | 路沿/道路边界对象列表 |
| `stop_lines` | 停止线矩形或多边形对象列表 |
| `crosswalks` | 斑马线覆盖区域的矩形或多边形对象列表 |
| `gt_path`（可选） | 81 个 GT 点，仅用于画图和计算排除起点的 ADE、末点 FDE，不作为模型输入 |

四类道路元素均可省略或设为 `[]`，包括完全空的 BEV。
每个对象用 `points` 给出按几何连接顺序排列的点；不是将所有元素的点混为一组：

```json
{"points": [[0, 1.75], [30, 1.75], [80, 1.75]],
 "color": "white", "line_style": "dashed", "confidence": 1.0}
```

可选属性：`confidence` 为 0–1（默认 1）、`color` 为 `white` / `yellow`、
`line_style` 为 `solid` / `dashed`。缺省颜色或线型对应编码中的未知属性。
开放折线至少两个点；停止线/斑马线按训练表示使用闭合多边形，至少三个不同顶点，
通常填写矩形四个角，沿周长顺序排列。**停止线不能直接只填两个端点**，应提供实际宽度的矩形。
所有点和 Guide 必须位于同一坐标系；Guide 起点允许偏离原点，模型预测起点固定为 `[0,0]`。

Guide 若只有少量折点，可以加 `--resample-guide`：按弧长取前 80 m、每 1 m 一个点。
折线总长不足 80 m 时会报错，不自动拉伸或外推。默认不重新采样已给定的 81 点 Guide。

编码完全复用 `Dataset.dataset.encode_sample`：按 checkpoint 配置限制容量，
常见为最多 32 个元素，每个重采样成 32 点。超限时沿用训练的筛选规则：
停止线/斑马线优先占位，再按距离保留开放折线。脚本报告实际保留数量，图片显示实际编码输入。

## 使用现有数据直接试跑

```bash
python infer.py \
  --checkpoint outputs/context_comparison_inside13/mixed/best.pt \
  --input outputs/v4_world300_inside13/audit/val.jsonl \
  --index 0
```

`--index` 从 0 开始。这个方式会同时绘制 GT，便于与自定义输入比较。
checkpoint 为参数，因此也能换成 control 或 Oracle 模型。一般场景建议先用 mixed/control；
Oracle 模型训练时仅见过筛选出的理想入口/目标出口边界，其在任意道路元素输入下的结果
不能代表相同的精度条件。随意输入可以检验运行及响应，定量精度仍需有 GT 的合理场景。
