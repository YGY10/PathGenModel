# 同一道路的多种模拟感知：人工审核预览

入口 `outputs/perception_review/20261105_v1/gallery.html`，总览 `overview.png`。
本批只用于视觉审核，不用于训练，不运行模型，不改变正在进行的训练或现有感知器。

四个固定场景：

- A：路口左转，进入前5m，World seed 30000000。
- B：路口右转，进入前5m，World seed 30007919。
- C：右侧2→3增道，过渡前8m，World seed 44000000。
- D：左侧3→4增道，过渡前8m，World seed 44000004。

增道场景不含减道；保留物理过渡区的内部无标线区域。场景沿用现有World构建函数，主要审查观测假设，不声称本次已经解决道路几何覆盖。

每例包括物理真值、无噪声观测基准、轻/中/较强各A/B两次独立抽样，共7个实际输入、8个预览面板。A/B按固定seed直接生成，不挑选、不重抽，质量档位的单次样本不保证严格单调缺失。

主对照固定FOV为前85m、后18m、左右22m，仅采用当前perception_v4.example.json的漏检/断裂/噪声参数。固定自车、道路、导航、Guide及GT；不是把原训练三档FOV混在一起比较。range_comparison.png另外只比较当前85/22、65/15、45/9m的视野差异，关闭随机几何扰动和漏检。

图示：蓝=车道线，深灰=路沿，红=停止线，青=斑马线，绿=目标参考，橙虚线=Guide。浅灰虚线为真值背景，不是感知输入。浅黄是道路本来缺内部标线的范围，不是感知漏检，也不表示路沿缺失。

观测图绘制实际encode_input保留/重采样后的几何（最多32个元素、每个32点）；本批没有容量丢弃。基准仍遵循现有感知器的最小可见段长度/框估计规则，不等于完全不做处理的物理World。名义FOV裁剪后会加噪声，所以少量坐标可越过框边，不额外修改训练感知器进行再次裁剪。

现有模拟器的限制也供审核：每个可见连续线段最多添加一个内部缺口；线条使用片段平移与点噪声，角度扰动用于区域框估计，不是对所有车道线旋转。当前范围和噪声是人工设定，尚未由真实数据标定。

每例comparison.png为整组，detail.png为基准/轻A/中A/强A局部放大。各quality_draw.json可直接infer.py；world.json、truth_ego.json、context.json保存完整审计上下文，manifest.json保存参数、seed、统计及不变量验证。

## 如何反馈

打开gallery.html，每张可选择“常见 / 偶尔有 / 不合理”，填写例如“近处路沿不应整段消失”“远处标线应该更碎”“中度横向漂移太大”。点击“下载我的反馈 JSON”得到perception_review_feedback.json。
反馈只存本浏览器，不自动上传；浏览器不允许本地存储时仍可下载。也可直接在对话中按“A中度B / C较强A”描述，不必给出概率或方差。

## 复现到新目录

```bash
conda activate torch310
python scripts/build_perception_review.py \
  --output-dir outputs/perception_review/my_new_preview \
  --seed 551105
```

输出目录须尚不存在，以保留旧预览。生成只用CPU，不读取模型权重。
