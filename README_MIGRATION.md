# 在另一台电脑继续研究

当前研究状态以 [RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md) 为准。
重点是 Oracle 全量模型：训练 ADE 0.1051 m、验证 ADE 0.1192 m；下一步先分析
32 点几何表达及纵向/横向误差，再决定是否改模型或 Loss。

快照已在独立空目录实际恢复并复核：630 条 Oracle 验证样本的 CPU ADE 为
0.119244 m。详细校验记录见 [MIGRATION_VERIFICATION.json](MIGRATION_VERIFICATION.json)。

## 1. 克隆与恢复

```bash
git clone https://github.com/YGY10/PathGenModel.git
cd PathGenModel
python scripts/restore_research.py
```

恢复脚本仅需 Python 3.10+ 标准库，校验分卷、归档和逐文件 SHA-256。
使用一个新克隆运行；存在目标输出时会停止，避免覆盖研究结果。
恢复通常需要数 GB 可用空间。请勿只下载代码而遗漏 `research_artifacts/`。

## 2. 环境

```bash
conda create -n toybev python=3.10
conda activate toybev
pip install -r requirements.txt
python -m pytest -q tests/test_model.py tests/test_losses.py tests/test_pose_progression.py
```

`torch` 请按新电脑 GPU/驱动选择适合的官方安装包；CPU 也能评估。
requirements 对 torch 保留兼容范围，原机器精确版本记录在快照 manifest 中：
torch 2.13.0+cu132、numpy 1.26.4、shapely 2.0.7、matplotlib 3.10.9。
更换设备、PyTorch 或 CUDA 后，浮点结果可能有细微差异。

## 3. 直接复核 Oracle

```bash
python evaluate.py \
  --checkpoint outputs/training_oracle_v1/full/best.pt \
  --device cpu --no-empty-bev \
  --output-dir outputs/training_oracle_v1/full/eval_val_new_machine
```

期望验证 ADE 约 0.1192 m、FDE 约 0.1415 m。旧评估保留在 eval_val/，不要覆盖。
配置路径相对项目根目录，运行命令前应进入该目录。
部分历史 JSON 报告写有旧绝对路径，恢复时会将报告路径迁到新根目录；
数据 JSONL、checkpoint、World/routes 和训练缓存 manifest 不会被重写；
固定验证补充缓存 manifest 的 source_path 会更新到新目录，其文件指纹因此会变化。

## 上传内容与省略内容

上传：
- 全部 Python 源码、测试、配置、文档、研究交接与依赖说明。
- 当前 Oracle 数据的精确 train/val/test 和小拟合划分、World/routes、导出报告。
- Oracle fit/full 的 checkpoint、训练历史、评估预测、图片和 TensorBoard 记录。
- inside13 真实感知数据的精确划分、当前 control/mixed 输出与固定扩展验证缓存。
- 小型历史研究报告和图片，保留在原有 outputs 路径下。

省略：
- 可从精确划分重建的重复 samples.jsonl；恢复时会重建其集合，顺序为 train/val/test。
- 多份旧版数据集、旧 checkpoint、Python/IDE 缓存。
- 两份大型随机训练补充缓存；需要继续 mixed 训练时重新生成：

```bash
python -m Scenario.build_fov_supplements \
  --input outputs/v4_world300_inside13/audit/train.jsonl \
  --worlds-dir outputs/v4_world300_inside13/worlds \
  --output-dir outputs/context_train300_inside13 --workers 4
```

旧 Mixed checkpoint 的严格续训会验证原缓存指纹；重建后若指纹不同，不能把它当作完全相同状态的无缝续训。
做新对照时从头训练更容易控制条件。Oracle 无扩展训练缓存依赖，可直接开展当前研究。

GitHub 中保存的是普通二进制分卷，不需要 Git LFS。每卷不超过 8 MiB，
`research_artifacts/manifest.json` 精确列出全部归档文件、原路径、大小和哈希。
数据与 checkpoint 的原始文件仍保留在原电脑，打包不会删除它们。

## 给新会话的启动消息

> 请先阅读 RESEARCH_HANDOFF.md、README_MIGRATION.md 和 README_ORACLE.md，
> 核实恢复后的 Oracle 数据及 full/best.pt。继续研究如何将验证 ADE 从约
> 0.1192 m 降到 0.05 m；先检查 32 点编码几何损失及纵横向误差，暂不启动长训练。
