# 下一轮：固定道路、多 Guide 的数据对照训练

先生成训练Guide缓存，再从头训练增强组。已生成的验证压力测试输出仅用于评测，不作为训练数据。当前模型/Loss保持原样，新增因素只有训练Guide采样。

## 实验设置

- 沿用原train 240 Worlds / val 30 Worlds，World划分不变，test不使用。
- 原数据JSONL、BEV元素、GT与导航均保留。训练缓存只保存Guide数值，不复制完整场景。
- 每条训练World/nav（左右转）生成8条连续参数Guide：2条仅转弯位置偏移、2条仅法向偏移、4条联合偏移。每条世界Guide跨所有pose一致截取。
- timing幅度1..6m，lateral幅度0.5..3m；正负号与专项评测一致。自交Guide重抽并记录次数。
- 左右转样本每次以50%概率保留原Guide，50%概率从8条中选择；直行保持原样。因此所有方向合计的增强比例约1/3。每个epoch仍为15120条基础输入，更新步数不因缓存扩大。
- 采样由seed/epoch/index确定，与DataLoader worker顺序无关；验证只用原始Guide。
- 这一轮两个配置都不使用FOV补充增强，避免同时改两个因素。主要对照为Control，不将与Mixed之间的差异全归因于Guide增强。
- 两组从同一随机seed重新初始化，原模型、Loss、LR=1e-4、batch16、最多100epoch、原验证ADE选best、20epoch无改善早停。

已有outputs/context_comparison_inside13/control可作首轮参考（最后epoch64，best验证ADE约0.60784m）。严格配对复跑配置为configs/train_guide_control_v1.json，与已有Control超参数一致、只改输出目录。增强组为configs/train_guide_aug_v1.json，另启用Guide缓存。两组起始随机参数与最大预算一致；早停时实际步数可能不同，比较时记录epoch。

## 在项目根目录执行

使用具有PyTorch等依赖的环境。本机可用conda环境torch310；当前环境缺少TensorBoard，下面关闭该日志，history.jsonl仍保存全部指标。

```bash
conda activate torch310
cd /mnt/workspace/PathGenModel

# 1. 只生成训练Guide缓存，不生成World、不训练。
python scripts/build_training_guide_cache.py \
  --output-dir outputs/guide_train_cache_v1

# 2. 增强组，从头训练。
python train.py --config configs/train_guide_aug_v1.json --no-tensorboard
```

不要将改变Guide策略的实验当作旧Mixed的无缝续训。数据与配置不同，旧--resume的完整状态恢复不适用于启动本轮。

如果需要严格配对Control，单独执行：

```bash
python train.py --config configs/train_guide_control_v1.json --no-tensorboard
```

不要求同时运行两个GPU任务；可以串行。若已有TensorBoard，可移除--no-tensorboard。

## 训练完成后

先评估原1890条验证输入，得到与新权重对应的原始预测档案：

```bash
python evaluate.py \
  --checkpoint outputs/guide_training_v1/aug/best.pt \
  --no-empty-bev \
  --output-dir outputs/guide_training_v1/aug/eval_val
```

再运行完全相同的Guide冲突条件：

```bash
python scripts/evaluate_guide_conflicts.py \
  --checkpoint outputs/guide_training_v1/aug/best.pt \
  --baseline-predictions outputs/guide_training_v1/aug/eval_val/predictions.npz \
  --output-dir outputs/guide_conflict_benchmark/guide_aug_v1
```

专项评测会核对原始Guide下的预测，所以必须传新模型自己的baseline-predictions，不能复用旧Mixed的预测档案。相同World/参数/有效案例构成不随模型改变，GT与输入哈希可用于逐案例比较。目录存在时需换新目录，避免覆盖。

如果重跑Control，用相同两条评估命令，将checkpoint/eval输出的aug改为control，并将专项输出目录改为guide_control_v1。已有Control也可以单独生成其原始与专项预测。

## 如何判断结果

同时查看原验证ADE、固定压力测试的ADE/路沿穿越率/道路外长度/侵入深度与出口横向误差。优先比较±3m等中等干预和clean/可见路沿子集，极端组合单独报告。不能只看增强训练loss或拿压力测试平均ADE对比原验证ADE。

旧Mixed的专项压力基线仅供总体参考：30World/540基础输入，原始0.891m/明显出道路3.33%；提前3m为1.766m/20.74%，延后3m为1.975m/21.48%。这些值属于Mixed，Guide数据增强的单因素结论应基于同配置Control对照。

生成与训练脚本已做小型测试：缓存World隔离与指纹、非Guide输入不变、确定性采样、原验证不增强、增强训练的checkpoint恢复。没有代替用户生成完整训练缓存或启动完整训练。

本轮相关测试结果：15通过、1项TensorBoard测试因依赖缺失跳过。双worker传输和两段续训与连续训练的一致性已验证；双worker测试需要允许本地进程间通信。
