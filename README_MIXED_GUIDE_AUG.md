# 混合训练：路口多 Guide 增强 + 原分流数据

沿用 `outputs/road_mixed_v2` 的训练/验证划分、BEV、GT、导航、采样比例和模型/Loss；只启用原有路口 Guide 增强。从头训练，输出新目录，不使用上一轮 checkpoint 续训。

路口左右转每次有50%概率使用原Guide、50%概率选择8条变体之一：提前/滞后1–6m、正负横向偏移0.5–3m及联合扰动；每条世界Guide在不同pose一致截取。路口直行及所有分流样本保持原Guide。每epoch仍为51462条样本；按当前组成，整体Guide替换比例约9.8%，不是全体样本的50%。验证不增强。

## 用户执行

在项目根目录、具有Torch/CUDA的环境运行：

```bash
conda activate torch310

# 检查GPU；新配置device=cuda，不会静默回退CPU。
python -c "import torch; print('CUDA:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable'"

# 原路口World/routes用于构造变体，但缓存绑定当前完整混合训练集的指纹。
# 使用新目录；旧guide_train_cache_v1不能直接用于此混合数据。
python scripts/build_training_guide_cache.py \
  --source outputs/v4_world300_inside13 \
  --mixed-source outputs/road_mixed_v2 \
  --output-dir outputs/guide_train_cache_road_mixed_v1

python train.py \
  --config configs/train_road_mixed_guide_aug_v1.json \
  --no-tensorboard
```

无需重新生成或混合道路数据。缓存仅来自混合训练集中的路口World，排除混合验证World；不复制完整场景，不启动训练。缓存脚本目录必须尚不存在，避免覆盖；已成功生成则直接进行训练。若混合数据发生改变，必须使用新目录重建对应缓存。

配置沿用上一轮seed42、batch16、LR1e-4、最多150epoch、25epoch早停、按原混合val ADE选best；设备固定cuda。训练结果在 `outputs/training_road_mixed_guide_aug_v1`，保留 `training_road_mixed_v2` 基线。日志仍显示 `val_intersection_ADE` 和 `val_lane_transition_ADE`，并记录实际Guide替换比例。

## 完成后评估

```bash
python evaluate.py \
  --checkpoint outputs/training_road_mixed_guide_aug_v1/best.pt \
  --output-dir outputs/training_road_mixed_guide_aug_v1/eval_val
```

除总体和分组ADE外，应复跑原有自定义输入，比较是否保持目标出口、是否修开路沿冲突；不能只用总体ADE判断增强成功。此轮仍未修改几何生成范围、分流可达性、感知FOV或模型结构，不能预先保证解决自由绘制场景问题。

## 兼容性验证

`tests/test_guide_training.py` 包含原缓存兼容、仅Guide变化、确定性/多worker/断点恢复、混合缓存指纹、分流left/right不被错误增强以及缺失路口键报错。混合缓存使用version2并声明intersection范围；原单路口version1缓存仍兼容。
