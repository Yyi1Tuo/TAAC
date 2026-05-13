# Baseline 修改记录

本文档记录目前对官方 Baseline 做过的改动，以及 `reinit_cardinality_threshold`
参数的含义。

## 1. 训练与推理加速

### 1.1 新增 AMP 参数

在 `train.py` 中新增以下参数：

- `--amp`：启用 CUDA AMP 自动混合精度。
- `--no_amp`：关闭 AMP。
- `--amp_dtype {bf16,fp16}`：选择 AMP 计算类型，默认 `bf16`。

在 `trainer.py` 中，训练和验证 forward 都包在 `torch.amp.autocast(...)` 中。
loss、AUC、logloss 相关 logits 会转回 `float32`，避免指标计算受低精度张量影响。

`fp16` 模式会启用 `torch.amp.GradScaler`；`bf16` 模式不使用 GradScaler。
如果请求 `bf16` 但当前 CUDA 设备不支持，会直接抛错，尽早暴露问题。

### 1.2 新增 torch.compile 参数

在 `train.py` 中新增以下参数：

- `--compile`：启用 `torch.compile`。
- `--no_compile`：关闭 `torch.compile`。
- `--compile_mode {default,reduce-overhead,max-autotune}`：选择编译模式。

实现上保留原始 `self.model` 用于 optimizer、checkpoint 和 sparse reinit，
只把 `self.forward_model` 指向编译后的 callable。这样保存的 checkpoint
仍然是原始模型的 `state_dict()`，不会出现 `_orig_mod.` 之类的 key 污染。

### 1.3 run.sh 默认开启加速

`run.sh` 当前默认追加：

```bash
--amp
--amp_dtype bf16
--compile
--compile_mode default
```

如果需要做对照实验，可以在命令末尾追加：

```bash
--no_amp --no_compile
```

注意：`torch.compile` 首次运行会有编译预热开销，短小 smoke test 可能变慢；
正式长训练通常会在后续 step/epoch 回收这部分开销。

## 2. 随机种子与可复现性

原始代码已经在 `train.py` 中调用 `set_seed(args.seed)`，会固定 Python、
NumPy、PyTorch CPU/CUDA RNG。

本次额外增强：

- 在 `utils.py` 中设置 `torch.backends.cudnn.benchmark = False`。
- 在 `PCVRParquetDataset` 中保存独立 `seed`。
- shuffle buffer 使用独立 `torch.Generator`，不再依赖外部全局 RNG 的当前状态。
- DataLoader worker 通过 `worker_init_fn` 显式设置 `random/np/torch` seed。

这些改动让相同 seed 下的数据 shuffle 顺序更稳定。需要注意的是，GPU 上仍可能存在
个别非确定性 kernel；如果强行开启 `torch.use_deterministic_algorithms(True)`，
可能会牺牲速度甚至导致某些合法算子报错，因此当前没有开启这个严格模式。

## 3. `reinit_cardinality_threshold` 参数说明

### 3.1 这个参数控制什么

训练循环在每个 epoch 结束后，会对部分 embedding 重新初始化，并重建 Adagrad
优化器状态。这是一种 cold-restart 策略，目标是降低小训练集上 embedding 记忆训练样本、
过拟合历史统计的风险。

`--reinit_cardinality_threshold` 控制哪些 embedding 会被重置：

```text
重置条件：vocab_size > reinit_cardinality_threshold
```

也就是说：

- `--reinit_cardinality_threshold 0`：重置所有 `vocab_size > 0` 的 embedding。
- `--reinit_cardinality_threshold 10000`：只重置 `vocab_size > 10000` 的 embedding。
- 阈值越大，被重置的 embedding 越少。
- time bucket embedding 会保留，不参与这个重置策略。
- 被 `emb_skip_threshold` 跳过、根本没有创建 embedding 的特征，自然也不会被重置。

### 3.2 为什么默认值是 0

工作人员反馈：大赛训练集约 1M，规模相对较小，应该尽量重置 embedding。
因此 Baseline 中 `reinit_cardinality_threshold=0` 的设计意图是：

```text
默认重置所有有效稀疏 embedding
```

之前帮助文档写成了 `0 = never reset any Embedding`，这是文档错误，不是代码逻辑错误。
目前已将 `train.py` 的 help 文案改为：

```text
0 = reset all Embeddings with vocab_size > 0
```

### 3.3 与 `reinit_sparse_after_epoch` 的关系

`--reinit_sparse_after_epoch` 控制从第几个 epoch 结束后开始重置。

默认：

```bash
--reinit_sparse_after_epoch 1
--reinit_cardinality_threshold 0
```

含义是：

```text
从第 1 个 epoch 结束开始，每个 epoch 结束都重置所有有效稀疏 embedding，
并重建 Adagrad optimizer state。
```

如果想减少重置强度，可以提高 `reinit_cardinality_threshold`，例如只重置高基数特征。

## 4. 已做验证

已完成以下 smoke test：

- `python -m py_compile` 检查通过。
- synthetic batch 上 AMP + compile 的 train/eval step 通过。
- 使用 HuggingFace `TAAC2026/data_sample_1000` 临时生成 parquet/schema 后，
  跑通过完整 1 epoch 训练、验证、保存 checkpoint。
- 同 seed 的 DataLoader 首批顺序一致。
- 小模型测试确认 `reinit_cardinality_threshold=0` 会触发 embedding 重置。

## 5. 环境补充

按测试需要，在 `NewsRecommend` conda 环境中安装了：

- `datasets`
- `tensorboard`

## 6. 训练/推理提交文件同步说明

比赛官方环境分训练和推理两阶段：

- 训练阶段调用 `run.sh`，会用到 `train.py`、`trainer.py`、`dataset.py`、`model.py`、`utils.py`。
- 推理阶段通常只包含 `dataset.py`、`infer.py`、`model.py`。

当前改动的同步建议：

- `model.py`：建议同步到推理阶段。推理加载 checkpoint 时，模型结构必须和训练阶段一致。
  本次没有改模型结构，但为了避免训练/推理文件版本漂移，仍建议提交同一份 `model.py`。
- `dataset.py`：建议同步到推理阶段。推理的数据解析、序列截断、time bucket 逻辑必须和训练一致。
  本次新增的 seed 参数有默认值，对推理兼容；不会改变 `is_training=False` 的标签占位逻辑。
- `infer.py`：已导入官方文件。当前推理阶段保持全精度，不启用 AMP。已将推理循环从
  `torch.no_grad()` 改为 `torch.inference_mode()`，并修正 `num_workers=0` 时不能设置
  `prefetch_factor` 的兼容性问题。没有默认启用 `torch.compile`，避免官方推理环境中出现额外
  编译开销或兼容性风险。
- `train.py`、`trainer.py`、`run.sh`、`utils.py`：推理阶段通常不需要同步，除非官方推理入口显式 import
  了其中某个文件。

特别注意：训练 checkpoint 旁边会保存 `train_config.json`，其中现在包含 `amp/compile/compile_mode`
等训练加速参数。推理脚本如果读取 `train_config.json`，应该只取模型构造相关字段，或者显式忽略
这些运行时加速字段，不能把整个 `train_config` 原样传给 `PCVRHyFormer(**kwargs)`。

## 7. Time 分支：绝对时间特征增强

本分支在保留原有 recency bucket 的基础上，增加样本级和序列事件级绝对时间特征。

### 7.1 数据侧新增字段

`dataset.py` 新增 `TIME_FEATURE_DIM=12` 和 `build_time_features(...)`。

每个 Unix 秒级 timestamp 会被转换为 12 维 `float32` 特征：

- daily sin/cos
- weekly sin/cos
- 30-day cycle sin/cos
- yearly sin/cos
- hour_norm
- day_of_week_norm
- is_weekend
- timestamp_scaled

说明：最初计划中写的是 11 维，但上面列出的完整特征实际是 12 维。实现中保留
`timestamp_scaled` 这个绝对时间信号，因此最终使用 12 维。

batch 中新增：

- `context_time_feats`: `[B, 12]`，来自样本级 `timestamp`
- `{domain}_abs_time_feats`: `[B, L, 12]`，来自每个序列域的事件 timestamp，padding 位置全 0

### 7.2 模型侧接入方式

`ModelInput` 新增：

- `context_time_feats`
- `seq_abs_time_feats`

模型新增两个结构参数：

- `context_time_dim`
- `seq_abs_time_dim`

接入方式：

- `context_time_feats` 拼接到 `user_dense_feats` 后，继续走现有 user dense NS token，不新增 NS token。
- 每个序列 domain 新增一个轻量 `Linear(seq_abs_time_dim, d_model, bias=False) + LayerNorm`，
  将 `{domain}_abs_time_feats` 投影后加到对应 sequence token embedding。
- 原有 `seq_time_buckets` 和 `time_embedding` 保持不变，用于表达行为距当前样本的 recency。

### 7.3 训练与推理配置

`train.py` 新增默认开启开关：

```bash
--use_context_time_feats / --no_context_time_feats
--use_seq_abs_time_feats / --no_seq_abs_time_feats
```

训练保存的 `train_config.json` 会记录这两个开关。`infer.py` 会根据 `train_config.json`
恢复：

- `context_time_dim = TIME_FEATURE_DIM` 或 `0`
- `seq_abs_time_dim = TIME_FEATURE_DIM` 或 `0`

旧 checkpoint 若缺少这些字段，推理 fallback 为 `0` 维，以保证旧模型可以 strict load。
推理阶段保持全精度，不启用 AMP。

### 7.4 验证

已完成：

- `python -m py_compile TAAC/train.py TAAC/trainer.py TAAC/dataset.py TAAC/model.py TAAC/infer.py`
- sample batch 形状检查：
  - `context_time_feats == [B, 12]`
  - 每个 `{domain}_abs_time_feats == [B, L, 12]`
  - padding 位置绝对时间特征全 0
- HuggingFace sample 小模型 1 epoch 训练、验证、保存 checkpoint 跑通
- 新 checkpoint 通过 `infer.py` strict load，并生成 `predictions.json`
- 旧 checkpoint 缺少时间特征配置时，`infer.py` fallback 到 0 维并 strict load 成功

## 8. SENet v2：RankMixer block token-wise SENet

`v2` 分支只在 HyFormer block 内部的 `RankMixerBlock` 输入处加入 token-wise SENet。
作用点固定在 token mixing 之前：

```text
combined query/NS tokens -> [B, T, d_model]
TokenSENet               -> [B, T, d_model]
token_mixing/FFN         -> [B, T, d_model]
```

新增参数：

```bash
--use_block_senet / --no_block_senet
--senet_reduction 4
```

旧 checkpoint 缺少 `use_block_senet` 时，`infer.py` fallback 为关闭，以保证 strict load 兼容。
