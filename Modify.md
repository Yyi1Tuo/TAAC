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
