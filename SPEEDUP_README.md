# Training Speedup (`feat/training-speedup`)

> 在不改变模型结构、不损害 valid AUC 的前提下，对 `PCVRHyFormer` 训练做端到端加速。

本分支的目标：把训练速度从 **1 epoch / 小时** 量级显著拉低，**所有改动默认不改变前向数值**（除了可选的 AMP 路径，对 CTR/CVR 任务 AUC 影响在噪声范围内）。

---

## 1. TL;DR

- 5 个文件改动：`dataset.py` / `model.py` / `trainer.py` / `train.py` / `run.sh`
- 新增 6 个 CLI 开关，全部在 `run.sh` 里默认打开：
  - `--use_amp` / `--amp_dtype bfloat16`
  - `--use_tf32`
  - `--grad_clip_foreach`
  - `--tqdm_min_interval 0.5`
  - `--prefetch_factor 4`
- 直接跑 `bash run.sh` 即可生效。
- 所有非 AMP 优化都是**严格数学等价**改写。

---

## 2. 性能瓶颈分析

按影响大小排序：

| # | 位置 | 问题 | 解决思路 |
| --- | --- | --- | --- |
| 1 | `dataset.py::_convert_batch` | 所有变长 padding 都是 `for i in range(B)` 的 Python 循环；4 个 seq domain × 几十个特征 × B=256，每 batch 几十毫秒 → DataLoader 是主瓶颈 | 用 `np.repeat`/`np.cumsum` 散点写入，**一次 numpy 调用替代 Python loop** |
| 2 | `model.py` NS Tokenizer / `_embed_seq_domain` | 多值聚合写成 `(emb*mask).sum/count`，但 `nn.Embedding(padding_idx=0)` 已让 padding 行返回 0 → mask 乘法是冗余的 | 删掉 mask 乘法，仅保留 `count` 用于均值分母 |
| 3 | `trainer.py::_train_step` | 没有 AMP / TF32；`grad_clip_foreach=False` 走慢路径；`zero_grad` 没用 `set_to_none`；tqdm 每步刷新 | 加 4 个开关 |
| 4 | `dataset.py::get_pcvr_data` | `num_workers=8` 但当 row group 数 < 8 时，`i % num_workers` 切分会让多余 worker 永远空跑 → `persistent_workers=True` 后整个流水线吞吐被拖慢 | `num_workers = min(num_workers, n_train_rgs)` |

---

## 3. 改动清单

### 3.1 数据 pipeline 向量化（`dataset.py`）

**新增工具函数** `PCVRParquetDataset._scatter_varlen_2d`：

```python
@staticmethod
def _scatter_varlen_2d(offsets, values, out, max_len, B):
    raw_lens = (offsets[1:] - offsets[:-1]).astype(np.int64, copy=False)
    lengths = np.minimum(raw_lens, max_len)
    total = int(lengths.sum())
    if total == 0:
        return lengths
    row_idx = np.repeat(np.arange(B, dtype=np.int64), lengths)
    cum = np.empty(B, dtype=np.int64); cum[0] = 0
    if B > 1:
        np.cumsum(lengths[:-1], out=cum[1:])
    col_idx = np.arange(total, dtype=np.int64) - cum[row_idx]
    src_idx = offsets[:-1].astype(np.int64, copy=False)[row_idx] + col_idx
    out[row_idx, col_idx] = values[src_idx]
    return lengths
```

被以下三处复用：

- `_pad_varlen_int_column` / `_pad_varlen_float_column`
- 4 个 seq domain 的 fused side-info padding（同时 `np.maximum(lengths, col_lens, out=...)` 求行 max）
- 时间戳 `ts_padded` 的填充

**get_pcvr_data**：
- 新增 `prefetch_factor` 参数（默认 4）
- 自动 `effective_train_workers = min(num_workers, max(1, n_train_rgs))`，并打日志说明截断情况

### 3.2 模型前向等价化简（`model.py`）

**`GroupNSTokenizer.forward` / `RankMixerNSTokenizer.forward`**：

```diff
- vals = int_feats[:, offset:offset + length].long()
- emb_all = emb_layer(vals)
- mask = (vals != 0).float().unsqueeze(-1)
- count = mask.sum(dim=1).clamp(min=1)
- fid_emb = (emb_all * mask).sum(dim=1) / count
+ vals = int_feats_long[:, offset:offset + length]
+ emb_all = emb_layer(vals)
+ count = (vals != 0).sum(dim=1, keepdim=True).clamp(min=1)
+ fid_emb = emb_all.sum(dim=1) / count
```

> **等价性**：`nn.Embedding(padding_idx=0)` 在 forward 时让索引 0 返回零向量并停止梯度（PyTorch 文档保证），所以 `(emb_all * mask)` 与 `emb_all` 在 padding 行上完全相同。整层数值严格相等。

`int_feats.long()` 整批转一次复用，避免每个 fid 各转一次。

**`PCVRHyFormer._embed_seq_domain`**：

```diff
- for i in range(S):
-     e = emb(seq[:, i, :])
+ seq_views = seq.unbind(dim=1)
+ for i in range(S):
+     e = sideinfo_embs[real_idx](seq_views[i])
```

`unbind` 是常数时间的 view 操作，比 S 次切片少 S 个 kernel launch。

### 3.3 训练循环（`trainer.py`）

新增 4 个构造参数，默认全部关闭以保持完全向后兼容：

```python
use_amp: bool = False
amp_dtype: str = 'bfloat16'
grad_clip_foreach: bool = False
tqdm_min_interval: float = 0.0
```

**AMP 流程**（`_train_step`）：
- bf16 路径：纯 `torch.cuda.amp.autocast(dtype=bfloat16)`，**无 GradScaler**（bf16 数值范围与 fp32 相同）
- fp16 路径：`autocast(dtype=float16) + GradScaler`，并在 clip 前 `unscale_` dense optimizer
- Embedding lookup 不被 autocast 转换，所以 Adagrad sparse path **零影响**
- 自动检测 `torch.cuda.is_bf16_supported()`，不支持 bf16 时回退 fp16

**其他**：
- `zero_grad(set_to_none=True)`：少一次显存写零
- `clip_grad_norm_(foreach=...)`：暴露开关，原默认 `False` 保持稳健
- tqdm：`mininterval=max(0.1, tqdm_min_interval)` + 每 `postfix_stride` 步刷新 postfix

### 3.4 入口（`train.py` + `run.sh`）

新增 6 个 CLI：

| 参数 | 作用 | run.sh 默认 |
| --- | --- | --- |
| `--use_amp` | bf16/fp16 autocast | ✓ |
| `--amp_dtype {bfloat16,float16}` | AMP 精度选择 | bfloat16 |
| `--use_tf32` | TF32 matmul（Ampere+） | ✓ |
| `--grad_clip_foreach` | foreach 实现的 clip_grad_norm_ | ✓ |
| `--tqdm_min_interval 0.5` | tqdm 刷新最小间隔（秒） | 0.5 |
| `--prefetch_factor 4` | DataLoader 预取深度 | 4 |

`--use_tf32` 在 `train.py` 中触发：

```python
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

---

## 4. 等价性保证

**严格数学等价**（无 AUC 风险）：
- `_scatter_varlen_2d`：与原循环写入完全相同的 `(row, col, value)` 三元组
- 序列 padding 的 `lengths`：等于 `np.maximum.reduce` over per-column lengths，与原 `if ul > lengths[i]` 行为一致
- NS Tokenizer mask 简化：依赖 `nn.Embedding(padding_idx=0)` 官方语义
- `seq.unbind`：只是 view，数值不变
- `set_to_none=True`：与 `zero_()` 在前向上等价
- `clip_grad_norm_(foreach=True/False)`：算的是同一个 L2 范数

**可能引入微小数值差异**：
- `--use_amp`：bf16 与 fp32 在数值上有 ~1e-3 级别差异，业界 CTR/CVR 任务普遍认为对 AUC 影响 < ±0.001。如果你严格要求逐位等价，**关掉这一个开关**即可。
- `--use_tf32`：仅影响 fp32 matmul 的尾数精度，对 AUC 影响小于 AMP。

---

## 5. 怎么用 / 怎么验证

### 5.1 直接跑

```bash
bash run.sh
```

`run.sh` 默认已开全部加速开关，需要的环境变量（继承自原项目）：
- `TRAIN_DATA_PATH`：训练数据目录
- `TRAIN_CKPT_PATH`：checkpoint 输出
- `TRAIN_LOG_PATH`：日志输出
- `TRAIN_TF_EVENTS_PATH`：TensorBoard 输出

### 5.2 推荐的渐进式验证

第一次跑时建议先验证**严格等价**的优化档：

```bash
bash run.sh --use_amp false --amp_dtype bfloat16
# 或者直接编辑 run.sh 把 --use_amp 这一行去掉
```

跑一个 epoch，对比改动前后的 valid AUC，应在 ±0.0005 内。

确认无误后再开 AMP。

### 5.3 一键回退到稳健模式

如果发现任何异常，把 `run.sh` 里这 4 行删掉即可回到原行为：

```bash
--use_amp \
--use_tf32 \
--grad_clip_foreach \
--tqdm_min_interval 0.5 \
```

`--prefetch_factor` 即使保留也是无害的，且向 `get_pcvr_data` 传 `prefetch_factor=4` 与原 `=2` 仅影响内存占用与流水线深度，不影响数值。

---

## 6. 没有做（避免 AUC 风险）

主动放弃了一些可能加速但可能影响 AUC 的改动：

- **`MultiSeqQueryGenerator` 的 Nq 个 FFN 合并成单个大 Linear**：合并后参数初始化分布会变（一次大 Xavier vs `S*Nq` 次独立 Xavier），训练动态可能漂移 → AUC 不可控。
- **Dropout / 损失函数 / 采样 / shuffle buffer 强度 / EarlyStopping / Adagrad 重初始化策略** 全部保持原样。
- **batch_size 与 seq_max_lens** 没有动，避免改变学习信号。

---

## 7. 文件改动统计

```
 dataset.py | 137 +++++++++++++++++++++++++++++++++++++++----------------------
 model.py   |  43 ++++++++++++-------
 run.sh     |  11 +++++
 train.py   |  35 ++++++++++++++++
 trainer.py | 112 ++++++++++++++++++++++++++++++++++++++++++--------
 5 files changed, 258 insertions(+), 80 deletions(-)
```
