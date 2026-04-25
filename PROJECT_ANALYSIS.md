# PCVRHyFormer 项目深度解析

> **项目**：2026 TAAC 推荐排序赛题 — PCVR (Post-Click Conversion Rate) 预估模型
> **核心模型**：`PCVRHyFormer` —— 多序列异构 Hybrid Transformer
> **代码位置**：`2026TAAC/{dataset.py, model.py, trainer.py, utils.py, train.py, run.sh, ns_groups.json}`

本文聚焦三个核心环节：**① 数据预处理 ② 模型架构 ③ 损失设计**，对每个细节做逐行级别的拆解。

---

## 目录

1. [数据预处理](#一数据预处理)
   - 1.1 [Schema 定义与五类特征](#11-schema-定义与五类特征)
   - 1.2 [Parquet 流式加载与 Row Group 切分](#12-parquet-流式加载与-row-group-切分)
   - 1.3 [`_convert_batch`：Arrow → Tensor 全流程](#13-_convert_batcharrow--tensor-全流程)
   - 1.4 [变长序列 Padding 策略](#14-变长序列-padding-策略)
   - 1.5 [时间分桶（Time Bucket）机制](#15-时间分桶time-bucket机制)
   - 1.6 [越界 ID 处理与 OOB 统计](#16-越界-id-处理与-oob-统计)
   - 1.7 [Shuffle Buffer 与多 Worker 并行](#17-shuffle-buffer-与多-worker-并行)
   - 1.8 [输出 Batch 数据结构](#18-输出-batch-数据结构)
2. [模型架构](#二模型架构)
   - 2.1 [整体数据流](#21-整体数据流)
   - 2.2 [Token 三大类与组装顺序](#22-token-三大类与组装顺序)
   - 2.3 [NS Tokenizer：两种变体](#23-ns-tokenizer两种变体)
   - 2.4 [Sequence Embedding 子模块](#24-sequence-embedding-子模块)
   - 2.5 [`MultiSeqQueryGenerator`：每个序列独立的 Q 生成](#25-multiseqquerygenerator每个序列独立的-q-生成)
   - 2.6 [三种 Sequence Encoder](#26-三种-sequence-encoder)
   - 2.7 [`CrossAttention` 与 `RoPEMultiheadAttention`](#27-crossattention-与-ropemultiheadattention)
   - 2.8 [`RankMixerBlock`：Token Mixing + Per-token FFN](#28-rankmixerblocktoken-mixing--per-token-ffn)
   - 2.9 [`MultiSeqHyFormerBlock`：核心交互单元](#29-multiseqhyformerblock核心交互单元)
   - 2.10 [输出层与分类头](#210-输出层与分类头)
   - 2.11 [关键约束 d_model % T == 0](#211-关键约束-d_model--t--0)
3. [损失设计与训练优化](#三损失设计与训练优化)
   - 3.1 [BCEWithLogitsLoss（默认）](#31-bcewithlogitsloss默认)
   - 3.2 [Focal Loss（不平衡场景）](#32-focal-loss不平衡场景)
   - 3.3 [双优化器：Adagrad（Sparse）+ AdamW（Dense）](#33-双优化器adagradsparse--adamwdense)
   - 3.4 [高基数特征冷重启](#34-高基数特征冷重启)
   - 3.5 [评估指标：Binary AUC + LogLoss](#35-评估指标binary-auc--logloss)
   - 3.6 [显存优化与训练稳定性](#36-显存优化与训练稳定性)
4. [核心超参速查表](#四核心超参速查表)
5. [设计亮点与一句话总结](#五设计亮点与一句话总结)

---

## 一、数据预处理

数据预处理逻辑全部位于 `dataset.py`，核心类是 **`PCVRParquetDataset`**（继承 `IterableDataset`），通过 `get_pcvr_data()` 装配 `DataLoader`。

### 1.1 Schema 定义与五类特征

`schema.json` 把所有特征划分为 **5 类**，每类对应一个 `FeatureSchema` 对象：

| 类别 | schema.json 字段 | 元素结构 | 含义 |
|---|---|---|---|
| 用户离散 | `user_int` | `[fid, vocab_size, dim]` | 用户标签、性别、年龄段等（带词表） |
| 物料离散 | `item_int` | `[fid, vocab_size, dim]` | 物料类目、来源、标签等 |
| 用户稠密 | `user_dense` | `[fid, dim]` | 预训练向量、统计特征（连续值） |
| 物料稠密 | (空) | - | 当前数据集未提供 |
| 行为序列 | `seq` | `{prefix, ts_fid, features:[[fid, vs], ...]}` | seq_a/b/c/d 4 个域的多列序列 |

**`FeatureSchema` 的核心数据结构**：
```python
class FeatureSchema:
    entries: List[Tuple[fid, offset, length]]  # 每个 fid 在拼接大向量中的位置
    total_dim: int                              # 拼接后的总维度
    _fid_to_entry: Dict[fid, (offset, length)]  # 快速查找
```

> **作用**：模型不需要逐 fid 维护 Embedding 字典，而是统一拿到 `(B, total_dim)` 的扁平张量，再用 `offset/length` 切片。这样可以**预分配 numpy buffer**，避免每个 batch 重新 `np.zeros`。

**4 个序列域 (`seq_a/b/c/d`) 的 max_len 默认值**：
- `seq_a:256, seq_b:256, seq_c:512, seq_d:512`（在 `run.sh` 通过 `--seq_max_lens` 指定）

---

### 1.2 Parquet 流式加载与 Row Group 切分

```python
# get_pcvr_data() 的关键逻辑
pq_files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
rg_info = []
for f in pq_files:
    pf = pq.ParquetFile(f)
    for i in range(pf.metadata.num_row_groups):
        rg_info.append((f, i, pf.metadata.row_group(i).num_rows))

n_valid_rgs = max(1, int(total_rgs * valid_ratio))   # 取尾部作为验证集
n_train_rgs = total_rgs - n_valid_rgs
if train_ratio < 1.0:                                # 训练集再按比例截取前段
    n_train_rgs = max(1, int(n_train_rgs * train_ratio))
```

**关键点**：
- 验证集 = **尾部 `valid_ratio`** 的 Row Group（保证时间序，避免数据泄露）
- 训练集 = **前段 `train_ratio` × 剩余 RG**（便于做小数据快速实验）
- **流式读取**：`pf.iter_batches(batch_size=B, row_groups=[rg_idx])` 每次只加载一个 RG，控制内存

**多 Worker 切分策略**（`__iter__` 中）：
```python
if worker_info.num_workers > 1:
    rg_list = [rg for i, rg in enumerate(rg_list) if i % num_workers == worker_id]
```
按 RG 取模分配，每个 Worker 独立读取自己负责的 RG，无需进程间通信。

---

### 1.3 `_convert_batch`：Arrow → Tensor 全流程

每次拉到一个 `pa.RecordBatch` (B 行)，按以下顺序转换：

```
RecordBatch (Arrow)
   │
   ├─► (1) 提取 timestamp + label
   │       label = (label_type == 2).astype(int64)   # 二分类标签
   │
   ├─► (2) user_int / item_int 写入预分配 buffer
   │       - dim==1：标量，直接 to_numpy + clip(<=0 → 0)
   │       - dim>1 ：变长 list，调用 _pad_varlen_int_column → [B, dim]
   │       - 每列做 OOB 检测 (_record_oob)，超过 vocab_size 的 clip 到 0
   │
   ├─► (3) user_dense 写入 buffer
   │       - 变长 float list，_pad_varlen_float_column → [B, dim]
   │
   └─► (4) 4 个序列域分别处理（fused 单趟写入 3D buffer）
           对每个 sideinfo 列：
              for i in range(B):
                  out[i, c, :ul] = vals[s:s+ul]  # 直接写入 [B, S, L] buffer
           out[out <= 0] = 0
           lengths[i] = max(lengths[i], ul)      # 取最长侧信息列作为序列实长
           然后做时间分桶 (见 1.5)
```

**性能优化亮点**：
1. **预分配 buffer**：`_buf_user_int` / `_buf_item_int` / `_buf_user_dense` / `_buf_seq[domain]` 在 `__init__` 中创建，每个 batch `[:] = 0` 复用
2. **预计算列索引**：`self._col_idx = {name: i}` 避免每个 batch 重复 `schema.get_field_index`
3. **预计算 plan**：`_user_int_plan = [(col_idx, dim, offset, vocab_size), ...]` 一次性确定所有列的处理方案
4. **fused 写入**：序列域直接写入 3D buffer `[B, n_feats, max_len]`，省去 `np.stack` 的开销

---

### 1.4 变长序列 Padding 策略

`_pad_varlen_int_column` / `_pad_varlen_float_column` 是核心 padding 函数：

```python
def _pad_varlen_int_column(arrow_col, max_len, B):
    offsets = arrow_col.offsets.to_numpy()       # Arrow ListArray 的 offset 数组
    values = arrow_col.values.to_numpy()
    padded = np.zeros((B, max_len), dtype=np.int64)
    lengths = np.zeros(B, dtype=np.int64)
    for i in range(B):
        start, end = int(offsets[i]), int(offsets[i + 1])
        raw_len = end - start
        if raw_len <= 0: continue
        use_len = min(raw_len, max_len)           # 截断
        padded[i, :use_len] = values[start:start + use_len]
        lengths[i] = use_len
    padded[padded <= 0] = 0                       # -1/null/0 都视为 padding
    return padded, lengths
```

**关键约定**：
- **0 = padding**（Embedding 层用 `padding_idx=0`）
- **-1 / null** 也归为 padding（数据中 -1 表示缺失）
- **截断方向**：保留前 `max_len` 个；序列域里 `out[i, c, :ul] = vals[s:s+ul]` 也是同样语义
- **长度信息**：`lengths[i]` 取所有 sideinfo 列里最长的那一列，用于后续构造 attention mask

---

### 1.5 时间分桶（Time Bucket）机制

为了显式建模"行为新鲜度"，把序列里每个行为的时间戳映射成一个**离散桶 ID**：

```python
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, ..., 60,                    # 秒级
    120, 180, ..., 3600,                   # 分钟级
    5400, ..., 86400,                      # 小时级
    172800, ..., 604800,                   # 天级
    1123200, ..., 2592000,                 # 周级
    4320000, ..., 31536000,                # 月级 ~ 年级
], dtype=np.int64)  # 共 64 个边界
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1  # = 65（含 padding=0）
```

分桶逻辑（`_convert_batch` 末尾）：
```python
time_diff = max(current_timestamp - behavior_timestamp, 0)
raw_bucket = clip(searchsorted(BUCKET_BOUNDARIES, time_diff), 0, 63)
bucket = raw_bucket + 1                    # 偏移到 [1, 64]，0 留给 padding
bucket[behavior_timestamp == 0] = 0        # padding 位强制为 0
```

**模型侧**（`PCVRHyFormer._embed_seq_domain`）：
```python
token_emb = token_emb + self.time_embedding(time_bucket_ids)  # 加性融合
# self.time_embedding = nn.Embedding(65, d_model, padding_idx=0)
```

**为什么是非线性分桶？**
- 推荐场景中，"几秒前"和"几小时前"的行为权重差异巨大，但"3 个月前"和"4 个月前"几乎等价
- 边界按 **秒/分/时/天/周/月** 几何级数划分，更符合用户兴趣衰减规律

---

### 1.6 越界 ID 处理与 OOB 统计

```python
def _record_oob(self, group, col_idx, arr, vocab_size):
    oob_mask = arr >= vocab_size
    if not oob_mask.any(): return
    # 累计统计：count, max, min_oob, vocab
    if self.clip_vocab:
        arr[oob_mask] = 0     # 超界的 clip 到 0（padding）
    else:
        raise ValueError(...)
```

**触发场景**：
- schema.json 里的 vocab_size 与实际数据不一致（数据里出现了未见过的更大 ID）
- 训练 / 推理数据不一致

**特殊情况 `vs == 0`**：
- 表示"该特征没有词表信息"
- 预处理直接 `arr[:] = 0`，模型侧也只创建一个 1-slot Embedding，永远不会越界

`dump_oob_stats()` 可在训练结束后输出统计报告，定位 schema 与数据不匹配的特征。

---

### 1.7 Shuffle Buffer 与多 Worker 并行

```python
# __iter__ 内部
buffer = []
for batch in pf.iter_batches(...):
    batch_dict = self._convert_batch(batch)
    if shuffle and buffer_batches > 1:
        buffer.append(batch_dict)
        if len(buffer) >= buffer_batches:
            yield from self._flush_buffer(buffer)   # 跨 batch 行级 shuffle
            buffer = []
    else:
        yield batch_dict
```

`_flush_buffer` 把 `buffer_batches` 个 batch 拼成大张量，`torch.randperm` 行级打散后再切回 batch_size 输出。

**优势**：在内存可控的前提下，提供比"仅 batch 内 shuffle"更好的随机性。

**多 Worker 配合**：
```python
DataLoader(train_dataset, batch_size=None,
           num_workers=8, pin_memory=True,
           persistent_workers=True, prefetch_factor=2)
```
- `batch_size=None`：因为 `IterableDataset` 已经在内部组好 batch
- `persistent_workers=True`：避免每个 epoch 重建 worker 进程
- `prefetch_factor=2`：每个 worker 预取 2 个 batch
- `set_sharing_strategy('file_system')`：避开 `/dev/shm` 容量限制

---

### 1.8 输出 Batch 数据结构

`_convert_batch` 最终返回：

```python
{
    'user_int_feats':  (B, user_int_total_dim) int64,
    'user_dense_feats':(B, user_dense_total_dim) float32,
    'item_int_feats':  (B, item_int_total_dim) int64,
    'item_dense_feats':(B, 0) float32,                # 当前为空
    'label':           (B,) int64,
    'timestamp':       (B,) int64,
    'user_id':         List[int] of length B,         # 推理时用，训练不用
    '_seq_domains':    ['seq_a','seq_b','seq_c','seq_d'],
    # —— 4 个序列域，每个域 3 个张量 ——
    'seq_a':              (B, n_sideinfo_a, max_len_a) int64,
    'seq_a_len':          (B,) int64,
    'seq_a_time_bucket':  (B, max_len_a) int64,       # ∈ [0, 64]
    'seq_b': ..., 'seq_b_len': ..., 'seq_b_time_bucket': ...,
    'seq_c': ..., 'seq_c_len': ..., 'seq_c_time_bucket': ...,
    'seq_d': ..., 'seq_d_len': ..., 'seq_d_time_bucket': ...,
}
```

`trainer._make_model_input` 会把上面的 dict 重组为 `ModelInput` NamedTuple，方便模型侧按字段访问。

---

## 二、模型架构

模型代码全部位于 `model.py`（~1700 行），核心类 `PCVRHyFormer`。下面按"自底向上"的顺序拆解。

### 2.1 整体数据流

```
                  ┌────────────────────────────────────────────────────┐
                  │              PCVRHyFormer.forward                   │
                  │                                                    │
 user_int  ──►── user_ns_tokenizer ──►─┐                               │
 user_dense ──►─ user_dense_proj ──►───┤                               │
                                       ├──► ns_tokens (B, num_ns, D)   │
 item_int  ──►── item_ns_tokenizer ──►─┤                               │
 item_dense ──►─ item_dense_proj ──►───┘                               │
                                                                       │
 seq_a/b/c/d ──►─ _embed_seq_domain ──►── seq_tokens_list[4]           │
                  (含 time_embedding)     (B, L_i, D) × 4              │
                                                                       │
                  ┌─────────────────────────────────────────┐          │
                  │          MultiSeqQueryGenerator          │          │
                  │  对每个 seq i：                            │          │
                  │   GlobalInfo_i = Concat(NS, MeanPool(seq_i))         │
                  │   Q_i = [FFN_{i,j}(GlobalInfo_i)]_{j=1..Nq}         │
                  └────────────┬────────────────────────────┘          │
                               │ q_tokens_list (B, Nq, D) × 4          │
                               ▼                                       │
                  ┌─────────────────────────────────────────┐          │
                  │  MultiSeqHyFormerBlock × N (N=2)         │          │
                  │   Step1: SeqEvolution (per-seq encoder) │          │
                  │   Step2: QueryDecoding (per-seq cross-attn)        │
                  │   Step3: 拼接 [Q_1..Q_S, NS] → RankMixerBlock      │
                  │   Step4: 拆回 Q_i 与 NS                            │
                  └────────────┬────────────────────────────┘          │
                               │                                       │
                               ▼                                       │
                  cat(Q_1..Q_S) → output_proj → (B, D)                 │
                               │                                       │
                               ▼                                       │
                          clsfier (MLP) → (B, action_num=1) logits     │
                  └────────────────────────────────────────────────────┘
```

### 2.2 Token 三大类与组装顺序

模型把所有信号统一成 token，主要分三类：

| Token 类别 | 来源 | 数量 | 是否参与 Sequence Encoder |
|---|---|---|---|
| **NS Tokens** | user_int + user_dense + item_int + item_dense | `num_ns`（默认 5+1+2+0 = 8） | 否，只参与 RankMixer |
| **Seq Tokens** | seq_a/b/c/d 的 sideinfo + time_bucket | `L_i` (变长，最大 256/512) | 是 |
| **Q Tokens** | Query Generator 从 NS 和 Seq pool 派生 | `num_queries × num_sequences`（2×4 = 8） | 否，参与 cross-attn 和 RankMixer |

**关键 NS 拼接顺序**（`forward` 中）：
```python
ns_parts = [user_ns]                              # (B, num_user_ns, D)
if has_user_dense: ns_parts.append(user_dense_tok)  # (B, 1, D)
ns_parts.append(item_ns)                          # (B, num_item_ns, D)
if has_item_dense: ns_parts.append(item_dense_tok)  # (B, 1, D)
ns_tokens = torch.cat(ns_parts, dim=1)            # (B, num_ns, D)
```

> **顺序的语义**：[user_int, user_dense, item_int, item_dense]，保证训练/推理一致。

### 2.3 NS Tokenizer：两种变体

#### 2.3.1 `GroupNSTokenizer`（按业务语义分组）

```
fid_1 ──► Embedding ──┐
fid_2 ──► Embedding ──┼──► concat ──► Linear(num_fids*emb_dim → d_model) ──► LayerNorm ──► SiLU ──► token_1
fid_3 ──► Embedding ──┘                                      (Group 1)
...
                       (每个 group 输出一个 token)
```

- 分组方案来自 `ns_groups.json`（如 7 个 user group + 4 个 item group）
- **多值特征处理**：`length > 1` 时按非零位 mask 做 mean pooling
  ```python
  vals = int_feats[:, offset:offset+length]
  emb_all = emb_layer(vals)                    # (B, length, emb_dim)
  mask = (vals != 0).float().unsqueeze(-1)
  fid_emb = (emb_all * mask).sum(1) / mask.sum(1).clamp(min=1)
  ```

#### 2.3.2 `RankMixerNSTokenizer`（默认，等切分）

```
所有 fid 的 Embedding ──► concat 成一个长向量 (B, total_emb_dim)
                          │
                          ├──► pad 到 num_ns_tokens 整除
                          │
                          ├──► split 成 num_ns_tokens 块
                          │
                          └──► 每块独立 Linear(chunk_dim → d_model) + LN + SiLU
                              └──► (B, num_ns_tokens, d_model)
```

- 来自 RankMixer 论文：让 token 数与特征语义解耦，便于调参
- `--user_ns_tokens 5 --item_ns_tokens 2` 直接指定 token 数

**两种 Tokenizer 共有的两个跳过机制**：
```python
skip = vs <= 0 or (emb_skip_threshold > 0 and vs > emb_skip_threshold)
if skip:
    embs.append(None)              # 不创建 Embedding
                                    # 前向时输出零向量
```
- `vs <= 0`：无词表信息
- `vs > emb_skip_threshold`（如 1e6）：词表过大省显存，零向量替代

### 2.4 Sequence Embedding 子模块

```python
def _embed_seq_domain(seq, sideinfo_embs, proj, is_id, emb_index, time_bucket_ids):
    # seq: (B, S, L)，S 个 sideinfo 列，L 个时间步
    emb_list = []
    for i in range(S):
        if emb_index[i] == -1:
            emb_list.append(zeros(B, L, emb_dim))   # 跳过的特征
        else:
            e = sideinfo_embs[real_idx](seq[:, i, :])  # (B, L, emb_dim)
            if is_id[i] and self.training:
                e = self.seq_id_emb_dropout(e)         # 高基数特征额外 dropout (rate*2)
            emb_list.append(e)
    cat_emb = torch.cat(emb_list, dim=-1)          # (B, L, S*emb_dim)
    token_emb = F.gelu(proj(cat_emb))              # Linear → (B, L, D)

    if num_time_buckets > 0:
        token_emb = token_emb + time_embedding(time_bucket_ids)  # 加 time emb

    return token_emb  # (B, L, D)
```

**两个独立阈值的协作**：
- `emb_skip_threshold`：决定**是否创建** Embedding（节省显存）
- `seq_id_threshold`：决定 **id 类**特征是否额外 dropout（缓解 id 过拟合）
- 二者完全独立：可以建 Embedding 但加 dropout，也可以跳过 Embedding 不加 dropout

**Padding mask** 由 `seq_lens` 通过 `_make_padding_mask` 构造：
```python
idx = arange(max_len).unsqueeze(0)         # (1, max_len)
mask = idx >= seq_len.unsqueeze(1)          # (B, max_len)，True 表示 padding
```

### 2.5 `MultiSeqQueryGenerator`：每个序列独立的 Q 生成

为什么要"每个序列独立生成 Q"？因为不同行为域（如曝光 vs 点击 vs 转化）的语义差异大，单一全局 Q 难以兼顾。

```python
for i in range(num_sequences):
    # 1. 序列 mean pooling（仅有效位）
    valid_mask = ~seq_padding_masks[i]           # (B, L_i)
    seq_pooled = (seq_tokens[i] * valid_mask.unsqueeze(-1)).sum(1) / count

    # 2. Concat NS + 当前序列的 pooled
    global_info = cat([ns_flat, seq_pooled], dim=-1)  # (B, (num_ns+1)*D)
    global_info = LayerNorm(global_info)              # 防止大维度拼接梯度爆炸

    # 3. 用 num_queries 个独立 FFN 派生 Q tokens
    queries = [FFN_{i,j}(global_info) for j in range(num_queries)]
    q_tokens = stack(queries, dim=1)                  # (B, num_queries, D)
```

**每个 FFN 结构**：
```
Linear((M+1)*D → 4*D) → SiLU → Linear(4*D → D) → LayerNorm
```

**总参数量**：`num_sequences × num_queries` 个独立 FFN

### 2.6 三种 Sequence Encoder

通过 `--seq_encoder_type` 选择，三种实现各有侧重：

#### (a) `SwiGLUEncoder`（轻量，无 attention）
```
x ──► LayerNorm ──► SwiGLU(4x) ──► Dropout ──┐
└────────────────────────────────────────────┴──► +
```
适用于序列短、特征已经足够丰富的场景。

#### (b) `TransformerEncoder`（默认）
标准 Pre-LN Transformer Layer：
```
x ──► LN ──► SelfAttn(RoPE) ──► +residual ──► LN ──► FFN(GELU,4x) ──► +residual
```

#### (c) `LongerEncoder`（Top-K 压缩，适配超长序列）
**自适应**两种模式：
- **L > top_k（首块）**：Cross Attention，Q = 最后 top_k 个 token，K/V = 全序列 → 输出 `(B, top_k, D)`
- **L ≤ top_k（后续块）**：Self Attention，Q=K=V = top_k tokens

```python
def _gather_top_k(x, key_padding_mask):
    valid_len = (~key_padding_mask).sum(1)            # (B,)
    actual_k = clamp(valid_len, max=top_k)
    start_pos = valid_len - actual_k                  # 取最近 top_k 个
    indices = start_pos.unsqueeze(1) + arange(top_k)  # (B, top_k)
    top_k_tokens = gather(x, dim=1, index=indices)
    # 同时返回 position_indices 用于 Q-side RoPE（Q 位置不连续，需独立 cos/sin）
```

**亮点**：cross-attn 时 Q 来自非连续位置，需要单独 gather RoPE 的 cos/sin：
```python
q_rope_cos = gather(rope_cos.expand(B, -1, -1), dim=1, index=q_pos_indices_expanded)
```
保证位置编码语义正确。

### 2.7 `CrossAttention` 与 `RoPEMultiheadAttention`

#### `RoPEMultiheadAttention`（自研多头注意力）

为什么不用 `nn.MultiheadAttention`？因为要支持 **RoPE** 和 **Q/K 不同 RoPE**（LongerEncoder 场景）。

```python
Q = W_q(query); K = W_k(key); V = W_v(value)
Q, K, V = reshape to (B, num_heads, L, head_dim)

if rope:
    K = apply_rope(K, rope_cos, rope_sin)
    Q = apply_rope(Q, q_rope_cos or rope_cos, q_rope_sin or rope_sin)

out = F.scaled_dot_product_attention(Q, K, V, attn_mask=..., dropout_p=...)
out = nan_to_num(out, 0.0)                  # 处理"全 padding"导致的 softmax NaN
out = reshape to (B, Lq, D)

# === 关键的门控输出 ===
G = W_g(query)                              # 初始化为 zeros + bias=1.0
out = out * sigmoid(G)                      # 门控权重，初始为 sigmoid(1)≈0.73
out = W_o(out)
```

**门控初始化技巧**：`W_g.weight=0, bias=1.0` → 训练初期 `sigmoid(G) ≈ 0.73`，相当于一个"温和的恒等映射"，让模型从一个稳定的起点学起。

#### `CrossAttention`

```python
residual = query
query = LN(query)
key_value = LN(key_value)
out = RoPEMHA(query, key_value, key_value, rope_on_q=False)  # 仅 KV 加 RoPE
out = residual + out
```

`rope_on_q=False`：因为 Q 是 NS pool 出来的全局 token，没有"位置"概念，不需要旋转。

### 2.8 `RankMixerBlock`：Token Mixing + Per-token FFN

来自 RankMixer 论文，结构精妙：

#### Step 1：Token Mixing（无参数 reshape）
```python
# 输入 Q: (B, T, D)，要求 D % T == 0，d_sub = D // T
Q_split   = Q.view(B, T, T, d_sub)              # 把 D 维切成 T 个子空间
Q_rewired = Q_split.transpose(1, 2)             # 交换 token 轴和 subspace 轴
Q_hat     = Q_rewired.view(B, T, D)             # 拼回 (B, T, D)
```
**效果**：每个新 token 的第 i 个子空间，来自原第 i 个 token 的第 token-th 个子空间 → **token 间无参数信息混合**。

#### Step 2：Per-token FFN（共享参数）
```python
x = LN(Q_hat)
x = fc1(x)         # (B, T, 4D)
x = GELU(x)
x = dropout(x)
Q_e = fc2(x)       # (B, T, D)
```

#### Step 3：残差 + Post-LN
```python
Q_boost = post_norm(Q + Q_e)
```

**三种模式**（`rank_mixer_mode`）：
- `full`：token mixing + FFN（要求 `D % T == 0`）
- `ffn_only`：跳过 token mixing，仅 FFN
- `none`：identity passthrough

### 2.9 `MultiSeqHyFormerBlock`：核心交互单元

每个 block 串联 4 个步骤：

```python
def forward(q_tokens_list, ns_tokens, seq_tokens_list, seq_padding_masks):
    # ---- Step 1: Sequence Evolution（每个 seq 独立编码） ----
    next_seqs, next_masks = [], []
    for i in range(S):
        seq_i', mask_i' = seq_encoders[i](seq_tokens_list[i], masks[i], rope_cos, rope_sin)
        # LongerEncoder 会输出更短的序列 + 新 mask

    # ---- Step 2: Query Decoding（每个 seq 的 Q 与自己的 seq 做 cross-attn） ----
    decoded_qs = []
    for i in range(S):
        q_i' = cross_attns[i](q_tokens_list[i], next_seqs[i], next_masks[i])

    # ---- Step 3: Token Fusion ----
    combined = cat(decoded_qs + [ns_tokens], dim=1)   # (B, Nq*S + Nns, D)

    # ---- Step 4: Query Boosting（RankMixer） ----
    boosted = self.mixer(combined)                     # (B, Nq*S + Nns, D)

    # ---- Step 5: Split 回 Q 和 NS ----
    next_q_list = [boosted[:, i*Nq:(i+1)*Nq] for i in range(S)]
    next_ns = boosted[:, Nq*S:]

    return next_q_list, next_ns, next_seqs, next_masks
```

**设计精髓**：
1. **隔离**：每个序列有独立的 encoder + cross-attn，避免域间噪声干扰
2. **融合**：所有 Q 和共享 NS 在 RankMixer 中做无参 token mixing → 让信息在 token 间流动
3. **递归**：boosted 输出作为下一 block 的输入，stack `num_hyformer_blocks=2` 层

### 2.10 输出层与分类头

```python
# 所有 block 跑完后
all_q = torch.cat(curr_qs, dim=1)               # (B, Nq*S, D)
output = all_q.view(B, -1)                       # (B, Nq*S*D)
output = output_proj(output)                     # Linear+LN → (B, D)

logits = clsfier(output)                         # (B, action_num=1)
```

**`clsfier` 结构**：
```
Linear(D → D) → LayerNorm → SiLU → Dropout → Linear(D → 1)
```

`action_num=1` 表示二分类，输出单个 logit；如果是多任务（如同时预测多个行为），可以改成 `>1`。

### 2.11 关键约束 d_model % T == 0

```python
T = num_queries * num_sequences + num_ns
if rank_mixer_mode == 'full' and d_model % T != 0:
    raise ValueError(...)
```

**原因**：RankMixerBlock 的 token mixing 需要把 `d_model` 等切成 T 份。

**示例（默认配置）**：
- `d_model=64, num_queries=2, num_sequences=4, num_user_ns=5, num_item_ns=2, has_user_dense=True`
- `num_ns = 5 + 1 + 2 = 8`
- `T = 2×4 + 8 = 16`
- `d_model % T = 64 % 16 = 0` ✓

**`ns_groups.json` 备选配置**：
- `T = 1×4 + (7 + 1 + 4) = 16` → 同样满足

如果不想被这个约束绑架，可以用 `--rank_mixer_mode ffn_only` 或 `none`。

---

## 三、损失设计与训练优化

训练逻辑全部位于 `trainer.py`（`PCVRHyFormerRankingTrainer`）和 `utils.py`（`sigmoid_focal_loss`、`EarlyStopping`）。

### 3.1 BCEWithLogitsLoss（默认）

```python
loss = F.binary_cross_entropy_with_logits(logits, label)
```

**优点**：数值稳定（在 logit 域计算），适合二分类。

**适用场景**：正负样本比例不太悬殊（如 1:10 ~ 1:100）。

### 3.2 Focal Loss（不平衡场景）

来自经典论文 *Focal Loss for Dense Object Detection*：

```python
def sigmoid_focal_loss(logits, targets, alpha=0.1, gamma=2.0, reduction='mean'):
    p = sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)        # 预测对类别的概率
    focal_weight = (1 - p_t) ** gamma                  # 易样本权重 ↓
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)  # 类别再加权
    loss = alpha_t * focal_weight * bce
    return loss.mean()
```

**两个超参**：
| 超参 | 默认 | 作用 |
|---|---|---|
| `alpha` | 0.1 | **正样本权重**。`alpha < 0.5` 表示**降低**正样本权重（适合正样本占多数的场景）。**注意**：这与"原 Focal Loss 论文中正样本稀少时 alpha=0.25 提高正样本权重"方向相反；本项目 PCVR 任务里大概率是正样本（点击后转化）少，但训练数据可能是正样本多，需结合数据比例确认 |
| `gamma` | 2.0 | **聚焦参数**。`gamma > 0` 让易分样本（`p_t` 接近 1）的损失被压低，模型更关注难样本 |

**Focal vs BCE 选择**：
```bash
--loss_type bce        # 默认
--loss_type focal --focal_alpha 0.1 --focal_gamma 2.0
```

### 3.3 双优化器：Adagrad（Sparse）+ AdamW（Dense）

推荐系统的标准配置：

```python
# 区分参数：所有 nn.Embedding 的 weight 算 sparse，其余算 dense
def get_sparse_params(self):
    sparse_ptrs = {m.weight.data_ptr() for m in self.modules() if isinstance(m, nn.Embedding)}
    return [p for p in self.parameters() if p.data_ptr() in sparse_ptrs]

# 优化器
sparse_optimizer = torch.optim.Adagrad(sparse_params, lr=0.05)
dense_optimizer = torch.optim.AdamW(dense_params, lr=1e-4, betas=(0.9, 0.98))
```

| 参数类别 | 优化器 | 学习率 | 理由 |
|---|---|---|---|
| Sparse（Embeddings） | **Adagrad** | 0.05（大） | 每个 ID 独立累积梯度二阶矩，**自适应学习率**，对长尾 id 更友好；稀疏更新效率高 |
| Dense（其他） | **AdamW** | 1e-4（小） | 标准做法，带 weight decay 防过拟合，betas=(0.9, 0.98) 来自 Transformer 调参经验 |

**反向更新**：
```python
self.dense_optimizer.zero_grad()
self.sparse_optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, foreach=False)  # 梯度裁剪
self.dense_optimizer.step()
self.sparse_optimizer.step()
```

> **注意**：`foreach=False` 是为了规避 PyTorch `_foreach_norm` 在某些 tensor shape 下的 CUDA kernel bug。

### 3.4 高基数特征冷重启

来源：快手 *MultiEpoch: Reusing Training Data for CTR Prediction* (arXiv 2305.19531)

**问题**：高基数 id 特征（如 user_id, item_id, vocab > 1e5）在第 2~3 个 epoch 极易过拟合。

**解决方案**：每个 epoch 结束后，把高基数 Embedding **重新 Xavier 初始化**，并保留低基数 Embedding 的 Adagrad 状态。

```python
if epoch >= reinit_sparse_after_epoch and self.sparse_optimizer is not None:
    # 1. 快照 Adagrad state（按 data_ptr 索引）
    old_state = {p.data_ptr(): self.sparse_optimizer.state[p] for p in sparse_params}

    # 2. 重新初始化高基数 Embedding，返回被重置的 ptr 集合
    reinit_ptrs = self.model.reinit_high_cardinality_params(reinit_cardinality_threshold)

    # 3. 重建 Adagrad 优化器
    self.sparse_optimizer = torch.optim.Adagrad(sparse_params, lr=sparse_lr)

    # 4. 恢复低基数 Embedding 的优化器状态
    for p in sparse_params:
        if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
            self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
```

**默认配置**：
- `reinit_sparse_after_epoch=1`：从第 1 个 epoch 末开始重置
- `reinit_cardinality_threshold=0`：默认不重置任何 Embedding
- 实际启用时通常设 `--reinit_cardinality_threshold 10000` 或更大

`reinit_high_cardinality_params` 在 `model.py` 中实现，遍历所有 seq embeddings + user/item NS tokenizer embeddings，时间桶 embedding **始终保留**（因为它低基数且语义稳定）。

### 3.5 评估指标：Binary AUC + LogLoss

```python
def evaluate(self):
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in valid_loader:
            logits, _ = self.model.predict(model_input)  # predict 不开 dropout
            all_logits.append(logits.cpu()); all_labels.append(label.cpu())

    probs = torch.sigmoid(cat(all_logits)).numpy()
    labels = cat(all_labels).numpy()

    # NaN 过滤（防止训练初期梯度爆炸导致 logits=NaN 把整个评估搞崩）
    nan_mask = np.isnan(probs)
    probs, labels = probs[~nan_mask], labels[~nan_mask]

    auc = roc_auc_score(labels, probs) if len(unique(labels)) >= 2 else 0.0
    logloss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels.float()).item()
    return auc, logloss
```

**EarlyStopping 监控指标**：`val_AUC`（越高越好），`patience=5`，验证连续 5 次不上升即停止。

**Best Checkpoint 管理**（`_handle_validation_result`）：
1. 用 `delta` 阈值预判"是否可能是新 best"
2. 是 → 删除旧的 `*.best_model` 目录，重新写 `model.pt` + sidecar files (`schema.json`, `ns_groups.json`, `train_config.json`)
3. 否 → 不动磁盘（避免空目录残留）

Sidecar 设计让 checkpoint **自包含**：评测端只需 `model.pt + schema.json + train_config.json` 即可独立加载。

### 3.6 显存优化与训练稳定性

| 技巧 | 实现位置 | 价值 |
|---|---|---|
| **`emb_skip_threshold`** | `model.py` 各 Tokenizer | 词表 > 1e6 的特征不建 Embedding，零向量替代，省 GB 级显存 |
| **`pin_memory=True` + `non_blocking=True`** | `trainer._batch_to_device` | CPU→GPU 异步拷贝，提升吞吐 |
| **`set_sharing_strategy('file_system')`** | `dataset.py` 顶部 | 多 worker 时避开 `/dev/shm` 上限 |
| **梯度裁剪 max_norm=1.0** | `_train_step` | 防止稀疏特征更新时梯度爆炸 |
| **`nan_to_num` after attention** | `RoPEMultiheadAttention` | 处理"全 padding"序列导致的 softmax NaN |
| **NaN 评估过滤** | `evaluate` | 一旦出现 NaN 不让评估指标也 NaN |
| **预分配 numpy buffer** | `dataset.py` `__init__` | 避免每个 batch 重复分配，CPU 提速 ~30% |

---

## 四、核心超参速查表

```bash
# 默认 RankMixer 配置（run.sh 默认）
python3 train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 --item_ns_tokens 2 \
    --num_queries 2 \
    --d_model 64 --emb_dim 64 \
    --num_hyformer_blocks 2 --num_heads 4 \
    --seq_encoder_type transformer \
    --hidden_mult 4 --dropout_rate 0.01 \
    --rank_mixer_mode full \
    --use_time_buckets \
    --batch_size 256 --lr 1e-4 \
    --sparse_lr 0.05 \
    --loss_type bce \
    --num_workers 8 --buffer_batches 20 \
    --emb_skip_threshold 1000000 \
    --patience 5 --num_epochs 999
```

| 类别 | 超参 | 默认值 | 含义 / 调优建议 |
|---|---|---|---|
| **结构** | `d_model` | 64 | backbone 隐层维度。需满足 `d_model % T == 0`（full 模式） |
| | `emb_dim` | 64 | 单个 Embedding 表的维度（拼接后投影到 d_model） |
| | `num_queries` | 2 | 每个序列域生成的 Q token 数。↑可提升表达，↑会让 T 变大 |
| | `num_hyformer_blocks` | 2 | HyFormer block 堆叠层数 |
| | `num_heads` | 4 | 多头数。需满足 `d_model % num_heads == 0` |
| | `hidden_mult` | 4 | FFN 隐层倍数（4× d_model） |
| | `dropout_rate` | 0.01 | backbone dropout；id 特征 emb dropout = `2×rate` |
| **NS Token** | `ns_tokenizer_type` | rankmixer | `rankmixer`（等切分） / `group`（按 ns_groups.json 分组） |
| | `user_ns_tokens` | 5 | rankmixer 模式下 user 的 NS token 数（0 = 自动） |
| | `item_ns_tokens` | 2 | rankmixer 模式下 item 的 NS token 数（0 = 自动） |
| | `ns_groups_json` | ns_groups.json | group 模式所需的分组配置 |
| **序列** | `seq_encoder_type` | transformer | `swiglu` / `transformer` / `longer` |
| | `seq_max_lens` | seq_a:256,seq_b:256,seq_c:512,seq_d:512 | 各域截断长度 |
| | `seq_top_k` | 50 | LongerEncoder 保留最近 K 个 token |
| | `seq_causal` | False | LongerEncoder 是否使用 causal mask |
| | `use_time_buckets` | True | 是否启用 time bucket embedding（共 65 个桶） |
| | `seq_id_threshold` | 10000 | vocab > 此值视为 id 特征，加额外 dropout |
| **稀疏** | `emb_skip_threshold` | 1000000 | vocab > 此值的特征不建 Embedding，零向量替代 |
| **优化器** | `lr` | 1e-4 | dense 参数的 AdamW lr |
| | `sparse_lr` | 0.05 | sparse 参数的 Adagrad lr |
| | `sparse_weight_decay` | 0.0 | Adagrad weight decay |
| | `reinit_sparse_after_epoch` | 1 | 第 N epoch 末开始冷重启高基数 Embedding |
| | `reinit_cardinality_threshold` | 0 | 冷重启的词表阈值（0 = 不重置任何） |
| **损失** | `loss_type` | bce | `bce` / `focal` |
| | `focal_alpha` | 0.1 | Focal Loss 正样本权重 |
| | `focal_gamma` | 2.0 | Focal Loss 聚焦参数 |
| **数据** | `batch_size` | 256 | 训练 + 验证共用 |
| | `train_ratio` | 1.0 | 训练 RG 取前 N% |
| | `valid_ratio` | 0.1 | 末尾 10% 作为验证集 |
| | `buffer_batches` | 20 | shuffle buffer 大小（按 batch 计） |
| | `num_workers` | 8 | DataLoader worker 数 |
| | `eval_every_n_steps` | 0 | 步级验证间隔（0 = 仅 epoch 末验证） |
| **训练控制** | `num_epochs` | 999 | 最大 epoch（通常被 EarlyStopping 提前终止） |
| | `patience` | 5 | EarlyStopping 容忍次数 |
| **位置编码** | `use_rope` | False | 是否启用 RoPE（仅 transformer/longer encoder 生效） |
| | `rope_base` | 10000.0 | RoPE 基频 |

---

## 五、设计亮点与一句话总结

### 5.1 数据预处理亮点

| 亮点 | 价值 |
|---|---|
| **Schema 驱动的扁平化布局** | 所有特征拼成一个大向量，模型按 (offset, length) 切片，省去字典查找 |
| **预分配 numpy buffer + fused 写入** | CPU 端单次 batch 转换提速 30%+，避免 GC 抖动 |
| **流式 IterableDataset + Row Group 切分** | 内存占用 O(1)，支持 TB 级数据；末尾切验证集保证时序 |
| **非均匀时间分桶（秒/分/时/天/周/月）** | 显式建模兴趣衰减，比线性桶或纯位置编码更符合推荐场景 |
| **OOB 自动 clip + 统计** | 训练数据噪声友好，事后可定位 schema/data 不一致问题 |
| **多级跳过策略**（vs<=0 / emb_skip_threshold） | 极端高基数特征零向量替代，省 GB 级显存 |

### 5.2 模型架构亮点

| 亮点 | 价值 |
|---|---|
| **多源序列异构融合** | 4 个行为域分别编码 + 独立 Q + 共享 NS，比单序列建模更细致 |
| **NS Tokenizer 双形态** | group（语义聚焦）/ rankmixer（数量自由），可灵活切换 |
| **Per-seq Query Generator** | 每个序列域独立派生 Q token，避免不同行为域的语义冲突 |
| **Longer Encoder 自适应模式** | 首块 cross-attn 压缩 + 后续 self-attn，超长序列也能高效处理 |
| **RankMixerBlock 无参 token mixing** | 用 reshape 实现 token 间信息流动，几乎不增加参数 |
| **门控注意力输出**（W_g 初始化 0+1） | 训练初期约等于恒等映射，stack 多 block 也稳定 |
| **Q/K 分离 RoPE** | LongerEncoder 中 Q 来自非连续位置，独立 gather cos/sin 保证语义正确 |
| **Time Bucket Embedding 加性融合** | 不增加 token 数，直接给序列每个位置注入新鲜度信号 |

### 5.3 损失与训练亮点

| 亮点 | 价值 |
|---|---|
| **BCE + Focal 双损失可切换** | 适配不同正负比 |
| **双优化器 Sparse/Dense 分离** | Embedding 用 Adagrad 自适应 lr，其余用 AdamW 防过拟合 |
| **高基数 Embedding 冷重启** | 缓解多 epoch 训练中长尾 id 过拟合（快手 MultiEpoch trick） |
| **梯度裁剪 + NaN 处理三层防护** | clip_grad / nan_to_num / 评估 NaN 过滤，鲁棒性强 |
| **EarlyStopping + Sidecar Checkpoint** | best ckpt 自包含 (model+schema+config)，部署即开即用 |
| **延迟磁盘 IO 的 best 判定** | 用 delta 阈值预判，避免空目录残留 |

### 5.4 一句话总结

> **PCVRHyFormer 是一个面向多源用户行为序列的 Hybrid Transformer 排序模型**：
> 用 **NS Token** 编码静态画像、用 **Per-seq Query + Cross-attention** 提取每个行为域的兴趣、用 **RankMixer** 在所有 token 间做无参信息融合，并辅以 **时间分桶 Embedding**、**双优化器**、**高基数冷重启**、**Focal Loss** 等推荐场景成熟工程技巧，最终在 PCVR 二分类任务上获得稳定的 AUC 提升。

---

## 附：典型 forward 一次的 shape 演变（默认配置）

设：`B=256, d_model=64, num_queries=2, num_sequences=4, num_user_ns=5, num_item_ns=2, has_user_dense=True`

```
输入：
  user_int_feats   : (256, user_int_total_dim)
  user_dense_feats : (256, 918)      # 全部 user dense 拼接
  item_int_feats   : (256, item_int_total_dim)
  seq_a            : (256, n_a, 256) ; seq_a_len: (256,) ; seq_a_time_bucket: (256, 256)
  seq_b/c/d        : 同理（c/d max_len=512）

NS Tokens：
  user_ns          : (256, 5, 64)
  user_dense_tok   : (256, 1, 64)
  item_ns          : (256, 2, 64)
  ns_tokens        : (256, 8, 64)    # num_ns = 5+1+2 = 8

Seq Tokens（4 个域）：
  seq_a tokens     : (256, 256, 64)
  seq_b tokens     : (256, 256, 64)
  seq_c tokens     : (256, 512, 64)
  seq_d tokens     : (256, 512, 64)

Q Tokens：
  q_tokens_list    : 4 × (256, 2, 64)   # 每个序列 num_queries=2 个 Q

RankMixer 输入 (T = 2*4 + 8 = 16)：
  combined         : (256, 16, 64)   # d_model=64 % T=16 == 0 ✓

输出：
  all_q            : (256, 8, 64)    # cat 4 个序列的 Q
  output           : (256, 64)        # output_proj
  logits           : (256, 1)         # clsfier
```

