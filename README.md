# PCVRHyFormer 项目零基础学习指南

> **从"什么是推荐系统"开始，一步一步搞懂这个项目的所有细节**
> 适合人群：刚接触深度学习/推荐系统的学生、想理解工业级排序模型的工程师
> 配套文档：[`PROJECT_ANALYSIS.md`](./PROJECT_ANALYSIS.md)（更紧凑的技术参考）

---

## 📚 学习路线图

```
🌱 基础认知
  第 1 讲：项目要解决什么问题？（PCVR、二分类）
  第 2 讲：数据长什么样？（三类特征详解）
  第 3 讲：候选集 item 有哪些特征？

🧠 模型核心
  第 4 讲：Embedding（离散 ID 如何变成向量）
  第 5 讲：Token 三大类（NS / Seq / Q）
  第 6 讲：注意力机制 Attention（Q / K / V）
  第 7 讲：HyFormer Block（模型的"思考单元"）
  第 8 讲：RankMixerBlock（无参 token mixing）

🎯 训练优化
  第 9 讲：损失函数（BCE / Focal Loss）
  第 10 讲：训练优化技巧（双优化器 / 冷重启 / EarlyStopping）
```

---

## 第 1 讲：项目要解决什么问题？

### 1.1 一个真实场景

想象你在刷抖音/小红书：
1. 系统给你推了一个视频 → 你**点击**了它（"曝光后点击" CTR）
2. 你看完后**点赞、收藏、关注、下单** → 这叫"**转化**"（Conversion）

> **PCVR (Post-Click Conversion Rate, 点击后转化率)**：在用户**已经点击**的前提下，预测他后续是否会发生"转化行为"的概率。

### 1.2 数学描述

模型要学一个函数：

```
f(用户特征, 物料特征, 用户历史行为) → 转化概率 p ∈ [0, 1]
```

- `p = 0.9` → 大概率会转化，应该多推
- `p = 0.05` → 大概率不会转化，少推

### 1.3 为什么是"二分类"？

每条训练样本只有两种结果：
- **label = 1**：用户后来转化了 ✅
- **label = 0**：用户没转化 ❌

模型输出一个 `logit`（任意实数），过 `sigmoid` 变成 `[0, 1]` 的概率。

```python
labels = (label_type == 2).astype(int64)   # label_type==2 表示转化
loss = F.binary_cross_entropy_with_logits(logits, labels)
```

### 1.4 推荐系统的特殊难点

普通分类（猫狗分类）只看一张图片，但推荐系统有 **3 种完全不同形态**的输入：

| 输入类型 | 例子 | 难点 |
|---|---|---|
| **离散 ID 特征** | user_id=12345, category=8 | 取值非常多（百万级），不能 one-hot |
| **稠密向量** | 用户预训练 embedding (918 维浮点) | 已是连续值，要会用 |
| **行为序列** | 最近 256 个点击过的 item ID | 变长、超长、要捕捉时序 |

**PCVRHyFormer 的核心目标**：把这 3 种输入统一编码、相互交互，输出一个概率。

---

## 第 2 讲：数据长什么样？三类特征详解

### 2.1 总览：三类特征 + 元信息

```
┌─────────────────────────────────────────────────────────────────┐
│ 一条样本 = 一次（用户 × 物料 × 时刻）的曝光事件                  │
├─────────────────────────────────────────────────────────────────┤
│  ① 离散 ID 特征 (int)        ─►  user_int + item_int            │
│  ② 稠密向量特征 (dense)      ─►  user_dense (item_dense 当前为空) │
│  ③ 行为序列特征 (seq)        ─►  seq_a + seq_b + seq_c + seq_d  │
│  ④ 元信息                    ─►  timestamp + label_type          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 第一类：离散 ID 特征（int）

**离散** = 取值是一个**有限的、可枚举的集合**，每个取值是一个 **ID 编号**（整数），ID 之间没有大小关系。

| 特征 | 取值 | 是不是离散 ID？ |
|---|---|---|
| 用户性别 | 0=男, 1=女, 2=未知 | ✅ 是 |
| 用户城市 | 1=北京, 2=上海, ..., 350=拉萨 | ✅ 是 |
| 物料类目 | 1=美妆, 2=数码, ..., 5000=母婴 | ✅ 是 |
| 用户年龄 | 25 岁 | ❌ 不是（25 比 24 大，是连续值） |

> ⚠️ **关键**：`user_id=12345` 和 `user_id=12346` **没有任何相似关系**，只是两个不同的"标签"。

#### 项目里的离散特征（来自 `ns_groups.json`）

```json
"user_ns_groups": {
  "U1": [1, 15],                                          // 2 个特征
  "U2": [48, 49, 89, 90, 91],                            // 5 个特征
  "U3": [80],                                            // 1 个特征
  "U4": [51, 52, 53, 54, 86],                            // 5 个特征
  "U5": [82, 92, 93],                                    // 3 个特征
  "U6": [50, 60, 94, ..., 109],                          // 18 个特征
  "U7": [3, 4, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66]  // 12 个特征
}
// 一共 46 个 user 离散特征
```

#### 每个离散特征的 3 个属性

```python
# schema.json 里：[fid, vocab_size, dim]
[1, 350, 1]  # fid=1, 词表 350, 单值
[48, 100, 5] # fid=48, 词表 100, 多值（最多 5 个）
```

| 属性 | 含义 |
|---|---|
| **fid** | 特征编号 |
| **vocab_size** | 词表大小（一共多少个不同取值） |
| **dim** | 单值 (=1) 还是多值 (>1) |

#### 数据存储：扁平大向量

```
user_int_feats: shape = (B, user_int_total_dim)
fid=1  (dim=1) → [0:1]
fid=15 (dim=1) → [1:2]
fid=48 (dim=5) → [2:7]   # 多值
...
```

> 💡 **为什么拼成大向量？** 可以**预分配 numpy buffer**，每个 batch 不重新分配，CPU 提速 30%+。

### 2.3 第二类：稠密向量特征（dense）

**稠密** = 取值是连续的浮点数，每一维都有数值意义。

#### 项目里的稠密特征

`ns_groups.json` 注释：
> "All user dense features (fid 61/62/63/64/65/66/87/89/90/91, **total_dim=918**) are concatenated and projected to a single NS token."

- **10 个 user_dense 特征**（fid: 61, 62, 63, 64, 65, 66, 87, 89, 90, 91）
- **总维度 918** 维
- **item_dense 当前为空**

```python
user_dense_feats: shape = (B, 918) float32
```

注意：dense 特征**只有 fid 和 dim**，没有 vocab_size（因为是连续值）。

#### 一个有趣的细节：fid 重复

部分 fid（62/63/64/65/66/89/90/91）**同时出现在 user_int 和 user_dense**：
- **离散部分**：把"用户消费金额"分桶后的 ID
- **稠密部分**：原始浮点数

两边一起用，捕捉**离散化的稳定性**和**连续值的精细度**。这是推荐系统常见技巧。

### 2.4 第三类：行为序列特征（seq）

序列 = **按时间排序的行为列表**。

```
时刻 -10s   -30s   -2min   -1h    -1天    -1周   ...
物料  [item3, item7, item12, item5, item89, ...]
类目  [c1,    c3,   c1,     c2,    c5,     ...]
来源  [s1,    s2,   s1,     s1,    s3,     ...]
```

#### 项目里的 4 个序列域

```bash
--seq_max_lens 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
```

| 域名 | 最大长度 | 推测含义 |
|---|---|---|
| `seq_a` | 256 | 比如近期点击序列 |
| `seq_b` | 256 | 比如曝光序列 |
| `seq_c` | 512 | 比如长期行为 |
| `seq_d` | 512 | 比如另一个域的行为 |

#### 数据形态

```python
seq_a:              shape = (B, n_sideinfo, max_len)  # 例 (256, 3, 256)
                            ↑    ↑           ↑
                          batch  3 个属性   256 个时刻
seq_a_len:          shape = (B,)             # 实际有效长度
seq_a_time_bucket:  shape = (B, max_len)     # 时间分桶
```

每个序列域包含：
- 若干 **sideinfo 列**（物料 ID、类目、来源 ...）
- 1 个**时间戳列** (`ts_fid`)

### 2.5 元信息（meta）

```python
timestamps  = batch.column('timestamp').to_numpy()         # 当前曝光时间
labels      = (batch.column('label_type') == 2).astype(int64)  # 是否转化
user_ids    = batch.column('user_id').to_pylist()          # 训练不用
```

### 2.6 总结：一张表看懂所有特征

| 类别 | 数量 | 数据形态 | 模型如何处理 |
|---|---|---|---|
| **user_int** | ~46 个 | `(B, total_dim)` int64 | NS Tokenizer → 5 个 token |
| **item_int** | ~14 个 | `(B, total_dim)` int64 | NS Tokenizer → 2 个 token |
| **user_dense** | 10 个 (918 维) | `(B, 918)` float32 | Linear(918→64) → 1 个 token |
| **item_dense** | 0 | `(B, 0)` | 跳过 |
| **seq_a/b/c/d** | 4 个域 | `(B, n_side, L)` int64 | Embedding → Seq Encoder → 256/512 个 token |

---

## 第 3 讲：候选集 item 有哪些特征？

项目里 item 的特征**只有两块**：

### 3.1 item_int（候选 item 的离散特征）

来自 `ns_groups.json` 的 `item_ns_groups`：

```json
"item_ns_groups": {
  "I1": [11, 13],
  "I2": [5, 6, 7, 8, 12],
  "I3": [16, 81, 83, 84, 85],
  "I4": [9, 10]
}
```

| 分组 | fid 列表 | 特征数 |
|---|---|---|
| **I1** | 11, 13 | 2 |
| **I2** | 5, 6, 7, 8, 12 | 5 |
| **I3** | 16, 81, 83, 84, 85 | 5 |
| **I4** | 9, 10 | 2 |
| **合计** | | **14 个 item 离散特征** |

涉及的 fid：`{5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 81, 83, 84, 85}`，共 14 个。

**模型怎么用**：
```
item_int_feats → item_ns_tokenizer → 2 个 item NS tokens (B, 2, 64)
```

### 3.2 item_dense（候选 item 的稠密特征）

```python
self.item_dense_schema: FeatureSchema = FeatureSchema()  # 空
'item_dense_feats': torch.zeros(B, 0, dtype=torch.float32)  # (B, 0)
```

**结论**：**item_dense 当前为空**。

### 3.3 为什么 item 特征比 user 少这么多？

| 角度 | 解释 |
|---|---|
| **数据来源限制** | item 侧的可用属性本身就比 user 少 |
| **行为序列补充** | user 行为序列里每个时刻的 sideinfo 记录了用户**历史交互过的 item 属性**，是 item 侧的"扩展" |
| **当前未接入多模态** | 比如 item 的图像/文本 embedding 还没加 |

### 3.4 如果要扩展 item 特征怎么办？

| 方向 | 怎么做 |
|---|---|
| **新增 item_int** | schema.json 加 fid，更新 ns_groups.json |
| **新增 item_dense** | schema.json 加 item_dense 配置，has_item_dense → True |
| **多模态信息** | 接入 BERT 文本向量、ResNet 图像向量等 |
| **历史统计** | item 近 7 天 CTR、过去 N 天的曝光数等 |

---

## 第 4 讲：Embedding —— 离散 ID 如何变成向量

### 4.1 为什么模型不能直接处理 ID？

模型只能处理**数字**，但 `city = 1`（北京）和 `city = 2`（上海）的 "1" 和 "2" 只是**编号**，**不是数值**。

```python
# 错误用法
output = Linear(city_id * weight)  
# 模型会以为"上海(2)是北京(1)的两倍"！
```

### 4.2 朴素方案：one-hot（为什么不行）

```
350 个城市：
北京 (1) → [1, 0, 0, ..., 0]   （350 维）
上海 (2) → [0, 1, 0, ..., 0]
```

**问题**：
- 🚫 **维度爆炸**：user_id 有 1 亿个就完蛋
- 🚫 **语义稀疏**：北京和上海的"距离"和北京和拉萨的一样

### 4.3 解决方案：Embedding（嵌入）

**核心思想**：给每个 ID 学一个**低维稠密向量**。

```
北京 (1) → [0.12, -0.34, 0.56, ...]   （比如 64 维）
上海 (2) → [0.15, -0.30, 0.51, ...]   （和北京相似）
拉萨 (350) → [-0.45, 0.67, ...]      （和北京差距大）
```

这些向量**不是手工设计**，而是模型在训练中**自己学出来**。

### 4.4 Embedding 在代码里就是一张"查表"

```python
nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
#            ↑              ↑          ↑
#         vocab+1         向量维度    padding 行 (ID=0) 永远是 0
```

**工作原理**：
```
              0    1    2   ...  350
              ↓    ↓    ↓        ↓
           ┌────┬────┬────┬─...┬────┐
embedding  │ 0  │京 v│沪 v│    │藏 v│   ← 每一列是 64 维向量
table      │... │... │... │    │... │
(64 × 351) └────┴────┴────┴─...┴────┘
              ↑
           padding 列：永远是 [0, 0, ..., 0]
```

**查表代码**：
```python
emb = nn.Embedding(351, 64, padding_idx=0)
city_id = torch.tensor([1, 2, 350])
city_vec = emb(city_id)  # shape: (3, 64)
```

### 4.5 为什么要 `+1` 和 `padding_idx=0`？

- **约定**：`ID=0` 表示**空值/缺失/padding**
- 真实 ID 范围 `[1, vocab_size]`
- `+1` 给 ID=0 留一行
- `padding_idx=0` 告诉 PyTorch：第 0 行**永远是 0，且不更新**

### 4.6 多值特征：mask + mean pooling

```python
# 用户兴趣标签：[15, 28, 67, 0, 0]（前 3 个是真实标签，后 2 个 padding）
vals = int_feats[:, offset:offset+5]            # (B, 5)
emb_all = emb_layer(vals)                        # (B, 5, 64)
mask = (vals != 0).float().unsqueeze(-1)         # (B, 5, 1)
count = mask.sum(dim=1).clamp(min=1)             # (B, 1)
fid_emb = (emb_all * mask).sum(dim=1) / count    # (B, 64)
```

**直观理解**：用户喜欢"美妆+数码+母婴"，把 3 个标签的 64 维向量平均一下。

### 4.7 Embedding 的"参数量"和"稀疏更新"

```
单表参数 = (vocab+1) × emb_dim
user_id (1 亿) × 64 = 64 亿参数 ❌
```

→ 项目里 `--emb_skip_threshold 1000000` 让超大词表跳过 Embedding，零向量替代。

**稀疏更新**：一个 batch 只查 256 个 user_id，所以 1 亿行表里**只有 256 行有梯度** → 用 **Adagrad**（稀疏友好）。

---

## 第 5 讲：Token 三大类

### 5.1 什么是 Token？

Transformer 的基本操作单位 = **token**（一个向量）。
模型在 token 间做 **attention**，让信息流通。

在推荐系统里没有自然的"词"，所以**手工设计** 3 类 token：

```
┌─────────────────────────────────────────────────────────────────┐
│                    PCVRHyFormer 的 Token 体系                    │
├─────────────────────────────────────────────────────────────────┤
│  ① NS Tokens  (Non-Sequence)         "我是谁、我看的是什么"      │
│      数量：8 个 (5 user + 1 user_dense + 2 item)                │
│  ② Seq Tokens (Sequence)             "我历史上做过什么"          │
│      数量：每域 256 或 512 个                                    │
│  ③ Q Tokens   (Query)                "模型要问的问题"           │
│      数量：num_queries × num_sequences = 2 × 4 = 8              │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 ① NS Tokens

**来源**：
```
user_int (46) ──► user_ns_tokenizer ──► 5 个 user NS tokens
user_dense (918) ──► Linear(918→64) ──► 1 个 user dense token
item_int (14) ──► item_ns_tokenizer ──► 2 个 item NS tokens

cat → ns_tokens (B, 8, 64)
```

#### user_int → 5 个 token 的过程（RankMixer 模式）

```
Step 1: 每个 fid 各自查 Embedding
fid=1 → emb1: (B, 64)
fid=15 → emb2: (B, 64)
... (46 个 fid)

Step 2: 全部拼接
all_embs = cat([emb1, ..., emb46]) → (B, 46*64) = (B, 2944)

Step 3: pad 到能被 5 整除
chunk_dim = ceil(2944/5) = 589
pad_size = 1

Step 4: 切 5 块，每块独立投影
chunk_i (B, 589) ──► Linear(589→64)+LN+SiLU ──► token_i (B, 1, 64)

cat → user_ns: (B, 5, 64)
```

#### NS Tokens 拼接顺序（重要！）

```python
ns_parts = [user_ns]                    # (B, 5, 64)
if has_user_dense:
    ns_parts.append(user_dense_tok)     # (B, 1, 64)
ns_parts.append(item_ns)                # (B, 2, 64)
if has_item_dense:
    ns_parts.append(item_dense_tok)     # (B, 1, 64)
ns_tokens = torch.cat(ns_parts, dim=1)  # (B, 8, 64)
```

固定顺序：`[user_ns, user_dense, item_ns, item_dense]`，训练/推理必须一致。

### 5.3 ② Seq Tokens

```
每个序列域独立处理（以 seq_a 为例）：

Step 1: 每个 sideinfo 列查 Embedding
emb_item     = item_emb(seq_a[:, 0, :])    # (B, 256, 64)
emb_category = cat_emb(seq_a[:, 1, :])     # (B, 256, 64)
emb_source   = src_emb(seq_a[:, 2, :])     # (B, 256, 64)

Step 2: 拼接同一时刻的属性
cat_emb = cat([emb_item, emb_cat, emb_src], dim=-1)  # (B, 256, 192)

Step 3: 投影到 d_model=64
token_emb = GELU(Linear(192→64)(cat_emb))  # (B, 256, 64)

Step 4: 加 time bucket embedding
token_emb = token_emb + time_embedding(time_bucket_ids)
```

最终：`(B, 256, 64)`，每个时刻 1 个 token。

#### Padding Mask

```python
def _make_padding_mask(seq_len, max_len):
    idx = torch.arange(max_len).unsqueeze(0)
    return idx >= seq_len.unsqueeze(1)  # True = padding
```

例如 `seq_len[i] = 87`：前 87 位有效，后 169 位是 padding，attention 自动跳过。

### 5.4 ③ Q Tokens（最容易疑惑的部分）

> **比喻**：Q Token 就像"**虚拟的提问者**"，通过 cross-attention 从 Seq Tokens 里**捞**出有用信息。

#### 为什么需要 Q Token？
- 序列有 256 个 token，太多
- 最终只需 1 个分类输出
- 用少数 Q Token attend 整个序列，**浓缩信息**

#### Q Token 的来源

`MultiSeqQueryGenerator` 为**每个序列域独立生成** `num_queries=2` 个 Q：

```
对每个序列 i：

  Seq_i Tokens (B, L_i, 64)
        │
        ├──► MeanPool (按 mask) → seq_pooled_i: (B, 64)
        │                              │
  ns_flat ─── cat ──► global_info_i: (B, 9*64)
                          │
                          ├──► FFN_{i,1} → q_{i,1}: (B, 64)
                          └──► FFN_{i,2} → q_{i,2}: (B, 64)
                          
                          stack → q_tokens_i: (B, 2, 64)
```

#### 为什么"每个序列独立"？

| 方案 | 问题 |
|---|---|
| 所有序列共享 1 套 Q | 一个 Q 既要"问点击"又要"问曝光"，语义混乱 |
| **每个序列独立 Q（项目采用）** | 每个 Q 有明确"提问对象"，语义清晰 |

#### `num_queries=2` 是什么？

每个序列域生成 **2 个不同的 Q Token**（用 2 个独立的 FFN）：
- `q_1` 可能学到"问长期偏好"
- `q_2` 可能学到"问短期意图"

### 5.5 三大 Token 对比

| 维度 | NS Token | Seq Token | Q Token |
|---|---|---|---|
| **代表什么** | 静态画像 | 历史行为 | 提问探针 |
| **数量** | 8（少而精） | 256~512（多） | 8（少） |
| **如何参与** | RankMixer | 被 Q 通过 cross-attn 查询 | cross-attn + RankMixer |
| **最终用途** | 提供画像信息 | 信息源 | **输出层只用 Q** |

---

## 第 6 讲：注意力机制 Attention

### 6.1 直观例子

> "**那只猫**在阳光下睡得很香，**它**的尾巴一抖一抖的。"

读到 "**它**" 时，你会自动联想到 "**那只猫**"。
- 你拿着 "它" 去**扫描前面的所有词**
- 给每个词打一个**相关性分数**
- 按分数加权平均得到"它"的真正语义

**这就是 Attention**：
> 对每个目标 token，计算它和所有源 token 的相关性，按相关性加权聚合。

### 6.2 Q、K、V 三角色

| 角色 | 全名 | 比喻 |
|---|---|---|
| **Q** | Query | "我想问什么？" |
| **K** | Key | "我能匹配什么问题？" |
| **V** | Value | "选我的话，我能给你什么信息？" |

**核心公式**：
```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
                     ─────────────────────    ───
                          注意力权重           加权聚合
```

### 6.3 一步步拆解

```
Q ∈ (Lq, d_k)，K ∈ (Lk, d_k)，V ∈ (Lk, d_v)

Step 1: 相关性分数
  scores = Q · K^T   shape: (Lq, Lk)

Step 2: 缩放（防止数值爆炸）
  scores = scores / √d_k

Step 3: softmax 归一化
  attn_weights = softmax(scores, dim=-1)   # 每行 = 1

Step 4: 加权聚合
  output = attn_weights · V   shape: (Lq, d_v)
```

### 6.4 手算小例子

- `Q = [1, 0]`
- `K = [[1, 0], [0, 1], [1, 1]]`
- `V = [[10, 20], [30, 40], [50, 60]]`

```
scores = Q · K^T = [1, 0, 1]
softmax([1, 0, 1]) ≈ [0.42, 0.16, 0.42]
output = 0.42×[10,20] + 0.16×[30,40] + 0.42×[50,60] ≈ [30, 40]
```

→ Q 主要从 V1 和 V3 捞信息，V2 贡献少。

### 6.5 Self-Attention vs Cross-Attention

| 类型 | Q 来自 | K/V 来自 | 项目应用 |
|---|---|---|---|
| **Self-Attention** | 自己 | 自己（同一序列） | TransformerEncoder 处理 Seq |
| **Cross-Attention** | A | B | Q Tokens 从 Seq Tokens 捞信息 |

### 6.6 Multi-Head Attention

把 Q/K/V 切成 `num_heads` 份，**每份独立做 attention**，捕捉**多种**关注模式。

```
输入 Q: (B, Lq, 64), num_heads=4

Q = Q.view(B, Lq, 4, 16).transpose(1, 2)  # (B, 4, Lq, 16)
                                          # 4 个独立的 16 维子空间

每个 head 独立做 attention，最后拼接 → (B, Lq, 64)
```

### 6.7 Padding Mask

```python
# True 表示 padding
key_padding_mask: (B, Lk)
sdpa_attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)
out = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_attn_mask)
```

padding 位置的 score → `-∞` → softmax 后权重 = 0 → 自动忽略。

### 6.8 项目特色：门控输出

```python
self.W_g = nn.Linear(d_model, d_model)
nn.init.zeros_(self.W_g.weight)
nn.init.constant_(self.W_g.bias, 1.0)

# forward
G = self.W_g(query)
out = out * torch.sigmoid(G)  # ← 门控
out = self.W_o(out)
```

**初始化技巧**：`W_g.weight=0, bias=1.0` → 训练初期 `sigmoid(1)≈0.73`，相当于"温和的恒等映射"，多 block 堆叠也稳定。

### 6.9 RoPE（旋转位置编码）

```
"我打了你"  vs  "你打了我"
```

如果直接做 attention，由于 attention 对**集合**操作（无视顺序），这两句表示完全一样！

**RoPE 解决方案**：在 Q 和 K 上做**旋转变换**，让点积值只取决于**位置差** `(i-j)`。

> 💡 RoPE 是 LLaMA、Qwen 等大模型的标配，比传统 PE 效果更好。项目里 `--use_rope` 控制开关。

---

## 第 7 讲：HyFormer Block —— 模型的"思考单元"

### 7.1 Block 在模型里的位置

```
NS Tokens, Seq Tokens × 4, Q Tokens × 4
                       │
                       ▼
        ┌──────────────────────────────┐
        │  MultiSeqHyFormerBlock #1    │  ← 第 1 层"思考"
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │  MultiSeqHyFormerBlock #2    │  ← 第 2 层"思考"
        └──────────────┬───────────────┘
                       ▼
              所有 Q Tokens → 输出层
```

默认堆叠 `num_hyformer_blocks=2` 层。

### 7.2 一个 Block 的 4 个步骤

```
Step 1: Sequence Evolution（每个序列独立编码）
Step 2: Query Decoding（Q 从序列里捞信息）
Step 3: Token Fusion（拼接 Q 和 NS）
Step 4: Query Boosting（RankMixer）
Step 5: Split 回 Q 和 NS
```

### 7.3 Step 1: Sequence Evolution

```python
for i in range(S):  # 4 个序列域
    seq_i_new, mask_i_new = self.seq_encoders[i](seq_tokens_list[i], ...)
```

**4 个独立的 encoder**：
```python
self.seq_encoders = nn.ModuleList([
    create_sequence_encoder(...) for _ in range(4)
])
```

**为什么独立？** seq_a（点击）、seq_b（曝光）等域语义不同，共享会互相干扰。

#### 三种 encoder 可选

| 类型 | 结构 | 适用 |
|---|---|---|
| **`swiglu`** | `LN → SwiGLU(4×) → +residual` | 序列短、想省算力 |
| **`transformer`**（默认） | 标准 Pre-LN：`Self-Attn + FFN` | 一般场景 |
| **`longer`** | Top-K 压缩：首块 cross-attn 缩短 + 后续 self-attn | 超长序列 |

### 7.4 Step 2: Query Decoding

```python
for i in range(S):
    decoded_q_i = self.cross_attns[i](
        q_tokens_list[i],  # Q：(B, 2, 64)
        next_seqs[i],      # K/V：(B, L_i, 64)
        next_masks[i],
    )
```

**也是 4 个独立的 cross-attention**，每个序列域专属。

**直观理解**：
- q_a_1 可能学到"专注问短期偏好"，attend 到序列末尾
- q_a_2 可能学到"问长期偏好"，attend 到更早期

#### 关键：`rope_on_q=False`

Q Tokens 是"虚拟提问者"，没有位置概念，只在 K/V 加 RoPE。

### 7.5 Step 3 + 4: Token Fusion + Query Boosting

```python
# Step 3: 拼接所有 Q + NS
combined = torch.cat(decoded_qs + [ns_tokens], dim=1)
# shape: (B, 2*4 + 8, 64) = (B, 16, 64)
#                ↑
#              这就是 T = 16

# Step 4: RankMixerBlock 做 mixing
boosted = self.mixer(combined)
```

### 7.6 Step 5: Split 回原结构

```python
next_q_list = []
offset = 0
for i in range(S):
    next_q_list.append(boosted[:, offset:offset+Nq, :])
    offset += Nq
next_ns = boosted[:, offset:, :]
```

为下一个 Block 准备好输入。

### 7.7 输出层

```python
all_q = torch.cat(curr_qs, dim=1)        # (B, Nq*S, D) = (B, 8, 64)
output = all_q.view(B, -1)               # (B, 512)
output = self.output_proj(output)        # Linear(512→64) + LN → (B, 64)

logits = self.clsfier(output)            # (B, 1)
```

**注意**：输出层**只用 Q Tokens**，不用 NS 和 Seq！

#### 分类头 `clsfier`

```python
nn.Sequential(
    nn.Linear(64, 64),
    nn.LayerNorm(64),
    nn.SiLU(),
    nn.Dropout(0.01),
    nn.Linear(64, 1)
)
```

### 7.8 设计精髓

| 精髓 | 作用 |
|---|---|
| **每序列独立 SeqEncoder** | 不同行为域语义不混 |
| **每序列独立 CrossAttn** | Q 和对应序列精准对齐 |
| **共享 RankMixer** | Q 间、Q 与 NS 全局融合 |
| **多层堆叠** | 递归地深化交互 |
| **输出只用 Q** | Q 已汇聚所有信息 |

---

## 第 8 讲：RankMixerBlock —— 用 reshape 实现 token mixing

### 8.1 为什么需要 RankMixer？

到 Step 3 时，16 个 token（8 个 Q + 8 个 NS）**还没互相交流过**。

朴素方案是上 self-attention，但参数太多。**RankMixer 用几乎无参的 reshape** 实现同样目的。

### 8.2 核心设计：3 步走

```
Step 1: Token Mixing（无参数）
Step 2: Per-token FFN（参数共享）
Step 3: 残差 + Post-LN
```

### 8.3 Step 1: Token Mixing（核心创新）

#### 关键约束：`d_model % T == 0`

```
d_model = 64, T = 16
d_sub = 64 / 16 = 4
```

#### 操作（3 行代码）

```python
def token_mixing(self, Q):
    B, T, D = Q.shape  # (B, 16, 64)
    
    # Step a: 把 D 切成 T 个子空间
    Q_split = Q.view(B, T, self.T, self.d_sub)
    # shape: (B, 16, 16, 4)
    
    # Step b: 交换 token 轴和 subspace 轴
    Q_rewired = Q_split.transpose(1, 2).contiguous()
    # shape: (B, 16, 16, 4)
    
    # Step c: 拼回
    Q_hat = Q_rewired.view(B, T, D)
    # shape: (B, 16, 64)
    
    return Q_hat
```

#### 小例子（T=4, D=8, d_sub=2）

```
原始 Q：
   token 0:  [a₀ a₁ | a₂ a₃ | a₄ a₅ | a₆ a₇]
   token 1:  [b₀ b₁ | b₂ b₃ | b₄ b₅ | b₆ b₇]
   token 2:  [c₀ c₁ | c₂ c₃ | c₄ c₅ | c₆ c₇]
   token 3:  [d₀ d₁ | d₂ d₃ | d₄ d₅ | d₆ d₇]

Mixing 后：
   新 token 0:  [a₀ a₁ | b₀ b₁ | c₀ c₁ | d₀ d₁]
   新 token 1:  [a₂ a₃ | b₂ b₃ | c₂ c₃ | d₂ d₃]
   新 token 2:  [a₄ a₅ | b₄ b₅ | c₄ c₅ | d₄ d₅]
   新 token 3:  [a₆ a₇ | b₆ b₇ | c₆ c₇ | d₆ d₇]
                 ↑       ↑       ↑       ↑
            来自原 t0  来自 t1  来自 t2  来自 t3
```

**神奇之处**：每个新 token 都包含了**所有原 token 的某一部分**，**零参数**完成信息流通。

### 8.4 Step 2: Per-token FFN

```python
x = self.norm(Q_hat)
x = self.fc1(x)        # Linear(64 → 256)
x = F.gelu(x)
x = self.dropout(x)
Q_e = self.fc2(x)      # Linear(256 → 64)
```

`fc1` 和 `fc2` 是**所有 token 共享的同一组参数**。

### 8.5 Step 3: 残差 + Post-LN

```python
Q_boost = self.post_norm(Q + Q_e)   # 残差从原始 Q 出发
```

**为什么从原始 Q 而不是 Q_hat？** 让 FFN 学的 `Q_e` 是"对原始 Q 的精炼"。

### 8.6 三种模式（`--rank_mixer_mode`）

| 模式 | 包含 | 适用 | d_model%T 约束 |
|---|---|---|---|
| **`full`**（默认） | Mixing + FFN + 残差 | 标准 | ✅ 必须 |
| **`ffn_only`** | 仅 FFN + 残差 | 不想被维度约束绑架 | ❌ |
| **`none`** | identity | 消融实验 | ❌ |

### 8.7 为什么用 RankMixer 而不是 self-attention？

| 维度 | Self-Attention | RankMixer |
|---|---|---|
| **参数量** | 4 × d² = 16K | 0（mixing 部分） |
| **复杂度** | O(T² × d) | O(T × d) |
| **表达能力** | 强 | 中（固定 mixing） |
| **适用** | T 大、需要灵活注意力 | T 小（如 16） |

**来源**：字节跳动 2024 论文《**RankMixer: Scaling Up Ranking Models in Industrial Recommenders**》

---

## 第 9 讲：损失函数 —— 模型怎么"学"

### 9.1 训练 = 让 loss 变小

```
1. 前向：模型吃 batch → 输出 logits
2. 算 loss：比较 logits 和 label
3. 反向：loss.backward() 算梯度
4. 更新：optimizer.step() 更新参数
```

### 9.2 logit → 概率

```python
logits = self.clsfier(output)   # (B, 1)，任意实数
prob = sigmoid(logit) = 1 / (1 + exp(-logit))   # 映射到 (0, 1)
```

| logit | prob | 含义 |
|---|---|---|
| 5.7 | ≈ 0.997 | 非常确定是正样本 |
| -2.3 | ≈ 0.091 | 很可能是负样本 |
| 0 | 0.5 | 完全不确定 |

### 9.3 BCE（Binary Cross Entropy）

#### 公式
```
loss_i = - [y · log(p) + (1-y) · log(1-p)]
```

| 真实 y | 模型 p | loss | 含义 |
|---|---|---|---|
| 1 | 0.99 | 0.01 | 预测对了 ✅ |
| 1 | 0.01 | 4.6 | 预测错了 ❌ |
| 0 | 0.99 | 4.6 | 预测错了 ❌ |
| 0 | 0.01 | 0.01 | 预测对了 ✅ |

#### 项目用 `BCEWithLogitsLoss`

```python
loss = F.binary_cross_entropy_with_logits(logits, label)
```

**为什么不手动 sigmoid + BCE？** `BCEWithLogitsLoss` 内部用 **log-sum-exp** 数值稳定技巧，永远不会溢出。

> 💡 **永远不要手动 `log(sigmoid(x))`，会导致 NaN！**

### 9.4 BCE 的局限：类别不平衡

```
1000 条样本：
  正样本：50 条 (5%)
  负样本：950 条 (95%)
```

模型偷懒：无脑预测 `prob=0.05`，平均 loss 已经很小。但**毫无业务价值**。

### 9.5 Focal Loss

> 来源：何恺明团队 2017 论文《**Focal Loss for Dense Object Detection**》

```
FL(p_t) = - α_t · (1 - p_t)^γ · log(p_t)
                  ─────────────  ─────
                   聚焦因子      原 BCE

p_t = p · y + (1 - p) · (1 - y)
α_t = α · y + (1 - α) · (1 - y)
```

#### 两个超参

##### `gamma` (γ)：聚焦因子

让"已经预测对的样本"的 loss **指数级下降**，模型把精力放在难样本上。

| 真实 y | 模型 p | p_t | (1-p_t)^2 | 效果 |
|---|---|---|---|---|
| 1 | 0.99 | 0.99 | 0.0001 | 易样本，loss 被压 |
| 1 | 0.10 | 0.10 | 0.81 | 难样本，几乎保留 |

##### `alpha` (α)：类别权重

```
α_t = α      if y = 1
      1 - α  if y = 0
```

**项目默认 `α=0.1`**：表示**降低**正样本权重（适合本项目正样本占多数的场景）。

#### Focal Loss 代码

```python
def sigmoid_focal_loss(logits, targets, alpha=0.1, gamma=2.0):
    p = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce
    return loss.mean()
```

### 9.6 选择建议

| 场景 | 推荐 |
|---|---|
| 正负较均衡（1:1 ~ 1:10） | **BCE** |
| 严重不平衡（1:100+） | **Focal** |
| 不确定 | 先 BCE 跑 baseline，再用 Focal 对比 |

### 9.7 评估指标：AUC + LogLoss

#### AUC（Area Under ROC Curve）

> 随机抽一对（正样本，负样本），模型预测正样本 prob 高于负样本的**概率**。

- AUC = 0.5：瞎猜
- AUC = 1.0：完美
- 推荐场景里 0.75 ~ 0.85 算不错
- **对类别不平衡鲁棒**，是推荐排序的"黄金指标"

#### LogLoss
就是 BCE 在验证集上的值，衡量**概率校准**好坏。

#### 评估代码

```python
def evaluate(self, epoch=None):
    self.model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            logits, _ = self.model.predict(model_input)  # 不开 dropout
            ...
    
    probs = torch.sigmoid(all_logits).numpy()
    
    # NaN 过滤（防训练初期梯度爆炸）
    nan_mask = np.isnan(probs)
    if nan_mask.any():
        probs, labels = probs[~nan_mask], labels[~nan_mask]
    
    auc = roc_auc_score(labels, probs)
    logloss = F.binary_cross_entropy_with_logits(...)
    
    return auc, logloss
```

---

## 第 10 讲：训练优化技巧

讲 4 个核心技巧：① 双优化器  ② 高基数冷重启  ③ EarlyStopping  ④ 数值稳定性。

### 10.1 双优化器

#### 问题：Embedding 和其他参数的"性格"完全不同

| 参数类别 | 数量级 | 更新频率 |
|---|---|---|
| **Sparse**（Embeddings） | 千万~亿 | 每 batch 只更新少数行 |
| **Dense**（Linear、LN 等） | 几十万 | 每 batch 全部更新 |

#### 解决方案：两类参数用两种优化器

```python
def get_sparse_params(self):
    sparse_params = set()
    for module in self.modules():
        if isinstance(module, nn.Embedding):
            sparse_params.add(module.weight.data_ptr())
    return [p for p in self.parameters() if p.data_ptr() in sparse_params]

def get_dense_params(self):
    sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
    return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]
```

```python
self.sparse_optimizer = torch.optim.Adagrad(
    sparse_params, lr=0.05, weight_decay=0.0)
self.dense_optimizer = torch.optim.AdamW(
    dense_params, lr=1e-4, betas=(0.9, 0.98))
```

#### 为什么 Sparse 用 Adagrad、Dense 用 AdamW？

##### Adagrad（Sparse 适用）

```
v_i = v_i + g_i²            # 累积梯度平方
θ_i = θ_i - lr · g_i / √v_i # 自适应学习率
```

- **每个参数独立的学习率**，对**长尾分布友好**（推荐里 user_id 严重长尾）
- 状态只存一个 `v_i`（AdamW 要存 2 个），**显存省一半**
- 支持稀疏更新

##### AdamW（Dense 适用）

```
m = β₁·m + (1-β₁)·g           # momentum
v = β₂·v + (1-β₂)·g²          # 二阶矩
θ = θ - lr · m / √v - wd·θ    # 带 weight decay
```

- **自适应 + momentum**，稳定性强
- `betas=(0.9, 0.98)` 是 Transformer 调参经验值

#### 学习率差距：0.05 vs 1e-4

| 优化器 | 学习率 | 为什么 |
|---|---|---|
| Adagrad (Sparse) | **0.05** | Embedding 稀疏更新，单次梯度小，需要大 lr |
| AdamW (Dense) | **1e-4** | Dense 参数每 batch 都更新，小 lr 防震荡 |

> 经验法则：Embedding 学习率通常是 Linear 学习率的 100~1000 倍。

#### 反向传播时同时更新

```python
self.dense_optimizer.zero_grad()
self.sparse_optimizer.zero_grad()

loss.backward()
torch.nn.utils.clip_grad_norm_(...)

self.dense_optimizer.step()
self.sparse_optimizer.step()
```

---

### 10.2 高基数特征冷重启（多 epoch 训练神器）

#### 问题：高基数 ID 的多 epoch 过拟合

- user_id 词表 1 亿
- 一个 epoch 每个 ID 只见 1~5 次
- 第 1 epoch 学得不错，第 2 epoch 开始过拟合（同样 ID 反复看，记住训练集噪声）

```
Epoch 1: train AUC=0.80, valid AUC=0.78（健康）
Epoch 2: train AUC=0.85, valid AUC=0.77（过拟合开始）
Epoch 3: train AUC=0.90, valid AUC=0.74（严重过拟合）
```

#### 解决方案：每个 epoch 末"冷重启"高基数 Embedding

> 来源：快手论文《**MultiEpoch: Reusing Training Data for Click-Through Rate Prediction**》（arXiv 2305.19531）

- **低基数特征**（如 city, vocab=350）：保留学到的 Embedding
- **高基数特征**（如 user_id, vocab=1 亿）：每 epoch 末**重新随机初始化**

#### 4 步走

```python
# Step 1: 快照 Adagrad 状态
old_state = {}
for group in self.sparse_optimizer.param_groups:
    for p in group['params']:
        if p.data_ptr() in self.sparse_optimizer.state:
            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

# Step 2: 重置高基数 Embedding
reinit_ptrs = self.model.reinit_high_cardinality_params(
    self.reinit_cardinality_threshold)

# Step 3: 重建 Adagrad 优化器
sparse_params = self.model.get_sparse_params()
self.sparse_optimizer = torch.optim.Adagrad(
    sparse_params, lr=self.sparse_lr,
    weight_decay=self.sparse_weight_decay)

# Step 4: 恢复低基数 Embedding 的优化器状态
for p in sparse_params:
    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
```

#### `reinit_high_cardinality_params` 实现

```python
def reinit_high_cardinality_params(self, cardinality_threshold=10000):
    reinit_ptrs = set()
    for emb_list, vocab_sizes, ... in [...]:
        for i, vs in enumerate(vocab_sizes):
            if int(vs) > cardinality_threshold:
                emb = emb_list[real_idx]
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0  # padding 行恢复零
                reinit_ptrs.add(emb.weight.data_ptr())
    return reinit_ptrs
```

#### 4 步精妙之处

| 步骤 | 目的 |
|---|---|
| Step 1: 快照旧状态 | 保存所有 sparse 参数的 Adagrad `v` |
| Step 2: 重置高基数 | Xavier 初始化 + padding 行强制为 0 |
| Step 3: 重建优化器 | weight 变了，必须重建 Adagrad 实例 |
| Step 4: 恢复低基数状态 | 低基数继续用旧 `v`，避免学习率"重启过猛" |

> 💡 **为什么不一并重置 Adagrad 状态？** Adagrad 的 `v` 越大，学习率越小。如果重置 `v=0`，重启后学习率会变得**特别大**，造成训练不稳。

#### 默认配置

```bash
--reinit_sparse_after_epoch 1
--reinit_cardinality_threshold 0   # 默认不启用
# 实际启用：--reinit_cardinality_threshold 10000
```

---

### 10.3 EarlyStopping + Checkpoint 管理

#### EarlyStopping 核心逻辑

```python
class EarlyStopping:
    def __init__(self, checkpoint_path, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score, model, ...):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
```

#### 典型表现

```
Epoch 1: AUC=0.78 (best)         → 保存 checkpoint
Epoch 2: AUC=0.79 (best)         → 保存
Epoch 3: AUC=0.80 (best)         → 保存
Epoch 4: AUC=0.795 (counter=1)
Epoch 5: AUC=0.792 (counter=2)
...
Epoch 8: AUC=0.789 (counter=5)   → 早停！加载 epoch 3 的 checkpoint
```

**核心价值**：
- **省时间**：避免无意义训练
- **防过拟合**：自动选验证集最佳模型
- **省人工**：不用手动盯指标

#### Checkpoint 自包含设计（项目亮点）

```python
def _write_sidecar_files(self, ckpt_dir):
    shutil.copy2(self.schema_path, ckpt_dir)         # 特征布局
    if self.ns_groups_path:
        shutil.copy2(self.ns_groups_path, ckpt_dir)  # NS 分组
    with open(os.path.join(ckpt_dir, 'train_config.json'), 'w') as f:
        json.dump(self.train_config, f, indent=2)    # 训练超参
```

最终目录：
```
ckpt_dir/
└── global_step12500.layer=2.head=4.hidden=64.best_model/
    ├── model.pt
    ├── schema.json
    ├── ns_groups.json
    └── train_config.json
```

**好处**：推理服务**只需这个目录**，不依赖训练代码或环境。

---

### 10.4 数值稳定性技巧（多层防护）

#### 第一层：梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,
    foreach=False  # 规避 PyTorch CUDA kernel bug
)
```

防止某个 batch 出现极端梯度，把参数瞬间"飞"到 NaN。

#### 第二层：Attention 输出 NaN 处理

```python
out = F.scaled_dot_product_attention(Q, K, V, attn_mask=...)
out = torch.nan_to_num(out, nan=0.0)
```

**为什么会出 NaN？** 某个 query 对应的 key 全是 padding，softmax 输入全是 `-∞`，输出 `0/0 = NaN`。

#### 第三层：评估时 NaN 过滤

```python
nan_mask = np.isnan(probs)
if nan_mask.any():
    probs = probs[~nan_mask]
    labels_np = labels_np[~nan_mask]
```

#### 第四层：BCEWithLogitsLoss 内部稳定

内部用 **log-sum-exp** 技巧避免 `log(0) = -∞` 溢出。

---

### 10.5 显存优化技巧

#### `emb_skip_threshold`：超大词表跳过 Embedding

```bash
--emb_skip_threshold 1000000
# vocab > 100 万的特征不建 Embedding，直接零向量替代
```

**对比**（1 亿词表 × 64 维）：
- 创建：100M × 64 × 4 bytes = **25.6 GB**
- 跳过：**0**

#### `pin_memory + non_blocking`

```python
DataLoader(..., pin_memory=True)
device_batch[k] = v.to(self.device, non_blocking=True)
```

CPU→GPU 拷贝和 GPU 计算并行，吞吐 +10~30%。

#### `set_sharing_strategy('file_system')`

```python
torch.multiprocessing.set_sharing_strategy('file_system')
```

避开 `/dev/shm` 上限。

#### 预分配 numpy buffer

```python
self._buf_user_int = np.zeros((B, total_dim), dtype=np.int64)

def _convert_batch(...):
    user_int = self._buf_user_int[:B]
    user_int[:] = 0  # 复用
```

CPU 提速 30%+。

---

### 10.6 训练流程全景图

```
┌────────────────────────────────────────────────────────────────┐
│  for epoch in range(num_epochs):                                │
│                                                                 │
│    ┌─── 训练阶段 ────────────────────────────────────┐          │
│    │  1. _batch_to_device (pin_memory + non_blocking)│          │
│    │  2. forward → logits                            │          │
│    │  3. loss = BCE / Focal                          │          │
│    │  4. loss.backward()                             │          │
│    │  5. clip_grad_norm_(max_norm=1.0)               │          │
│    │  6. dense_optimizer.step() (AdamW)              │          │
│    │  7. sparse_optimizer.step() (Adagrad)           │          │
│    └─────────────────────────────────────────────────┘          │
│                                                                 │
│    ┌─── 评估阶段 ────────────────────────────────────┐          │
│    │  with torch.no_grad():                          │          │
│    │    遍历 valid_loader，收集 logits + labels      │          │
│    │    NaN 过滤                                     │          │
│    │    val_auc, val_logloss                         │          │
│    └─────────────────────────────────────────────────┘          │
│                                                                 │
│    ┌─── EarlyStopping + Checkpoint ──────────────────┐          │
│    │  if val_auc > best_score + delta:               │          │
│    │    保存 model.pt + sidecar files                │          │
│    │    counter = 0                                  │          │
│    │  else:                                          │          │
│    │    counter += 1                                 │          │
│    │    if counter >= patience: early_stop = True    │          │
│    └─────────────────────────────────────────────────┘          │
│                                                                 │
│    ┌─── 高基数冷重启（epoch 末） ─────────────────────┐          │
│    │  1. 快照 Adagrad state                          │          │
│    │  2. 重置高基数 Embedding                         │          │
│    │  3. 重建 Adagrad 优化器                          │          │
│    │  4. 恢复低基数 Embedding 的优化器状态            │          │
│    └─────────────────────────────────────────────────┘          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 🎯 全篇核心要点速查表

### 数据层

| 要点 | 一句话总结 |
|---|---|
| 三类特征 | int（离散 ID）/ dense（稠密向量）/ seq（行为序列） |
| user_int | 46 个特征，分 7 组 |
| item_int | 14 个特征，分 4 组 |
| user_dense | 10 个特征，total_dim=918 |
| item_dense | 当前为空 |
| seq | 4 个域 (a/b/c/d)，max_len 256 或 512 |

### 模型层

| 要点 | 一句话总结 |
|---|---|
| Embedding | 离散 ID 查表得到稠密向量，padding_idx=0 |
| Token 三大类 | NS（画像）/ Seq（行为）/ Q（提问） |
| Attention | softmax(QK^T/√d)·V，加权聚合 |
| Multi-Head | 切多份独立 attention，捕捉多种关注模式 |
| RoPE | 旋转编码，注入相对位置信息 |
| HyFormer Block | 序列演化 → Q 解码 → 拼接融合 → RankMixer |
| RankMixer | reshape + per-token FFN，无参 token mixing |
| 输出 | 只用 Q Tokens → Linear → sigmoid |

### 训练层

| 要点 | 一句话总结 |
|---|---|
| BCE Loss | -[y·log(p) + (1-y)·log(1-p)] |
| Focal Loss | BCE × (1-p_t)^γ × α_t，处理不平衡和易难样本 |
| BCEWithLogitsLoss | 数值稳定版，永远用它 |
| 双优化器 | Sparse → Adagrad (lr=0.05)，Dense → AdamW (lr=1e-4) |
| 梯度裁剪 | max_norm=1.0 防爆炸 |
| 高基数冷重启 | 每 epoch 末重置高基数 Embedding |
| EarlyStopping | 监控 AUC，patience 不提升即停 |
| Checkpoint 自包含 | model.pt + schema + ns_groups + train_config |
| 评估指标 | AUC（排序）+ LogLoss（校准） |

---

## 🚀 快速上手指南

### 训练

```bash
cd 2026TAAC
bash run.sh
```

### 关键超参（`run.sh`）

```bash
--d_model 64                    # token 维度
--num_hyformer_blocks 2         # HyFormer 层数
--num_heads 4                   # attention 头数
--user_ns_tokens 5              # user NS token 数
--item_ns_tokens 2              # item NS token 数
--num_queries 2                 # 每序列 Q token 数
--seq_max_lens 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
--seq_encoder_type transformer  # swiglu / transformer / longer
--use_rope                      # 启用 RoPE
--rank_mixer_mode full          # full / ffn_only / none
--loss_type bce                 # bce / focal
--sparse_lr 0.05                # Adagrad 学习率
--dense_lr 1e-4                 # AdamW 学习率
--patience 5                    # EarlyStopping 容忍轮数
--emb_skip_threshold 1000000    # 超大词表跳过阈值
```

### 修改建议

| 想干什么 | 改什么 |
|---|---|
| 加大模型 | `--d_model 128 --num_hyformer_blocks 4` |
| 处理超长序列 | `--seq_encoder_type longer` |
| 不平衡严重 | `--loss_type focal --focal_alpha 0.75 --focal_gamma 2.0` |
| 启用冷重启 | `--reinit_cardinality_threshold 10000` |
| 加新 item 特征 | 改 `schema.json` + `ns_groups.json` 的 `item_ns_groups` |

---

## 📁 项目文件说明

| 文件 | 作用 |
|---|---|
| `train.py` | 训练入口，解析参数 + 构建模型 + 启动训练 |
| `dataset.py` | Parquet 数据加载 + batch 转换 |
| `model.py` | 模型定义（PCVRHyFormer 主类 + 各组件） |
| `trainer.py` | 训练循环 + 评估 + EarlyStopping |
| `utils.py` | Focal Loss + EarlyStopping 工具类 |
| `ns_groups.json` | NS 特征分组配置 |
| `run.sh` | 训练启动脚本 |
| `PROJECT_ANALYSIS.md` | 项目深度技术分析 |
| `QUESTIONS_AND_OPTIMIZATIONS.md` | 学习中发现的疑问和优化方向 |
| `README.md` | 本文档（学习指南） |

---

## 🎓 学习成果检查清单

完成本指南后，你应该能：

- [ ] 解释 PCVR 是什么、为什么是二分类问题
- [ ] 区分离散 ID、稠密向量、行为序列三类特征的形态和处理方式
- [ ] 说出 user 和 item 各有哪些特征、各几个 token
- [ ] 解释 Embedding 为什么需要、padding_idx 的作用
- [ ] 说清楚 NS / Seq / Q 三类 token 的来源和用途
- [ ] 用自己的话解释 Q/K/V 的角色和 Attention 公式
- [ ] 说清楚 Self-Attention 和 Cross-Attention 的区别
- [ ] 画出 HyFormer Block 的 4 步骤流程
- [ ] 解释 RankMixer 的 token mixing 是怎么做到无参的
- [ ] 说出 BCE 和 Focal Loss 的区别和适用场景
- [ ] 解释为什么用双优化器，Sparse 和 Dense 各用什么
- [ ] 说清楚高基数冷重启的 4 步流程
- [ ] 解释 EarlyStopping 监控什么指标，patience 是什么意思

---

## 🔬 进阶方向

学完这份指南，你可以继续探索：

1. **跑一遍代码**：`bash run.sh`，对照日志理解每一步
2. **改参数对比**：`--seq_encoder_type swiglu` vs `transformer`
3. **扩展模型**：加新特征、调 token 数、换 NS Tokenizer
4. **看相关论文**：
   - RankMixer (字节, 2024)
   - MultiEpoch (快手, 2023)
   - Focal Loss (FAIR, 2017)
   - RoPE (RoFormer, 2021)
5. **看优化记录**：`QUESTIONS_AND_OPTIMIZATIONS.md`

---

## 💬 致谢

本指南由 AI 辅助整理，感谢你的耐心学习 🌟

如有疑问，欢迎随时讨论！
