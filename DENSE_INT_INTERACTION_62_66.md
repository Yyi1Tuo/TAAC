# 62-66 Dense/Int 交互增强方案说明

## 1. 背景

当前仓库里，`fid 62-66` 这组特征本身就比较特殊：

- 它们在 `user_int_feats` 中存在一份离散化后的 `int_array`
- 它们在 `user_dense_feats` 中又存在一份对应的原始连续值 `float_array`

也就是说，同一个语义信号同时有两种表达：

1. `int` 路径：更稳定，便于 embedding 学习分桶后的粗粒度模式
2. `dense` 路径：信息更细，但容易尺度大、分布偏

仓库原始实现里，这两部分虽然都被用了，但**没有对 62-66 做显式的 pairwise 交互建模**：

- `int` 部分照常进入 `user_ns_tokenizer`
- `dense` 部分只是在所有 `user_dense` 拼接后，统一过一个 `Linear + LN`，生成单个 `user_dense token`

这意味着模型只能依赖后续网络隐式学习：

- 某个 `fid` 的分桶 id 和原始值之间有什么关系
- 哪些 `fid` 的 dense/int 配对更重要
- 这种配对是否应该被强化后再送入后续序列交互

这次改动的目标，就是把这部分信号显式化。

---

## 2. 设计目标

本次改动优先满足以下目标：

1. **保留现有时间特征逻辑不动**
2. **保留当前 `rankmixer` 主干不动**
3. **不改变 `num_ns`，避免影响 `T = num_queries * num_sequences + num_ns`**
4. **不改数据 batch 输出 shape，减少训练/推理不兼容风险**
5. **通过 CLI 开关控制，便于比赛平台直接做 A/B**

因此，本次不是重写 tokenizer，也不是大幅修改数据结构，而是在现有 `user_dense token` 前面增加一条轻量交互支路。

---

## 3. 总体思路

### 3.1 原始路径

原始前向大致可以理解为：

1. `user_int_feats` -> `user_ns_tokenizer` -> `user_ns`
2. `user_dense_feats` -> `user_dense_proj` -> `user_dense_tok`
3. `item_int_feats` -> `item_ns_tokenizer` -> `item_ns`
4. `[user_ns, user_dense_tok, item_ns, ...]` 拼成 `ns_tokens`
5. `ns_tokens` 再去驱动后续 Q token 和多序列交互

### 3.2 新增路径

新增一条仅针对 `fid 62-66` 的交互分支：

1. 从 `user_int_feats` 中抽取 `62-66` 对应的 `int_array`
2. 用现有 tokenizer 的 embedding 逻辑，把每个 fid 的 `int_array` 池化成一个 `int_emb`
3. 从 `user_dense_feats` 中抽取同 fid 的 dense slice
4. 每个 dense slice 单独投影成 `dense_emb`
5. 构造显式 pair 特征：

```text
[int_emb, dense_emb, int_emb * dense_emb, |int_emb - dense_emb|]
```

6. 每个 fid 生成一个 `pair token`
7. 将 `62-66` 的 5 个 `pair token` 做均值池化，得到一个总的 `interaction token`
8. 把这个 `interaction token` 加到原有 `user_dense_tok` 上

最终得到：

```text
enhanced_user_dense_tok = user_dense_tok + interaction_token
```

然后仍然按原路径拼到 `ns_tokens` 中。

---

## 4. 为什么这样设计

### 4.1 为什么不直接改 dataset 输出

比赛环境更重视稳定性。当前 `dataset.py` 输出的字段已经被：

- `trainer.py`
- `train.py`
- `infer.py`
- `model.py`

整条链路消费。

如果为交互分支额外新增 batch 字段，会让：

- 训练和推理构造输入都发生变化
- checkpoint 兼容性变复杂
- 推理端更容易因为版本不一致报错

所以本次方案选择：

- **不新增 batch tensor**
- 只把 schema 信息传进模型
- 在模型内部按 fid->offset/length 查表切 slice

这样改动范围最小。

### 4.2 为什么不增加新的 NS token

新增 token 虽然更“正统”，但会改变：

- `self.num_ns`
- RankMixer 里依赖的 `T`
- `d_model % T == 0` 约束
- 旧配置的可复用性

这会让实验变量变多，不利于快速判断“62-66 交互是否本身有效”。

本次改法选择：

- **不新增 token 数**
- 只增强现有 `user_dense token`

这样收益更容易归因。

### 4.3 为什么交互特征用这四部分

对每个 fid，交互向量由四部分组成：

1. `int_emb`
2. `dense_emb`
3. `int_emb * dense_emb`
4. `abs(int_emb - dense_emb)`

原因如下：

- `int_emb`：保留分桶后的离散语义
- `dense_emb`：保留连续值投影后的细粒度语义
- `乘积`：强调同维共振，类似显式二阶交叉
- `差值`：强调两种表达是否不一致

相比只做简单相加，这种写法更容易让线性层区分：

- 两者一致且同时强
- 两者不一致
- 某一边信号强，另一边弱

---

## 5. 代码改动说明

## 5.1 `model.py`

主要改动在 `PCVRHyFormer`。

### 新增初始化参数

- `user_int_feature_ids`
- `user_dense_feature_ids`
- `user_dense_feature_specs`
- `enable_dense_int_interaction`
- `interaction_hidden_dim`
- `interaction_dropout`
- `interaction_fids`

这些参数的作用：

- 前三者用于建立 `fid -> offset/length` 映射
- 后四者用于控制交互分支是否开启及其容量

### 新增内部缓存

模型初始化时会缓存：

```python
self._user_int_fid_to_idx
self._user_dense_fid_to_spec
```

分别用于：

- 把 `fid` 映射到 `user_int_feature_specs` 的位置
- 把 `fid` 映射到 `user_dense_feats` 中的 `(offset, length)`

### 新增方法 `_init_dense_int_interaction`

这个方法负责初始化交互分支的模块：

- 每个 fid 一个 `dense -> emb_dim` 的投影层
- 每个 fid 一个 `pair_feat -> d_model` 的投影层

如果请求的 fid 不在 int 和 dense 两边同时存在，就自动跳过。

### 新增方法 `_build_dense_int_interaction_token`

这是核心逻辑。

它会：

1. 遍历 `interaction_fids`
2. 从 `user_int_feats` 中取对应离散特征
3. 复用 `self.user_ns_tokenizer._embed_feature(...)` 拿到 `int_emb`
4. 从 `user_dense_feats` 中切出该 fid 的 dense slice
5. 得到 `dense_emb`
6. 拼接四类交互特征
7. 生成 `pair_tok`
8. 对所有 `pair_tok` 做 mean pooling

最后返回一个 `(B, d_model)` 的 `interaction token`。

### 前向注入点

在 `forward()` 和 `predict()` 中，注入位置一致：

```python
user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats))
interaction_tok = self._build_dense_int_interaction_token(...)
if interaction_tok is not None:
    user_dense_tok = user_dense_tok + self.user_dense_interaction_dropout(interaction_tok)
user_dense_tok = self.user_dense_dropout(user_dense_tok).unsqueeze(1)
```

注意这里的策略是：

- 先构造基础 `user_dense_tok`
- 再叠加交互增强
- 最后再进原有的 dropout

这样能保持现有分支结构基本不变。

---

## 5.2 `train.py`

新增了 4 个 CLI 参数：

```bash
--enable_dense_int_interaction
--interaction_hidden_dim
--interaction_dropout
--interaction_fids
```

同时在构建 `model_args` 时，新增了：

- `user_int_feature_ids`
- `user_dense_feature_ids`
- `user_dense_feature_specs`

这保证训练时模型拿得到 schema 级别的位置信息。

默认值设置为：

- `enable_dense_int_interaction = False`
- `interaction_hidden_dim = 64`
- `interaction_dropout = 0.1`
- `interaction_fids = 62,63,64,65,66`

因此旧训练命令不受影响。

---

## 5.3 `infer.py`

推理端做了两类同步：

1. `_FALLBACK_MODEL_CFG` 增加新参数
2. `build_model()` 构造模型时同步传入：
   - `user_int_feature_ids`
   - `user_dense_feature_ids`
   - `user_dense_feature_specs`

这样可以保证：

- 新 checkpoint 可以正确恢复交互分支
- 旧 checkpoint 在默认关闭时也可继续推理

---

## 5.4 `run.sh`

新增了 3 组实验示例：

- `E0` baseline
- `E1` 主方案
- `E2` 增强方案

方便直接复制去平台跑。

---

## 6. 关键实现细节

### 6.1 int 特征如何取 embedding

本次没有另写一套 `int_array` embedding 逻辑，而是复用：

```python
self.user_ns_tokenizer._embed_feature(fid_idx, user_int_feats, None)
```

这样做的好处：

- 与主干 `user_ns_tokenizer` 的 embedding 方式完全一致
- 单值 / 多值特征逻辑统一
- `padding`、`mean pooling` 规则一致

风险是它依赖 tokenizer 的内部私有方法，但当前仓库结构稳定，这种复用是可接受的工程折中。

### 6.2 dense slice 如何处理

每个 fid 的 dense 长度不同：

- `62` 最大长度 5
- `63` 最大长度 11
- `64` 最大长度 18
- `65` 最大长度 49
- `66` 最大长度 66

因此不能共享同一个 dense 投影层，否则输入维度对不上。

本次实现是：

- 每个 fid 单独一个 `Linear(length -> emb_dim) + LayerNorm`

这样虽然参数略多一些，但逻辑清晰且最稳。

### 6.3 为什么对 5 个 fid 取 mean pooling

这一层的目标是生成一个“总的交互增强信号”，而不是额外扩展 token 数。

候选方式有：

- 直接拼接再投影
- attention 聚合
- mean pooling

本次选择 `mean pooling`，因为：

- 不改 token 数
- 参数最少
- 实验归因简单
- 更适合先验证“这条交互分支是否有效”

如果后续验证有效，再考虑更强的聚合器。

---

## 7. 兼容性与风险控制

### 7.1 兼容性

本次改动保持了以下兼容性：

- 默认关闭时，旧训练逻辑不变
- 默认关闭时，旧推理逻辑不变
- `num_ns` 不变
- `T` 不变
- 不需要改 `dataset` batch key

### 7.2 风险点

主要风险有 4 个。

#### 风险 1：交互分支过强，导致过拟合

因为 `62-66` 同时用了：

- 原本的 int 路径
- 原本的 dense 路径
- 现在新增的 pair 路径

有可能变成对这组特征过度强化。

应对方式：

- 保持默认关闭
- 用 `interaction_dropout`
- 先跑小实验矩阵，而不是直接全量采用

#### 风险 2：dense 数值尺度差异较大

`62-66` 的原始值范围很大，从几十到几百万不等。

虽然当前实现通过 `Linear + LayerNorm` 做了投影和归一化，但如果后续发现训练不稳定，可以继续考虑：

- 对 dense 原值先做 `log1p`
- 对非零值做标准化
- 对超长尾值做 clip

本次先不做，是为了避免一次改动里混入太多变量。

#### 风险 3：复用 `_embed_feature` 是内部耦合

当前实现直接调用了 tokenizer 的内部方法。如果后续 tokenizer 改实现，交互分支也要一起看。

短期内这是可以接受的，因为：

- 逻辑复用减少了重复代码
- 与主干 embedding 行为严格一致

#### 风险 4：收益未必来自“正确交互”，也可能只是增大了容量

这也是为什么实验里要保留：

- baseline
- 主方案
- 更大 hidden_dim 的增强方案

否则很难区分：

- 是交互形式本身有效
- 还是只是因为参数量变大

---

## 8. 实验建议

由于比赛不能本地跑，建议直接在平台跑以下 3 组。

### E0：Baseline

```bash
bash run.sh
```

作用：

- 作为当前主干对照

### E1：主方案

```bash
bash run.sh \
  --enable_dense_int_interaction \
  --interaction_hidden_dim 64 \
  --interaction_dropout 0.1
```

作用：

- 验证最小风险交互分支是否带来稳定提升

### E2：增强方案

```bash
bash run.sh \
  --enable_dense_int_interaction \
  --interaction_hidden_dim 128 \
  --interaction_dropout 0.05
```

作用：

- 如果 E1 提升不明显，判断是否是交互分支容量不够

---

## 9. 如何解读实验结果

建议主要看：

- 平台主指标 AUC
- 平台 logloss
- 最佳 step 是否前移
- 是否更早出现过拟合

### 情况 1：E1 优于 E0

说明：

- `62-66` 的 dense/int 显式交互是有效的

后续可以继续：

- 精调 `interaction_dropout`
- 尝试更优聚合方式

### 情况 2：E1 不明显，E2 优于 E1

说明：

- 交互方向可能是对的
- 但当前分支容量偏小

后续可尝试：

- 更大的 `interaction_hidden_dim`
- 更强的聚合器

### 情况 3：E1、E2 都不如 E0

说明可能有两种：

1. 这组交互对当前任务帮助有限
2. 交互分支方式不对，或者重复建模过强

后续再考虑更激进方案：

- 把 `62-66` 并入 `user_dense_as_int` 路径
- 新增 interaction token，而不是只加到 `user_dense_tok`
- 改成 attention 聚合而不是 mean pooling

---

## 10. 后续可扩展方向

如果这版有收益，后面可以继续演进。

### 方向 1：从 `62-66` 扩展到 `89-91`

因为 `89-91` 也同时存在于 int 和 dense 两条路径中。

### 方向 2：把 mean pooling 改成 attention pooling

即让模型自己学习：

- 哪个 fid 更重要
- 不同样本下该关注哪一组 pair

### 方向 3：把交互结果作为独立 token

当前只是增强 `user_dense_tok`。

如果后面想让这条信号更显式地参与 Q token 生成，可以把它扩成新的 NS token，但这会改变 `num_ns`，需要重新检查超参数约束。

### 方向 4：加入 dense 侧预处理

例如：

- `log1p`
- z-score
- 非零值归一化
- clipping

尤其适合 `62-66` 这种长尾连续值。

---

## 11. 本次改动总结

这次方案本质上是在现有主干不动的前提下，对 `fid 62-66` 增加一条显式 pairwise 交互支路：

- 保留原始 `int` 路径
- 保留原始 `dense` 路径
- 新增 `dense/int` 交互增强
- 不改 token 数
- 不改时间特征
- 不改 batch 输出结构
- 通过开关控制，适合直接比赛平台验证

这是一个偏稳妥、工程风险低、实验归因清晰的版本。

如果它带来提升，就说明你队友提到的“`pair_tokenizer + 时间特征` 方向”至少在 `62-66` 这组共享 fid 上是有价值的；如果没有提升，也能较清楚地知道问题不在“是否显式交互”，而可能在“交互形式是否足够强”。
