# 序列新增特征总结

本文只总结当前新增到每个序列上的两部分特征：

1. 基于原始时间戳构造的 6 维周期时间特征
2. 基于截断后真实长度构造的 1 维序列长度特征


## 1. 新增特征概览

当前对每个序列域新增了两类辅助特征：

- `time_features`：对序列中每个 item 的原始时间戳做周期编码，得到 6 维特征
- `seq_len_feat`：对该序列域截断后的真实有效长度做归一化，得到 1 维特征

它们最终都会拼接到单个序列 token 的特征维上，再投影到统一的 `d_model`。


## 2. 6 维周期时间特征

### 2.1 输入来源

每个序列 item 都有自己的原始时间戳。数据侧会先把该序列域的时间戳 pad 成：

```text
ts_padded: (B, L)
```

其中：

- `B` 是 batch size
- `L` 是该序列域的截断长度上限
- 每个位置对应一个 item 的原始 unix timestamp
- padding 位置时间戳为 `0`


### 2.2 特征定义

当前时间周期特征一共 6 维，顺序为：

```text
[hour_sin, hour_cos, day_sin, day_cos, week_sin, week_cos]
```

其中三组周期语义分别是：

- `hour`：一天中的第几小时
- `day`：一周中的第几天，也就是 `day-of-week`
- `week`：一个月中的第几周，也就是 `week-of-month`


### 2.3 具体计算方式

#### 2.3.1 hour

```text
hour = 时间戳对应的小时，范围 [0, 23]
hour_angle = hour * 2π / 24
hour_sin = sin(hour_angle)
hour_cos = cos(hour_angle)
```


#### 2.3.2 day

```text
weekday = 一周中的第几天，范围约为 [0, 6]
day_angle = weekday * 2π / 7
day_sin = sin(day_angle)
day_cos = cos(day_angle)
```


#### 2.3.3 week

当前 `week` 使用“月内第几周”的 5 段划分：

- 第 1 周：1-7 日
- 第 2 周：8-14 日
- 第 3 周：15-21 日
- 第 4 周：22-28 日
- 第 5 周：29-31 日

公式为：

```text
day_of_month = 该月中的第几天，从 1 开始
week_of_month = clip((day_of_month - 1) // 7, 0, 4)
week_angle = week_of_month * 2π / 5
week_sin = sin(week_angle)
week_cos = cos(week_angle)
```


### 2.4 padding 处理

时间戳为 `0` 的位置视为 padding，对应 6 维时间周期特征全为 `0`。

最终数据侧输出：

```text
[domain]_time_features: (B, L, 6)
```


### 2.5 模型中的使用方式

模型先对该序列域所有 side info 特征分别做 embedding，再在最后一维拼接。时间周期特征会直接拼到每个 token 的特征维上：

```text
side info concat:
(B, L, S * emb_dim)

+ time_features:
(B, L, S * emb_dim + 6)
```

也就是说，这 6 维是 item-level 特征：

- 序列里的每个 item 都有自己独立的一组时间周期值
- 它们是按 token 逐位置拼接进去的


## 3. 真实序列长度特征

### 3.1 输入来源

对每个序列域，数据侧都会按该域的 `max_len` 做截断：

```text
raw_len = 原始序列长度
use_len = min(raw_len, max_len)
```

最终保留下来的长度是截断后的真实有效长度 `use_len`，并输出为：

```text
[domain]_len: (B,)
```

所以这里的长度特征表示的是：

```text
截断后的真实有效长度
```

不是原始未截断长度。


### 3.2 归一化方式

模型侧会对这个长度做 `log1p` 归一化：

```text
seq_len_norm = clip(log(1 + seq_len) / log(1 + max_len), 0, 1)
```

其中：

- `seq_len` 是截断后的真实有效长度
- `max_len` 是该序列域的最大长度上限


### 3.3 扩展与拼接方式

归一化后的长度是一个标量，但当前实现不是只用一次，而是复制到该序列域的每个 token 上：

```text
seq_len_feat: (B, L, 1)
```

注意这里同一条序列里的所有 token，共享同一个长度值，只是在 `L` 维上复制了一遍。

然后把它继续拼到 token 的特征维上：

```text
(B, L, S * emb_dim + 6)

+ seq_len_feat:
(B, L, S * emb_dim + 6 + 1)
```


## 4. 最终拼接结果

对单个序列域来说，新增特征拼接后的整体流程是：

```text
side info embedding concat:
(B, L, S * emb_dim)

+ 6 维时间周期特征:
(B, L, S * emb_dim + 6)

+ 1 维真实长度特征:
(B, L, S * emb_dim + 7)

Linear -> d_model:
(B, L, d_model)
```


## 5. 一句话总结

当前新增部分只有两项：

- 对每个序列 item 的原始时间戳，构造 `hour / day-of-week / week-of-month` 的 6 维周期特征，并逐 token 拼接
- 对每个序列域截断后的真实有效长度做 `log1p` 归一化，得到 1 维长度特征，再复制到该序列域每个 token 上拼接
