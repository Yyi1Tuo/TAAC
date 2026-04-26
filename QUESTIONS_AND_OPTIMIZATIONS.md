# 学习过程中的疑问 & 优化点记录

> 用于记录在学习 PCVRHyFormer 项目过程中产生的疑问、改进想法、待调研的优化方向。
> 格式：每条记录包含 **背景 / 疑问 / 可能的优化方向 / 状态**。

---

## Q1：NS Tokenizer 的"直接 cat → 切分"是否最优？能否优化？

**提出时间**：2026-04-26
**相关代码**：`model.py` 中 `RankMixerNSTokenizer.forward`

### 背景
当前 `RankMixerNSTokenizer` 的处理流程是：
1. 把所有 user_int 特征的 Embedding **简单拼接** 成一个长向量 `(B, total_emb_dim)`
2. **均匀切分** 成 `num_ns_tokens` 块
3. 每块独立 `Linear(chunk_dim → d_model)` + LN + SiLU 投影成一个 NS Token

```python
# 核心代码片段
cat_emb = torch.cat(all_embs, dim=-1)               # (B, total_emb_dim)
if self._pad_size > 0:
    cat_emb = F.pad(cat_emb, (0, self._pad_size))   # 补齐到能整除
chunks = cat_emb.split(self.chunk_dim, dim=-1)      # 等分
tokens = [F.silu(proj(chunk)) for chunk, proj in zip(chunks, self.token_projs)]
```

### 疑问 / 潜在问题
1. **切分边界是"无语义"的**：单纯按位置切，可能把同一个特征的 64 维向量切到两个不同的 chunk 里，破坏特征语义完整性。
2. **特征拼接顺序敏感**：拼接顺序由 `groups` 决定，顺序变了，每个 token 看到的特征也变了，可能影响稳定性。
3. **每个 token 的"信息含量"不可控**：可能某个 chunk 全是低基数特征（信息少），某个 chunk 全是高基数 id（信息密度高），分布不均。
4. **没有显式的特征交叉建模**：直接 Linear 投影是线性操作，特征之间的高阶交叉只能依赖后续 RankMixer 的 token mixing，但 token mixing 也是无参的 reshape，表达力有限。

### 可能的优化方向（待调研）
| 方向 | 思路 | 预期收益 | 实现难度 |
|---|---|---|---|
| **A. 显式分组投影**（GroupNSTokenizer） | 用 `ns_groups.json` 按业务语义分组，每组单独投影成 1 个 token | 语义清晰、可解释性强 | ⭐ 已有实现，切换即可 |
| **B. Self-Attention 聚合** | 把每个 fid 的 Embedding 看成一个 token，用 Self-Attn 自动聚合成 K 个 token | 自动学特征交互 | ⭐⭐⭐ 中等 |
| **C. Cross-Attention with learnable queries** | 用 K 个可学习 Query 对所有 fid embedding 做 cross-attn，类似 Perceiver | 数量自由 + 自动加权 | ⭐⭐⭐⭐ 较高 |
| **D. 特征交叉网络**（DCN / FM 风格） | 在投影前先做显式 2 阶交叉 | 捕捉特征 pairwise 交互 | ⭐⭐⭐ 中等 |
| **E. Mixture-of-Experts 投影** | 不同 chunk 用不同的 expert 投影头 | 增强特定语义子空间 | ⭐⭐⭐⭐ 较高 |


### 推荐优先验证
- **A**（最稳健）：直接切到 GroupNSTokenizer 做对照实验
- **B 或 C**（最有潜力）：Self-Attn / Cross-Attn 聚合，可能是新 SOTA 方向

### 状态
- [ ] 待对照实验：A vs 当前 RankMixer
- [ ] 待调研：B / C 在推荐场景的相关工作

---
