# 解题说明 · MorphKV Island Fusion

## 回源依据

- 论文：MorphKV，[arXiv:2503.00979](https://arxiv.org/abs/2503.00979)。
- 仓库：`ghadiaravi13/MorphKV`（main，抓取时 tree sha `de62fc8c…`）。
- 主逻辑：`morphkv/models/patch_llama.py` 的 `LlamaAttentionMorph.morphkv_hierarchical_cache`（第 266-405 行）与 `LlamaAttentionMorph.forward` 里的调用点（第 477-503 行，`'unimp' in morph_type` 才走层级路径）。
- 缓存装配：`morphkv/morph_cache.py` 的 `DynamicCache.fuse_kv_hierarchical`（第 143-182 行，`torch.cat([fused_keys, imp_keys, window_keys], dim=2)`）与 `DynamicCache.cleanup`（第 106-133 行，按 mask `index_select` 剔除）。

关键片段（原样摘录）：

```python
past_attn = scores_per_kv[:, :, -(self.WIN_SIZE + 1):-1, :-(self.WIN_SIZE + 1)]
importance = nn.functional.softmax(past_attn / self.fuse_temperature, dim=-1).mean(dim=2)
past_val_norms = past_key_value.value_cache[self.layer_idx][:, :, :num_past, :].norm(dim=-1)
importance = importance * past_val_norms
num_important = int(self.MAX_CAPACITY * self.imp_budget)
num_fused_slots = self.MAX_CAPACITY - self.WIN_SIZE - 1 - num_important
island_ids_all = torch.cumsum(is_important, dim=-1)
weights = unimp_importance / island_mass.gather(2, unimp_island_ids).clamp(min=1e-8)
island_scores = island_mass / island_counts.clamp(min=1)
```

`imp_budget=0.5`、`fuse_temperature=1.0` 分别来自 `config.morphkv.get('imp_budget', 0.5)`（第 218 行）与 `float(config.morphkv.get('fuse_temperature', 1.0))`（第 215 行），`LongBench/pred_single.py` 的 `-ib/--imp_budget`、`-ft/--fuse_temperature` 默认值同样是 0.5 / 1.0。

## 上游差异（照抄会踩坑的地方）

1. **importance 还乘了 value 范数**。题面口述只到 `softmax(past_attn/T).mean(heads)`，但源码第 300-301 行把 `importance` 再乘 `past_val_norms = ||v||_2`（`past` 部分）。本题保留这一步，并允许用 `value_norms` 显式传入以避免「范数依赖数值大小」带来的测试噪声。
2. **上游的 `-1` 是当前 token**。`num_fused_slots = MAX_CAPACITY - WIN_SIZE - 1 - num_important`，那个 `1` 与 `past_attn[..., -(WIN_SIZE+1):-1, ...]` 的切法都留给正在生成的 token。本题是纯压缩算子，没有当前 token，故 `n_fused = max_capacity - window_size - n_budget`，窗口也只到 `keys[:, P:, :]`。
3. **GQA 分组**。上游先 `scores.view(bs, key_heads, -1, q, p).max(dim=2)` 把 query 组压成 per-KV-head，再取窗口 query。本题 `attn` 直接按 KV head 给出，分组不在题内。
4. **并列名次上游不可控**。源码用 `topk(..., sorted=False)` 再 `sort()`，同分谁入选取决于实现。本题把 tie-break 写死：重要位与岛屿都取更小的下标/岛号，选完再升序排列。
5. **空岛的处理方式不同**。上游把 `island_counts == 0` 的分数置 `-inf`、布局补零槽，再用 `zero_pad_offset` 的 `-inf` 在注意力里屏蔽。本题不返回 attention offset，改为用 `island_counts == 0` / `island_idx == -1` 把同样的信息暴露给调用方。
6. **累加精度**。上游在模型 dtype（bf16/fp16）里 `scatter_add_`；本题统一 fp64 累加再回写原 dtype，保证 CPU 确定性。
7. **层级路径不是仓库默认**。`forward` 只有 `'unimp' in morph_type` 才走 `morphkv_hierarchical_cache`，而 README/`pred_single.py` 的默认 `morph_type` 是 `max_fused`（走 `morphkv_mask` 的整段淘汰）。也就是说岛屿融合是 MorphKV 的「hierarchical/unimp」配置，而非开箱默认；本题按该函数本体出题。
8. `cleanup` 的 `index_select` 重建 cache、`cache_size/len` 记账、`fusion_done` 只融一次等工程细节都没有进题。

## 与既有题的边界

- `kv-snapkv-pooling`：窗内 query 对 past 的注意力**按 W 求均值再 avg_pool1d 平滑**，尾部 window 用 `max+1` 强留；只产出分数，不动 K/V。
- `kv-merging-cos-nearest`：把最低分槽位按**余弦/最近邻**合进 dense 区的一个槽位，是「任意两槽配对」。
- `kv-wkvd-least-squares-merge`：合并权重由**最小二乘**解出，而不是重要性加权。
- `kv-simlayerkv`：按层做**二值 keep/evict** 判定，不做 token 级融合。
- 本题：top-K 单例 + **连续岛**内 importance 加权平均（post-RoPE K/V 同步融合）+ 按**岛均分**选 top-K，无相似度检索、无最小二乘、无逐层阈值。

## 为可判而做的简化

- 模型前向被拆掉：`attn`（窗口 query 的 logits）、`keys/values`（post-RoPE KV）、`max_capacity/window_size/imp_budget/fuse_temperature/value_norms` 全部显式入参，不 import transformers、不依赖 cache 对象。
- 不返回 `attn_logit_offsets` / `zero_pad_offset`，也不做 GQA 的 `repeat_kv`；这些是推理框架侧的记账，与算子数值无关。
- 返回 `important_idx / island_idx / island_counts` 三个元数据，使布局可被逐位校验（否则补零槽与真实零向量无法区分）。
