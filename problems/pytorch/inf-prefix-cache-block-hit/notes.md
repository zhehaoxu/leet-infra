# 解题说明 · Automatic Prefix Caching

## 为什么键要编码整条前缀

KV 值不仅取决于本块 token，还取决于它前面所有 token（注意力是位置与历史的函数）。因此两个内容相同、但前缀不同的 block，其 KV 完全不同，不能共享。真实 vLLM 用 `block_hash = hash(parent_block_hash, tuple(token_ids))` 把父块哈希串进来，等价于本题"累积前缀元组作键"的做法——键天然携带整条前缀身份。

## 命中必须连续

因为复用是"跳过整段前缀的 prefill"，只要第 $j$ 块前缀失配，第 $j+1$ 块即便内容巧合相同，其前缀也不同 → 无法复用。所以从块 0 顺序比对，首个 miss 即可 `break`。用 trie 表示时，命中数 = 沿路径能走通的最大深度。

## 残缺块不计

末尾不满 `block_size` 的部分不单独成块、不入缓存：KV block 的写入是按整块落盘的，半块没有稳定的哈希身份，也无法被别的请求精确对齐复用。故 $n_\text{full}=\lfloor \text{len}/bs\rfloor$。

## 命中率的意义

多轮对话、共享 system prompt、few-shot 模板这类高度重复前缀，命中后可直接省去对应 token 的 prefill，显著降低 TTFT 与算力占用。命中率随共享前缀深度增长而上升，本题的 `[0,2,...]` 正是第二条请求复用了前两条已缓存的 2 个满块。
