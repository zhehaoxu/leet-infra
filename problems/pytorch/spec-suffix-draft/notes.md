# 解题说明 · Suffix/REST 最长后缀检索草稿

## 检索式草稿 = 给上下文找最长"曾见过的尾巴"

REST/Suffix Decoding 不训练 draft 模型，而是维护一个 datastore（历史 prompt→续写、或全局生成 token 流），每次解码拿当前 context 去里面找**最长的、能对上某条记录连续子串的后缀**，然后把那条记录里紧跟其后的一段直接当草稿抄过来。直觉：如果当前尾巴在语料里出现过，它后面大概率还会按同样方式继续。

## 匹配与 tie-break

对 $n$ 从 $\text{len(context)}$ 递减到 $\text{min\_match}$，取 context 长度 $n$ 的后缀 $s$，在每条 entry 里找 $s$ 的首次出现。选优的三重优先级——**更长后缀 > 更早 entry > 更早位置**——保证结果唯一：
- "更长优先"是把最长尾巴的续写当作最可信草稿；一旦某 entry 里已找到不短于当前 $n$ 的匹配，就没必要再看更短的 $n$（可提前 break）；
- 同长时按 entry 扫描顺序，第一条命中的胜出；
- 同 entry 同长时按 $i$ 升序，最早出现处胜出。

草稿长度截到 $\gamma$，且不越过 entry 末尾（匹配点在 entry 结尾时草稿自然为空）。

## 与 Lookahead(179) 的分工

| | 匹配单位 | 输出 | 回退 |
|---|---|---|---|
| Lookahead | 定长窗 $[\text{min\_n},\text{max\_n}]$ 后缀，逐步 | 每步 1 token | 有确定性回退 |
| **Suffix/REST（本题）** | **变长最长后缀，一次** | **$\le\gamma$ 个草稿** | 无（找不到就空） |

## 边界自检

- `check_example`：`[1,2,3]` 三长后缀在 entry0 命中，草稿 `[4,5]`；更短的 `[2,3]` 被忽略。
- `check_no_match`：只有 1 长的尾巴能对上、但低于 `min_match` → 空草稿。
- `check_tie_break_entry`：两条 entry 同长命中 → 取更早 entry 的后继。
- `check_earliest_position`：同 entry 内多处命中 → 取最早位置。
- `check_gamma_truncate`：匹配点接近 entry 末尾 → 草稿被截到剩余长度。
- `check_matches_oracle`：随机 datastore/context/gamma/min_match，与"枚举全部 (n,entry,pos) 候选再取 tuple 最大"的独立实现对拍。
