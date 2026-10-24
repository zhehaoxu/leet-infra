# 解题说明 · Medusa 候选树验证

## 把头输出摊成一棵树

Medusa 的 $K$ 个头各预测未来第 $k$ 个位置的 top-$c$ 候选。把它们按"深度 = 位置偏移"叠起来，就得到一棵候选树：根在当前位置，深度 $d$ 的节点代表"前 $d$ 个投机 token 的某一条具体续写"。节点集合恰好是**所有候选路径的全部前缀**——所以实现时不必显式建树，把每条路径切成 `c[:1], c[:2], ..., c` 丢进一个 `set` 去重即可。

## Typical acceptance 沿父链传播

Medusa 原论文用 **typical acceptance**（[arXiv:2211.17192](https://arxiv.org/abs/2211.17192) §3.2）而非概率比拒绝采样：位置 $i$ 的候选 token $x$ 只要满足 $p_i(x)\ge\varepsilon$ 就被接受，不需要 draft 分布 $q$、也没有残差重采样。关键的树性质是：**一条深度 $d$ 的路径被接受，当且仅当它的整条父链（每一层前缀）都被接受**——因为第 $i$ 层 token 只有在前面 $i-1$ 个 token 都成立时才会真正被目标模型验证。因此逐层判定 `P[i-1][path[i-1]] >= eps` 并对整条路径取 `all` 即可。

## 选最长、并列取字典序

被接受节点可能来自多条不同分支，验证阶段一次性并行算完，最后吐出**最长**的那条（一次生成最多 token）。若两条同深，用**字典序最小**保证结果唯一可判。设选中长度为 $D$，则 bonus token 从 `P[D]`（紧接在吐出的 $D$ 个 token 之后的位置）采样——这正是 tree attention 相对线性 draft 多赚的一个 token。

## 与其它投机解码题的区别

| 方法 | 结构 | 接受判据 |
|---|---|---|
| Leviathan/Chen 拒绝采样 (155) | 单序列 | $\min(1,p/q)$ + 残差重采样 |
| Typical acceptance (本题族, 178) | 单序列 | $p\ge\varepsilon$ |
| **Medusa (本题)** | **候选树** | 逐层 $p\ge\varepsilon$，取最长 |
| Sequoia (176/177) | 通用 token 树 | 多分支残差 / 位置接受率 DP |

## 边界自检

- `check_example`：深度 3 分支 `[1,2,0]` 压过两条深度 2 分支，bonus=`CDF(P[3],0.4)=1`。
- `check_tie_break_lexicographic`：`[1,2]` 与 `[1,3]` 同深都被接受 → 取字典序更小者。
- `check_only_root`：eps 抬高使无候选被接受 → 只吐 bonus（`D=0`，从 `P[0]` 采样）。
- `check_matches_oracle`：随机候选树 + 随机目标分布，与独立纯 Python 前缀枚举实现对拍。
