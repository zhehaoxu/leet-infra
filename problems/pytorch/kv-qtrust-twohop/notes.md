# 解题说明 · Q-Trust 两跳打分

## 一句话

一跳用最后 `W` 个 query 的注意力给每个 key 打分；把这份分数当作**其它 query 的可信度**，再拿它加权做第二跳；请求段在权重上恒为 1、且永不被剔除。

## 两跳到底在算什么

记 \(G=H/KV\)，\(A^{\text{g}}_q(i)=\sum_{h\in\text{group}}A_h(q,i)\)。一跳的定义就是 \(s(i)=\sum_{q\in W}A^{\text{g}}_q(i)\)，于是

\[
\text{score}(i)=\frac{1}{Z}\Bigl[s(i)+\sum_{q\notin W}s(q)\,A^{\text{g}}_q(i)\Bigr],\qquad Z=\sum_{q\notin W}s(q)+|W|
\]

第一项是请求自己的注意力（被 1 的用户数加权后仍按 \(s\) 归一），第二项是「请求所信任的那些 query」再贡献一次。上游注释里 \(\sum_q s(q)=W\cdot G\) 说的就是这件事。注意这个正比关系**只在没有 query 被剔除时成立**：剔除后 \(w\) 被置 0 却不再归一化，等式右边会多出被丢掉的那些项。

## 五个易错口径

1. **请求段原始权重是 1**，不是 \(s(q)\)：上游 `raw[:, N - W:] = 1.0` 就写在归一化之前。
2. **阈值是 `tau / N`**，分母是序列总长，不是非请求 query 数、也不是 `window`；比较用 `>=`。
3. **请求段永不剔除**：`keep[:, N - W:] = True` 是在阈值比较**之后**覆盖，所以 `tau` 再大也只剔除文档 query。
4. **`kept_query` 是并集**：每个 KV 头算自己的 `w`，但一个 query 的注意力行是组内所有 query 头共享的，所以只要头上有一票保留就得算（上游 `fast.py` 的 `keep.any(dim=0)`）。
5. **同分取小下标**：`torch.topk` 不定义同分顺序，示例与测试都构造了全同分局面。

## 判分点

- `first_hop` 是组内求和，且 \(\sum_j s(j)=W\cdot G\) 恒成立（每行注意力质量为 1）；
- `query_weight` 归一化后 \(\sum=1\)（无剔除时），被剔除的位置恰好为 0；
- `score` 的因果性：没有 \(q\ge i\) 的存活 query 时 `score(i)` 必须为 0；`selected` 后 `recent` 位恒为尾部、竞标名额只来自 `[0, N-recent)`、每行升序无重复。

## 边界

- `budget = recent`：没有竞标名额，`selected` 只剩尾部；`budget >= N` 直接报错（上游 `select_keys` 在 `N <= budget` 时返回 `None`，也就是不驱逐）。
- `window = 1` 时一跳退化成「最后一行的组内注意力和」；`window = N` 时退化成整列求和。
- `tau = 0` 是 dense 路径：所有 query 都算，`Σw = 1`。
- 全部非请求 query 被剔除时 `Σw = n_req / Σ_j raw_w(j) < 1`，`score` 不是重新归一化的结果。

## 复杂度

一跳 \(O(H\,W\,N\,D)\)，二跳 \(O(H\,|Q_{\text{kept}}|\,N\,D)\)，对比全量 \(O(H N^2 D)\)。选择性计算省的就是 \(|Q_{\text{kept}}|/N\) 这一项；上游把它揉进 FlashAttention-2 的 LSE（`qtrust/kernels.py`），本题只考打分口径。

## 出处

`Dirac-Notation/A2SF` 的 `qtrust/`（Q-Trust 的纯 PyTorch 参考实现；仓库没有 README、代码与注释里也没有 arXiv 编号，故只引仓库与代码路径）：

- `qtrust/reference.py` L1-27：整段 docstring 就是规格说明；
- L31-40 `_attention_rows`：`cols > rows` 的因果掩码 + softmax；
- L43-50 `first_hop`：`A.view(KV, H // KV, W, N).sum(dim=(1, 2))`；
- L53-73 `query_weights`：L63-64 `raw[:, N - W:] = 1.0`、L67 归一化、L70 `keep = w >= tau / N`、L71 请求段强制保留、L72 `w = w * keep`；
- L76-90 `second_hop`：按 block 走 `einsum("kq,kgqn->kn")`，整块权重为 0 就跳过；
- L100-109 `select_keys`：L104 `N <= budget` 返回 `None`、L106 `head = N - recent`、L107 在 `score[:, :head]` 上 `topk(budget - recent)`、L108-109 拼上 `arange(head, N)`；
- `qtrust/fast.py` L17-19 与 `kvcompress/scorers/twohop.py` L72-80：`keep.any(dim=0)` 得到真正要算的 query 行。

**本题自己的补充**（上游没写、题面显式钉死）：请求段用显式 `request_mask` 而不是「最后 W 个」；`kept_query` 以并集形式返回；同分取小下标；形状与超参走 `ValueError` 校验。上游同仓库的 `kvcompress/` 里是 AdaSelector（Ada-KV 的再实现），与本仓库已有的 `kv-adakv-headwise` 撞车，本题只走 `qtrust/` 的两跳路径。

## 相关题目

- `kv-quest-page-bound`：page 粒度的注意力上界选页，完全不看 query 权重；本题是 token 粒度、且分数来自两跳传播。
- `kv-snapkv-pooling`：只有一跳（观测窗内的真实注意力），本题的第二跳才是重点。
- `kv-adakv-headwise`：跨头分预算，属 `kvcompress/` 那条线，与本题无关。
