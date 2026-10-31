# 解题说明 · SCOPE 解码期窗口预算与不连续驱逐

## 一句话

两个整型累加器 + 两道串行的门 + 一次按分数降序的 topk：预算每 `delta` 个 token 才长 1，`discontinuous` 再叠一层「攒 `delta` 个 token 才动一次」的突发门，前缀和尾部不参与竞争。

## 为什么是这套账

上游 `update_kv_in_decoding` 每次调用只处理**一层**的 KV，所以 `current_decoding_step`、`jump_step` 都是「每层调用 +1」的类变量（`SnapKVCluster.current_decoding_step` L438；同一组计数器与预算式在 `PyramidKVCluster` / `H2OKVCluster` / `ALLKVCluster` 里逐行重复，见 L190 / L769 / L1019）。一个解码 token 会恰好触发 `num_hidden_layers` 次调用，于是

```
floor(((t * L + l) / (delta * L))) == floor(t / delta)      for l < L
```

预算与突发计数器在同一个 token 的 `L` 次调用里取值恒定，本题就把它们折叠成「每 token 推进 `L` 格」。这条折叠不是自证：判分用的 oracle 真把 `L` 层展开逐调用模拟，只有上面这个恒等式成立时两者才同解。`l < L` 的边界也刚好安全——`t / delta` 的小数部分最大是 `(delta - 1) / delta`，加上 `l / (delta * L) < 1 / delta` 仍到不了 1。

`delta` 不是自由超参：`run_longgenbench.py` L195 把它注释掉了，改成从输出长度反推

```
delta = (output_max_len - decoding_recent_size) // (decoding_window_size - decoding_recent_size)
```

含义是「预算每 `delta` 个 token 长 1，正好在生成结束时从 `rec` 长到 `decoding_window_size`」。`decoding_window_size` 在 `adaptive` / `discontinuous` 里除了这一行**根本不出现**，所以题面里这两个分支对 `decoding_window_size` 不敏感是对的。

## 四个易错口径

- **计数器按层调用走，不按 token 走**。写成每 token +1，`dw` 会每 token +1，`discontinuous` 的 `evict_steps` 变成每步都动。
- **两道门是串行的**：`jump_step` 只在 `k_len >= mcp + dw` 的调用里推进（上游 L588 的 `if` 在前、L590 的 `elif` 在后）。预算门卡住的那些步既不驱逐也不攒 `jump`，所以突发周期会整体后移——`check_gates_are_in_series` 钉的就是这点。
- **`jump_step` 清零要等满 `L` 次驱逐调用**（L594-597）。因为 `jump_step` 只会取值 `0, L, 2L, ...`，`delta * L` 是 `L` 的整数倍，「半个 token 内跨过阈值」不可能发生，折叠后的 `jump = 0` 一步清零才等价。
- **`keep = dw - rec` 可能为 0**：`adaptive` 的首段预算就是 `rec`，此时门一过就驱逐，但整段候选区被清空，长度直接掉到 `mcp + rec`。漏判这个分支会以为「没驱逐」。

## 本题自己的口径

- 分数由入参 `attn [T, H, P+T]` 直接给，**按绝对位置索引**；上游是 `q @ kᵀ` 再 softmax（L527-528）后按**槽位序号**索引。softmax 单调，所以只要顺序不变，本题给的概率值与上游真实概率不必对应；不在缓存里的位置列直接忽略。
- 上游三个动作里只留调度：预填充的 `pooling` + 聚类 topk（L474-508）与 `kv-snapkv-pooling` 重复，剥掉；KV 张量重建（`gather` / `cat`）用位置列表模拟代替。
- `init_positions` 允许无序、允许位置大于 `mcp`：真实流程里预填充压缩已经把 prompt 缓存打乱，冻结的是「前 `mcp` 个槽位」。
- 同分按候选区先后取，是**本题定的口径**——上游用 `torch.topk`，其并列打破不属公开契约。
- 上游 `__init__` L452 无条件 `assert decoding_window_size - decoding_recent_size > 0`，但 `decoding_window_size` 在 `adaptive` / `discontinuous` 里压根没用到。本题只在 `slide` 分支保留这条约束，其余分支放过。
- 类级计数器跨样本、跨轮次都不会自动归零（要显式 `reset()`）。本题按「一次生成长度」建模，`cds` / `jump` 都从 0 起。

## 判分点

4 键 int64：`window_sizes [T]`、`cache_lens [T]`、`evict_steps [E]`、`kept_positions [H, 最终长度]`。`kept_positions` 逐元素比对，选中段保持分数降序，不排回位置升序；`evict_steps` 是**步号**不是槽位号，`"None"` 时是空张量。

## 复杂度

`O(T * H * (P + T))` 排序 + 模拟，判题规模 `T <= 400`、`H <= 8`、`P <= 512`。

## 出处

- 仓库 `Linking-ai/SCOPE`（awesome list 里 `[Code]` 指向的组织，亲见），`model/kv_utils.py`：`SnapKVCluster` L432，类级计数器 L438-440，构造断言 L452，`update_kv` L474-508，`update_kv_in_decoding` L510-624——`'None'` L518、`'slide'` L520-550（门 L524、softmax 求和 L529、`topk` L537）、`'adaptive'` L551-581（预算 L553、`+= 1` L554、门 L556、`topk` L569）、`'discontinuous'` L583-623（预算 L585、`+= 1` L586、门 L588、`jump_step <` L590、`jump_step += 1` L591、`jump_layer` L594-597、`topk` L609）、`ValueError('Decoding metric not supported')` L624。
- 同一段调度在 `PyramidKVCluster`（L24，块体 L147-262）逐行重复，只有类名前缀不同；`H2OKVCluster`（L627，预算式 L769 / 突发门 L806）与 `ALLKVCluster`（L928，预算式 L1019 / 突发门 L1056）同样如此，但 `H2OKVCluster` 的 `'slide'` 分支把 `topk` 作用在整段 `attn_cache` 上（L726），`PyramidInferCluster`（L265）根本没有这套预算式——本题取 `SnapKVCluster` 这条路径。
- `run_longgenbench.py`：默认 `--max_capacity_prompts 512`（L259）、`--decoding_window_size 1024`（L262）、`--decoding_recent_size 128`（L263），`--delta` 参数在 L194 被注释掉，L195 改为反推；`decoding_window_size` / `decoding_recent_size` 在 L192-193 逐层写入 attn config。
- 论文标题按 README L5 与 Citation 段（L187-191）是 *SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation*，bibkey `wu2024scope`；arXiv Atom 接口（`export.arxiv.org/api/query?id_list=2412.13649`）里 `arxiv:comment` 字段写「ACL 2025」，README 徽章同标 2412.13649。`discontinuous` 是代码里的分支名，README 只写「Three Decoding Strategies」，没有给出对应的英文术语表。
- 对拍证据：把真实 `SnapKVCluster` 实例化 `L` 份（每层一份、各自维护自己的缓存），用 `head_dim` 的第 4 个通道编进绝对位置、同一个 token 的所有层共用一份 query，13 组配置 × 4 个分支的 `evict_steps` / `window_sizes` / `cache_lens` / 逐槽位 `kept_positions` 全部与本题 oracle 一致（`ALL OK`）。

## 相关题目

`kv-snapkv-pooling`（被剥掉的预填充段）、`kv-pyramid-layer-budget`（同一份调度在 PyramidKV 里的逐层预算）、`kv-chunk-independent-topk`（另一种按分数取 topk 的解码期裁剪）、`kv-homer-chunk-merge`（合并代替驱逐）。
