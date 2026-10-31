# 解题说明 · Page-EntroKV 的分组 Rényi 熵与页级保留

## 一句话

每个 query head 先算一条 sink 隔离的 Rényi 熵，组内 `softmax(-熵 / tau_g)` 加权把 `r` 个 head 的**原始**注意力池成一条曲线，按 `block_size` 分页取 `max`，再按页做 top-k。

## 为什么是这套账

- GQA 里一块物理 KV 缓冲被组内 `r` 个 query head 共享。README 的说法是：per-head top-k 会让引擎保留各 head 选择的并集，最多膨胀 `r` 倍；而均值池化会把专家 head 的偏好稀释 `(1-eps)/r`。页面粒度则对齐引擎的 `[N_blocks, H_KV, B, D_head]` 物理块，选中的页可以直接映射到块表。
- 熵是「这个 head 有多集中」的代理：分布越尖，熵越低，`softmax(-ent / tau_g)` 给它的权重越大。于是组内由一个（或少几个）专家 head 主导，而不是被同组其它 head 平均掉。
- sink 必须隔离：序列开头几个位置天然吸走大量注意力（attention sink），不先削掉，所有 head 的熵都会被压到相近的高位，权重失去区分度。代码是「先把前 `min(sink_size, T)` 个位置置 0，再对剩余位置重归一化」。
- 页分数取页内最大值：一页里只要有一个 token 被盯上，整页就该留下；配合非负注意力，零填充到 `block_size` 倍数不会改变页 max。

## 四个易错口径

- **熵用隔离行，池化用原始行**。`core.py` L29-31 算出的是削掉 sink 并重归一化后的行，L32/L35 分组与池化用的却是原始 `attn_weights`。两处共用一个行变量，`page_scores` 就会全错——`check_sinks_are_isolated_in_entropy_only` 用一行 `[100, 100, 0.1, 0.2]` 把这点钉死：熵只看后两个位置，页 0 的分数仍然是 100。
- **softmax 在组内做**：`dim=-1` 作用在 `ent.view(B, H_KV, r)` 的 `r` 个 head 上，不是全 head 维。分组是**连续**切分 `h // r`（`view` 的语义），不是交错 `h % H_KV`。
- **`log(sum p^alpha + 1e-12) / (1 - alpha)` 的符号**：`alpha > 1` 时分子为负、分母为负，熵为正；写成 `alpha - 1` 会整体变号。`alpha == 1.0` 是单独分支，走 Shannon 熵；传整数 `1` 也落在这里。
- **先填充再取页 max，`nb` 用 `ceil`**：`pad = (block_size - T % block_size) % block_size`，填充后才 `view(...).max(-1)`；反过来会少一页或者把补的 0 当成新页。

## 本题自己的口径

- 上游 `forward` 只返回页掩码（L44）。`entropies`、`group_weights`、`page_scores`、`retained_tokens` 四个键是**本题自己定的契约**：README 的重量化公式 `w = softmax(-H2_sink / tau_g)` 与 block-max 映射里这些量都在，但代码不返回，导出来是为了把每一步都钉住。
- 平手时选页号小的，是本题定的口径。上游用 `torch.topk`，其并列顺序不属于公开契约（实测等值时返回索引 0，但这是实现细节）。
- 上游不校验任何入参；本题要求 `attn >= 0`（零填充的页 max 中性依赖它）、`tau_g > 0`、`alpha > 0`、`0 <= num_retained_blocks <= nb`，其余非法入参抛 `ValueError`。
- `sink_size > T` 时整行都算 sink，这是上游 `min(self.sink_size, T)`（L28）的写死行为，本题保留；`sink_size == 0` 合法（不隔离）。
- 上游是 `nn.Module` + `@torch.inference_mode()`；本题是纯函数，无状态、无 device 依赖，`attn` 只读不改。
- 全 sink 行归一化后全 0，`0**alpha = 0`、`log(0 + 1e-12)` 有限，熵不会变成 NaN；这条靠的是上游那个 `clamp(min=1e-12)`，不是额外的特判。

## 判分点

5 键：`entropies [B, H_Q]`、`group_weights [B, H_KV, r]`、`page_scores [B, H_KV, nb]` 三个 float 键按 `atol = 1e-5` 逐元素比；`retained_pages [B, H_KV, nb]`、`retained_tokens [B, H_KV, T]` 两个 int64 键（值是 0/1，不是 bool）精确比。后两个键别压成一维，`page_scores` 也别提前 reduce 成 `[B, H_KV]`。

## 复杂度

`O(B * H_Q * T + B * H_KV * nb * log nb)`，额外空间就是返回的 5 个张量；判题规模 `B <= 4`、`H_Q <= 16`、`H_KV <= 8`、`T <= 512`。

## 出处

- awesome list `HuangOwen/Awesome-LLM-Compression`：`### KV Cache Compression` 段标题在 L2010，本题条目在 L2101，写的是 *Page-EntroKV: Hardware-Aligned, Entropy-Weighted KV-Cache Eviction under Grouped-Query Attention*、`Arxiv 2026`，`[Paper]` 指向 arXiv 2610.03135，`[Code]` 指向 `github.com/bruce12-glitch/PageEntro-KV`（亲见）。
- arXiv 官方 Atom 接口（`export.arxiv.org/api/query?id_list=2610.03135`）返回的标题与上面一致，`published` 为 2026-10-02；仓库 `paper/main.tex` 的 `\title` 在 L101，同一标题。
- 实现：`page_entrokv/core.py`（共 44 行）。`PageEntroKVGroupPooler` L6，`__init__` L7-15（默认 `32/8/16/0.5/2.0/4`），`renyi_entropy` L17-22（`eps = 1e-12` L18，`alpha == 1.0` 分支 L19-20，Rényi 分支 L21-22）；`forward` L24-44：sink 掩码 L27-28、隔离与重归一化 L29-30、熵 L31、按原始 `attn` 分组 L32、熵分组 L33、组内 softmax L34、池化 L35、填充 L36-38、页划分 L39、页 max L40、`topk` L41、掩码与返回 L42-44。
- 仓库的自述状态：README「Pilot-tier status in this repo version」段明说这是 pilot tier；`page_entrokv/kernels/README.md`、`baselines/snapkv_patch.py`、`baselines/vllm_block_manager_patch.py`、`evaluation/run_guardrails.py` 的文件内容都只有一行以 `PENDING:` 开头的说明。本题只照 `core.py` 的代码，不引用任何吞吐/精度结论。
- `page_entrokv/allocator.py::largest_remainder_quotas` 里 `remainders` 恒为 `0.0`、`adjusted` 直接拷贝 `quotas`，实际分配退化成「余数全给前几个层」，且与 `kv-palu-rank-budget` 的预算分配重复，故不单独立题。
- 对拍证据：真实 `PageEntroKVGroupPooler` 与本题参考实现在 200 组随机配置上页掩码完全一致（`B ∈ {1,2}`、`H_KV ∈ {1,2,3}`、组比 `r ∈ {1,2,4,6}`、`T ∈ {5,7,13,16,17,33,40}`、`block_size ∈ {1,2,4,8,16}`、`tau_g ∈ {0.1,0.5,1,3}`、`alpha ∈ {1,1.5,2,3}`、`sink_size ∈ {0,1,4,6}`、`k` 在 `[0, nb]` 内随机）；导出的 `entropies` / `group_weights` / `page_scores` 与逐行照抄 L26-44 的转录版在 `rtol 2e-5` 内一致。另有边界用例：`k = 0`、`k = nb`、`T < block_size`、`T % block_size == 0`、全 sink 行、`T = 1`、组比 1 与 6，全部一致。

## 相关题目

`kv-palu-rank-budget`（同一套 GQA 分组与层间预算）、`kv-snapkv-pooling`（按 head 独立 top-k 的对照）、`kv-quest-page-bound`（另一种页粒度选择）、`kv-homer-chunk-merge`（合并代替驱逐）。
