# 详解：解码吞吐的 batch 曲线、KV 读这堵墙与拐点

## 出处

- **大 batch 也救不了 decode**：Pope 等, *Efficiently Scaling Transformer Inference*（MLSys 2022）指出解码是
  访存受限的，吞吐上限由「每 token 要搬多少字节」决定，权重可跨 batch 复用、KV 不可复用；FlexGen
  （Sheng 等, ICML 2023）把同样的字节/FLOP 折成显存与延迟约束下的最优分块。
- **每 token 的 KV 字节** $2\cdot L\cdot n_{kv}\cdot d\cdot T\cdot \text{bytes}$ 沿用题库 `kv-cache-memory`
  （vLLM/PagedAttention 里 KV 按 token 线性增长、每步全量重读）。
- **$2N$ FLOP/token**（矩阵乘近似）沿用 `inference-throughput`；「算强度不低于 ridge 记 compute-bound、
  相等也算」沿用 `roofline-arithmetic`（Williams 等, CACM 2009）。

## 核心与易错点

1. **两项的 batch 依赖不同**：权重读 $W=N\cdot\texttt{weight\_bytes}$ 与 $B$ 无关，KV 读 $B\cdot K$ 严格线性。
   $\text{bytes}/B = W/B + K$ 是**递减**的，所以下降空间只有 $W/B$ 那一项，降到 $K$ 就见底了。
2. **单步时间是两个候选的 max**，不是相加也不是取瓶颈那一项的比例：
   $\max(\text{bytes}/bw,\ B\cdot 2N/peak)$。写成分段函数时**必须和 max 等价**，否则拐点两侧对不上。
3. **瓶颈判定用严格小于**：`is_memory_bound = flops_step < bytes_step*ridge`，ridge $=peak/bw$。
   相等记 compute-bound，与 `roofline-arithmetic` 一致；判题里专门造了 `peak_flops == bandwidth`、
   `context_tokens=0`、`B=1` 的**恰好相等**用例钉这个口径（写成 `<=` 就会翻车）。
4. **上限是两个 cap 的 min**：$peak/(2N)$ 是算力 cap（与 batch 无关），$bw/K$ 是 $B\to\infty$ 时 KV 读的 cap。
   长上下文时后者更小——这就是「7B/A100 在 2048 上下文下最多 7450 tokens/s」的来源；$K=0$ 时只有算力 cap。
5. **拐点的闭式有符号陷阱**：由 $B\cdot 2N\cdot bw\ge peak\cdot(W+B\cdot K)$ 整理成
   $B\,(2N\,bw-K\,peak)\ge peak\,W$。**系数 $\le 0$ 时无解，记 -1**，表示无论 batch 多大都进不了 compute-bound；
   直接除过去会得到负数或反向不等号。
6. **`context_tokens=0` 时拐点与参数量无关**：$B=peak\cdot \texttt{weight\_bytes}/(2\cdot bw)$，A100 fp16 就是
   $312e12*2/(2*2e12)=156$，7B 和 70B 都一样——很反直觉，但正是校验公式有没有写错的地方。
7. **`balance_batch` 是「最小正整数解」，且是物理量**：不要拿 `max_batch` 去截断它，`max_batch` 只服务于
   `min_batch_for_slo`。浮点比值算出的候选要用判据本身校正 ±1（判题里的搜索法就是在验这一点）。
8. **SLO 反解靠单调性**：$\text{tokens\_per\_s}(B)$ 在 memory 段递增、在 compute 段恒等于 $peak/(2N)$，
   整体**单调不减**，所以二分正确；目标高于上限、或 $B$ 被 `max_batch` 卡住时都要返回 -1，
   `target_tokens_per_s=0` 时返回 1（不要返回 0 或 -1）。
9. **`step_time_ms` 是毫秒、吞吐是秒**：混用时相差 1000 倍；`weight_fraction` $=W/(W+BK)$ 是 HBM 读的占比，
   它掉到 0 就是「加 batch 无效」的直接信号（prefix caching / KV 量化就是为了把 $K$ 压小）。
10. **分工**：单算子上限在 `roofline-arithmetic`，纯算力吞吐在 `inference-throughput`，
    KV 容量在 `kv-cache-memory`，MoE 通信在 `distributed-moe-communication-bottleneck`；
    本题只管「权重 + KV 两种字节随 batch 的变化」。

## 对拍口径

参考解：`step_time = max(bytes/bw, flops/peak)`（总量式）、拐点用闭式
`ceil(peak*W/(2N*bw-K*peak))` 再用判据前后各校正一步、`min_batch_for_slo` 用**二分**。判题 `_oracle`
全部换路：吞吐先除以 batch 得「每 token 时间」再取 max 求倒数（与总量式结构不同）、
`weight_fraction` 写成 $1/(1+BK/W)$、`ceiling` 写成 $1/\max(2N/peak, K/bw)$（$K=0$ 时退化为算力 cap，
不需要 `inf` 分支）、拐点用 **1,2,4,… 指数扩张 + 二分搜索**（完全不碰闭式除法）、
`min_batch_for_slo` 用**从 1 起的线性扫描** brute force。另设结构性断言：吞吐随 batch 单调不减且从不超过
`ceiling`、拐点之后恒等于算力 cap 且 step_time 与 batch 成正比、`context_tokens=0` 时拐点恒为 156 且
`weight_fraction=1`、上下文翻倍时 `ceiling` 减半而 step_time 几乎不动、GQA 头数减半时 `ceiling` 翻倍、
SLO 边界（$B$ 达标而 $B-1$ 不达标、`max_batch` 卡住时 -1、目标超过上限时 -1、target=0 时为 1）、
KV 主导时 `balance_batch=-1` 且吞吐逼近但绝不越过 $bw/K$。

## 复杂度

参考解 $O(\log \texttt{max\_batch})$（SLO 二分）＋常数级拐点校正；判题的指数扩张是 $O(\log B)$，
线性扫描是 $O(\texttt{max\_batch})$，用例里 `max_batch` 控制在几千以内，整套几十毫秒~百毫秒完成。
