# KV 块预算下的请求准入与顺延

## 问题

continuous batching 的引擎每一轮都要回答两个问题：这一轮**已经跑着**的请求还能不能再吐一个 token，以及**还在排队**的请求能不能被拉进 running 集合。答案都取决于同一份资源：KV 块的剩余量。

一个请求在 prefill 阶段需要 `ceil(prompt_len / block_size)` 个块；之后每生成一个 token，只有当它跨过块边界时才需要再要一块。块不够时，传统引擎会做抢占（swap 或 recompute，见 `inf-lru-kv-cache-eviction` 与 `inf-paged-kv-block-alloc` 的邻居题）。本题约定**不抢占**：拿不到新块的 decode 只是本步顺延（stall），下一步再试。

另一个容易忽略的点：准入阶段不是「队头装不下就整队停」——那样一个大请求能把整个队列堵死。本题要求扫描整个等待队列，装得下的都拉起来，装不下的跳过继续看后面的。

## 任务

实现 `solve(num_blocks, block_size, prompt_lens, max_new, max_steps)`：

- 输入：`num_blocks` 物理块数；`block_size` 每块 token 数；`prompt_lens` `(N,)` int64；`max_new` `(N,)` int64；`max_steps` 模拟步数上限。
- 每一步先解码、后准入，两阶段内都按请求 id 升序处理：
  - **解码阶段**：对已准入且未完成的请求，计算吐出一个 token 后需要的块数 `ceil((prompt_len + 已生成 + 1) / block_size)`；多出来的块从空闲池取，取不到就顺延一步、本步不生成。
  - **准入阶段**：对未准入的请求，prefill 需要 `ceil(prompt_len / block_size)` 块；够就准入并记下准入步，不够就跳过。
- 请求完成（已生成达到 `max_new`）时立刻归还全部块。
- 输出：`admit_step` `(N,)`（-1 表示从未准入）、`finish_step` `(N,)`（-1 表示未完成）、`stalls` `(N,)`（各请求顺延的总步数）、`peak_used` int64 标量（空闲块最少的时刻对应占用块数）。

## 示例

```
num_blocks = 3, block_size = 4
prompt_lens = [4, 6], max_new = [4, 2]

step 0  准入: r0 需 1 块 -> 准入(占 1)；r1 需 2 块 -> 准入(占 2)  已占 3
step 1  r0 想吐第 1 个 token -> 需 ceil(5/4)=2 块，空 0 块 -> 顺延
        r1 想吐第 1 个 token -> 需 2 块，已持有 2 块 -> 直接吐
step 2  r0 再顺延(共 2 次)
        r1 吐第 2 个 token -> 完成，归还 2 块
step 3  r0 拿到 1 块(占 2)，吐第 1 个 token
step 4..6  r0 依次吐完 4 个 token，step 6 完成

admit_step  = [0, 0]
finish_step = [6, 2]
stalls      = [2, 0]
peak_used   = 3
```

## 约束

- **两阶段顺序固定**：先解码后准入。同一轮里刚完成的请求归还的块，可以让同轮等待中的请求立刻被准入。
- **同阶段内按 id 升序**：谁先拿块直接影响后面谁能拿到，必须确定。
- **不抢占**：拿不到块的 decode 只顺延，不会把别人踢下去（那是另一类策略）。
- **准入不等同于队头阻塞**：装不下就跳过，继续扫描后续请求。
- **`max_new = 0`**：准入即完成，`finish_step` 等于 `admit_step`。
- **`max_steps` 用尽仍没有完成的请求**：`finish_step` 记 `-1`，且它已经累计的顺延次数要照实返回。
