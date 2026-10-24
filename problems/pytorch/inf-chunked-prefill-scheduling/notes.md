# 解题说明 · Chunked Prefill

## 为什么要分块

prefill 是 compute-bound（一次算整段 prompt），decode 是 memory-bandwidth-bound（每步 1 token）。让一条长 prefill 独占一步，会拉高同批 decode 请求 token 间的时延（ITL）。Chunked prefill 把长 prefill 打散成"每步一小口"，与 decode 混在同一批里跑，使 ITL 更平稳、GPU 也能填满算力预算。

## 预算分配次序

题面把优先级定为「decode 先占、剩余给 prefill」：$\text{avail}=B-n_\text{dec}$。这对应 vLLM 里 decode token 一定被调度、prefill 用余量的策略。当 $n_\text{dec}\ge B$（decode 已铺满预算）时该 prompt 本步完全进不来，`avail<=0` 直接返回空列表——这是最容易被忽略的边界。

## 与整模型调度的差别

真实调度器每步的 $n_\text{dec}$ 会随请求完成而增减，且多个 waiting prefill 会竞争同一份 `avail`。本题只锁定"单条 prompt 在恒定 decode 占用下的切块序列"，把动态部分剥离以保证判题确定、公式纯粹。chunk 大小只依赖 $\text{avail}$，元素和恒为 `prompt_len`。
