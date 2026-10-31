"""参考答案：排序计划 + gather 还原 + reshape/sum 合并，完全照着 grouped_mm 的槽位代数走。

四个容易漏掉的点：
1. 排序必须**稳定**且按 (专家编号, 槽位编号) 升序。并列顺序决定了 `perm`、`inv_perm`、
   `token_rows` 三张表的具体数值；不稳定的排序在同样的输入下给出另一套索引，判题就无法
   写死期望值。真实 kernel 不在乎同专家内部的行序，但题目要考的就是这套索引代数。
2. 哨兵（id == E）**不 clamp 进排序**，让它自然排到尾部：`counts` 只统计 0..E-1，
   `offsets[-1]` 就是有效行数，grouped mm 只需算前 `offsets[-1]` 行。但取 `diag` 时
   索引必须先在界内（`clamp(max=E-1)`），算完之后再把哨兵行整行清零——这正是上游
   「kernel 把哨兵行留在未初始化状态，靠 pre/post mask 兜住」的等价写法。
3. `inv_perm[perm] = arange(S)`，方向不能写反。写反了得到的是 `perm` 自己，还原出来的
   顺序是错的，而 `out` 的形状仍然对得上，只能靠数值对拍发现。
4. 合并用 `unsort -> view(T,k,D) -> sum(1)`，不用 `index_add_`。两条路径的数值差异来自
   归约顺序；reshape+sum 按 j 的原始顺序累加，是确定性的，`index_add_` 在 CUDA 上走
   atomicAdd 会不确定，而且会以输出张量的 dtype 累加。
"""

import torch


def _check(hidden, expert_ids, routing_weights, diag):
    for name, tensor in (("hidden", hidden), ("expert_ids", expert_ids),
                         ("routing_weights", routing_weights), ("diag", diag)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
    if hidden.dtype != torch.float64 or routing_weights.dtype != torch.float64 or diag.dtype != torch.float64:
        raise ValueError("hidden, routing_weights and diag must be float64")
    if expert_ids.dtype != torch.int64:
        raise ValueError("expert_ids must be int64")
    if hidden.dim() != 2 or min(hidden.shape) < 1:
        raise ValueError("hidden must be a non-empty 2-D [T, D] tensor")
    if expert_ids.dim() != 2 or min(expert_ids.shape) < 1:
        raise ValueError("expert_ids must be a non-empty 2-D [T, k] tensor")
    if expert_ids.shape != routing_weights.shape:
        raise ValueError("routing_weights must match expert_ids shape")
    if expert_ids.shape[0] != hidden.shape[0]:
        raise ValueError("expert_ids must have one row per token")
    if diag.dim() != 2 or diag.shape[0] < 1 or diag.shape[1] != hidden.shape[1]:
        raise ValueError("diag must be [E, D] with D matching hidden")
    if int(expert_ids.min()) < 0 or int(expert_ids.max()) > diag.shape[0]:
        raise ValueError("expert ids must be in [0, E] where E is the sentinel")


def solve(hidden, expert_ids, routing_weights, diag):
    _check(hidden, expert_ids, routing_weights, diag)
    num_tokens, hidden_dim = hidden.shape
    top_k = expert_ids.shape[1]
    num_experts = diag.shape[0]
    slots = num_tokens * top_k

    flat_ids = expert_ids.reshape(-1)
    flat_weights = routing_weights.reshape(-1)

    sorted_ids, perm = torch.sort(flat_ids, stable=True)
    token_rows = torch.div(perm, top_k, rounding_mode="floor")
    gather_ids = torch.clamp(sorted_ids, max=num_experts - 1)
    grouped = hidden[token_rows] * diag[gather_ids]
    sentinel = sorted_ids >= num_experts
    if bool(sentinel.any()):
        grouped = grouped.masked_fill(sentinel.unsqueeze(-1), 0.0)

    counts = torch.bincount(flat_ids[flat_ids < num_experts], minlength=num_experts)[:num_experts]
    offsets = torch.cumsum(counts, dim=0)

    weighted = grouped * flat_weights[perm].unsqueeze(-1)
    inv_perm = torch.empty(slots, dtype=torch.int64)
    inv_perm[perm] = torch.arange(slots, dtype=torch.int64)
    out = weighted[inv_perm].reshape(num_tokens, top_k, hidden_dim).sum(dim=1)
    return out, grouped, perm, inv_perm, token_rows, counts, offsets
