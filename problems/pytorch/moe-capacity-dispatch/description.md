# MoE 路由：容量分派与 Token Dropping

## 题目描述

真实 MoE 服务里每个专家只能吃下固定数量的 token。**容量分派**给每个专家设上限 $C$，超出的按**位置顺序**丢弃（保留原 batch 里先到的 token）。这是 Shazeer OLNN / GShard 的经典 drop-policy="position"。

## 规则

实现 `solve(logits, k, capacity_factor)`：

1. `probs = softmax(logits, -1)`；每行取 `topk(probs, k)` 得 `top_vals (T,k)` / `top_idx (T,k)`。
2. **renorm** 得 `top_w = top_vals / top_vals.sum(-1, keepdim=True)`。
3. `routing_map (T,E) bool`：`routing_map[t, top_idx[t,j]] = True`。
4. `routing_probs (T,E)`：`routing_probs[t, top_idx[t,j]] = top_w[t,j]`，其余 0。
5. 每专家容量 $C = \lceil T \cdot k / E \cdot \text{capacity\_factor} \rceil$。
6. **Position-based drop**：按 token 索引 $t = 0, 1, \ldots, T-1$ 顺序扫描 `routing_map`，每个专家保留前 $C$ 个 True，其余置 False。
7. `final_map (T,E) bool` = 保留下来的路由掩码；`final_probs (T,E)` = `routing_probs * final_map.float()`；`dropped = int(routing_map.sum()) - int(final_map.sum())`。

返回 `(final_map, final_probs, dropped)`。

## 示例

```
logits = [[2,0,0,0], [1,0,0,0], [0,2,0,0], [0,3,0,0],
          [0,0,2,0], [0,0,1,0], [0,0,0,2], [0,0,0,3]]
k=1, capacity_factor=0.5

C = ceil(8*1/4*0.5) = 1
final_map = [[1,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
             [0,0,1,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]]
dropped = 4
```

每个专家（4 个）都想吃 2 个 token，容量只有 1，故保留第一个（t=0/2/4/6）、丢弃第二个（t=1/3/5/7）。

## 约束

- CPU float32；`final_map.dtype == torch.bool`；tie 按较小专家索引优先。
