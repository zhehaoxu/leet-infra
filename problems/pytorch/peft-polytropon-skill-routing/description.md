# Polytropon 技能清单路由

Polytropon（论文 *Combining Modular Skills in Multitask Learning*）与 Multi-Head Adapter Routing（MHR，*Multi-Head Adapter Routing for Cross-Task Generalization*）已合入 HF PEFT（`peft/tuners/poly`）。核心思想：每层维护一个由 `n_skills` 个 LoRA 组成的「技能清单」，一个按任务索引的路由函数学习混合权重；`n_splits > 1` 时把 LoRA 的秩切成互不相交的块，每块独立路由（MHR）。

## 参数与形状

- `poly_lora_A`：(n_splits, n_skills, in//n_splits, r)
- `poly_lora_B`：(n_splits, n_skills, r, out//n_splits)
- router `module_logits`：(n_tasks, n_splits·n_skills)，`uniform_(-1e-3, 1e-3)`

## 初始化（RNG 域）

A/B 逐 (skill, split) 块初始化，**循环 skill 在外、split 在内**：

- A：`kaiming_uniform_` 作用于 `(r, d_in)` 张量（fan_in = d_in = in//n_splits，a=√5 → bound = 1/√d_in），**转置后**写入 `A[split, skill]`；
- B：`init_weights=True` 时全零（训练起点零增量）；False 时 `kaiming_uniform_` 作用于 `(d_out, r)`（fan_in = r），转置后写入；
- router：所有 A 块抽完后 `uniform_(-1e-3, 1e-3)`。

注意 kaiming 的浮点细节：`a = math.sqrt(5)` 时 `1 + a·a` 不精确等于 6（误差 ~1e-16），bound 因此带同样误差——对拍 RNG 流必须用完全相同的公式，否则 uniform 抽样出现位级差异。

## 路由（推理分支）

```
logits = module_logits[task_ids].view(-1, n_splits, n_skills)
s = sigmoid(logits)
weights = s / (s.sum(-1, keepdim=True) + 1e-12)
```

三个判题敏感点：

- **归一化的是 sigmoid 值，不是 softmax**——与 softmax 数值不同；
- **+1e-12 EPS 护栏**：sigmoid 下溢为 0 时防止 0/0=NaN（可判题：全 `-800` logits 用例下结果必须有限且为零）；
- 越界 task_id 抛 `ValueError`（`>=` 边界）。

训练分支的 `RelaxedBernoulli.rsample` 是随机的，不在判题域。

## 前向组合

```
A = einsum("bqs,qsdr->bqdr", w, poly_lora_A).reshape(bs, in, r)
B = einsum("bqs,qsrd->bqrd", w, poly_lora_B).transpose(1, 2).reshape(bs, r, out)
delta = x.bmm(A).bmm(B) / r        # 缩放严格是 1/r，不是 α/r 或 1/√r
```

reassembly 语义：A 的 reshape 把 split 0 的输入块放在前 `in//n_splits` 列；B 先 transpose(1,2) 再 reshape，输出列按 split-major 排列。恒等式：`A_mixed[:, :in//ns]` 只由 split 0 的加权和决定。

## 参数账

lora = ns·sk·(in//ns)·r + ns·sk·r·(out//ns)；router = n_tasks·ns·sk；`n_splits=1` 退化为普通多 LoRA 清单。

## 判题域

单入口 `solve(mode, **kw)`，五个 mode：`init_params` / `route` / `combine` / `forward_delta` / `param_count`。RNG 域显式 generator；数值锚点 float64。
