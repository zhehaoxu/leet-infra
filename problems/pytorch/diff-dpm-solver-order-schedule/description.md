# DPMSolverMultistep 的多步阶数与终步降阶调度

## 题目描述

DPM-Solver++ 是「多步」采样器：每一步可以复用前几步的模型输出，用二阶或三阶公式推进一步，阶数越高单步精度越好但需要更长的历史。历史还没攒够时只能退回低阶，最后一步则通常被强制压回一阶（因为再往前推已经走到 \(t=0\)，高阶项的差分没有意义）。

`DPMSolverMultistepScheduler.step` 里的这段调度纯粹是整数记账：它不碰张量，只决定「第 \(i\) 步用几阶」，以及维护一个 `lower_order_nums` 计数器。本题只实现这层调度，返回每步的实际阶数与逐步的计数器轨迹。

## 规则

记 \(N = \text{num\_inference\_steps}\)，`order` = `solver_order` \(\in \{1,2,3\}\)。

**1. 两个终步标志**（第 \(i\) 步，\(0 \le i < N\)）：

\[ \text{floor\_final} = (i = N-1) \wedge \big(\text{euler\_at\_final} \vee (\text{lower\_order\_final} \wedge N < 15) \vee \text{final\_sigmas\_type} = \text{"zero"}\big) \]

\[ \text{second\_final} = (i = N-2) \wedge \text{lower\_order\_final} \wedge N < 15 \]

**2. 选阶**。`low` 是进入该步时的 `lower_order_nums`（初值 0）：

- 若 `order == 1` 或 `low < 1` 或 `floor_final`，用 **1 阶**；
- 否则若 `order == 2` 或 `low < 2` 或 `second_final`，用 **2 阶**；
- 否则用 **3 阶**。

**3. 计数器**：退出该步时，只要 `low < order` 就 `low += 1`。注意降阶的终步也会让计数器自增——上游是在分支之后无条件增量，而不是只在「真用了高阶」时增量。

**4. 参数校验**：

- `"deis"` 被静默改写成 `"dpmsolver++"`（上游的别名兼容）；
- `algorithm_type` 属于 `{"dpmsolver", "sde-dpmsolver"}` 时，`final_sigmas_type` 不能是 `"zero"`，否则抛 `ValueError`。

返回 `(orders, counters)`：`orders` 是长度 \(N\) 的整数列表，`counters` 是长度 \(N+1\) 的整数列表，`counters[0] = 0`，`counters[i+1]` 是走完第 \(i\) 步之后的计数。

## 示例

```
solve(5, solver_order=3)              -> orders=[1, 2, 3, 2, 1]  counters=[0, 1, 2, 3, 3, 3]
solve(4)                              -> orders=[1, 2, 2, 1]     counters=[0, 1, 2, 2, 2]
solve(3, solver_order=1)              -> orders=[1, 1, 1]        counters=[0, 1, 1, 1]
solve(20, solver_order=3, lower_order_final=False)
                                      -> orders=[1, 2] + [3]*17 + [1]
solve(20, solver_order=3, lower_order_final=False, final_sigmas_type="sigma_min")
                                      -> 最后一步仍是 3 阶（floor_final 为假）
solve(14, solver_order=3, final_sigmas_type="sigma_min")
                                      -> 末两步 [2, 1]
solve(15, solver_order=3, final_sigmas_type="sigma_min")
                                      -> 末两步 [3, 3]（N < 15 不再成立）
```

第一行里 `N=5 < 15`，所以 `i=4` 是 `floor_final`（降为 1 阶）、`i=3` 是 `second_final`（降为 2 阶）；`i=0` 因 `low=0<1` 只能 1 阶，`i=1` 因 `low=1<2` 只能 2 阶，`i=2` 才第一次用满 3 阶。

## 约束

- `num_inference_steps` 必须是整数（`bool` 不算）且 \(\ge 1\)；`solver_order` 必须是 \(1/2/3\) 的整数。
- `lower_order_final` / `euler_at_final` 必须是 `bool`（传 `1` 或 `None` 都抛 `ValueError`）。
- `final_sigmas_type` 只接受 `"zero"` / `"sigma_min"`；`algorithm_type` 只接受上表四种取值（含 `"deis"` 别名）。
- `algorithm_type` 为噪声口径（`dpmsolver` / `sde-dpmsolver`）且 `final_sigmas_type="zero"` 时抛 `ValueError`。
- 返回值中的元素必须是 python `int`，不能是 `bool`。
