# 详解：DPMSolverMultistep 阶数调度与终步降阶

## 出处

HuggingFace diffusers 的 `src/diffusers/schedulers/scheduling_dpmsolver_multistep.py`：`step` 里从 `lower_order_final` / `lower_order_second` 两个布尔量到 `if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final` 的三段分支，以及末尾的 `self.lower_order_nums += 1`。公式来自 DPM-Solver++（Lu et al., 2211.01095）的多步高阶更新与 diffusers 对「终步降阶」的经验处理。

## 核心与易错点

1. **`N < 15` 是硬阈值**：上游把「步数少于 15 就开启终步降阶」写死在 `lower_order_final and num_inference_steps < 15` 里。\(N=14\) 与 \(N=15\) 的结果因此在最后两步完全不同（`[2,1]` 对 `[3,3]`），这是本题最容易被想当然跳过的地方——很多人会以为阈值是开区间或按 `solver_order` 缩放。
2. **`final_sigmas_type="zero"` 单独也会触发降阶**：即使 `lower_order_final=False`、`N` 很大，只要终端 σ 是 0，最后一步仍然被压成一阶。原因是走到 σ=0 时 \(x_0\) 与去噪输出的换算会退化，高阶差分项失去定义。
3. **降阶终步也推进计数器**：上游在分支之后无条件执行 `if self.lower_order_nums < self.config.solver_order: self.lower_order_nums += 1`，所以 `counters` 的最后一项常常等于 `solver_order`，而与最后一步实际用了几阶无关。示例里 `N=5, order=3` 的 `orders` 以 `[2,1]` 收尾，但 `counters` 仍是 `[..., 3, 3, 3]`。把增量写进「用了 3 阶」的分支里，会让整个计数器轨迹错位。
4. **`second_final` 依赖用户参数而不是 `floor_final`**：公式里第二个降阶看的是**入参** `lower_order_final` 与 `N < 15`，与 `euler_at_final`、`final_sigmas_type` 无关。所以 `N=5, lower_order_final=False, final_sigmas_type="sigma_min"` 得到 `[1,2,3,3,3]`——只有最后一步被别的条件挡住，倒数第二步并不降。
5. **`euler_at_final` 只影响最后一步**：它进入 `floor_final` 的或运算，对 `second_final` 完全无影响；`N=30` 时开它只把末位从 3 变 1。
6. **`"deis"` 是静默别名**：上游构造函数把 `algorithm_type="deis"` 改写成 `"dpmsolver++"` 而不是报错。本题保留这一行为，因此 `solve(5, solver_order=3, algorithm_type="deis")` 与默认配置逐项相同。
7. **噪声口径禁止 zero 终端**：`dpmsolver` / `sde-dpmsolver` 两个 `algorithm_type` 在构造函数里就拒绝 `final_sigmas_type="zero"`，因为它俩的公式需要显式的 `sigma_min`。所以「噪声算法 + 默认 zero」必须抛 `ValueError`，测试里专门列了这两条。
8. **一阶不是特例而是下界**：`solver_order == 1` 时永远走第一支，`counters` 却仍然从 0 涨到 1 然后停住——`low < order` 恒不成立。`check_counters` 对 `steps/order` 做了全组合扫描，验证 `counters[-1] == min(steps, order)` 且每步增量只可能是 0 或 1。
9. **纯整数、无张量**：这一题的输出是两条 python `int` 列表，所以不需要 float64/float32 的精度约定；需要坚持的是「`bool` 不算整数」，`solver_order=True` 必须报错。

## 对拍口径

判题用一个独立的纯 Python oracle（`_oracle`）重新实现同一套分支，再与参考解逐项比对：

- **示例值**：`N=5/order=3`、`N=4/默认`、`N=3/order=1` 三组序列硬编码，并断言所有元素是 `int` 而非 `bool`；
- **长程轨迹**：`N=20` 全速跑到 `[1,2] + [3]*17 + [1]`，计数器 `[0,1,2] + [3]*18`；
- **终步降阶开关**：`lower_order_final` 开/关在 `N=5` 下给出 `[1,2,3,2,1]` 与 `[1,2,3,3,3]`；`N=20` 关掉后仅靠 `final_sigmas_type="zero"` 仍把末步压成一阶；
- **15 步阈值**：`N=14` 末两步 `[2,1]`，`N=15` 末两步 `[3,3]`；
- **`euler_at_final`**：`N=30` 下只改末位；
- **计数器不变量**：7 种步数 × 3 种阶数扫描，验证长度、起点、终点、每步增量 \(\in \{0,1\}\)、上界 `<= order`；
- **别名与算法族**：`"deis"` 与默认结果相同，两种噪声算法在 `sigma_min` 终端下也给出同一序列；
- **组合对拍**：`steps=1..20` × `order=1..3` × 4 组标志组合与 oracle 完全一致；
- **非法输入**：13 组（步数为 0/负/`bool`/浮点、阶数为 0/4/`bool`、未知 `final_sigmas_type`、未知 `algorithm_type`、噪声算法配 zero 终端、`lower_order_final=1`、`euler_at_final=None`）都必须抛 `ValueError`。

共九组断言。

## 复杂度

\(O(N)\) 时间、\(O(N)\) 空间，只做整数比较与自增；不涉及任何张量运算。
