# 解题说明 · AWQ 缩放-裁剪联合搜索

## 出处

- AWQ（Lin et al., MLSys 2024, arXiv:2306.00978）：逐输入通道缩放保护与大激活相连的权重；论文还给出权重量化前的**裁剪**（clip）作为抑制离群值的第二手段，两者一起搜。
- AutoAWQ 的实现把这两步拆成 `awq/quantize/auto_scale.py` 的 scale 网格（带 `max_shrink` 收缩参数）与 `awq/quantize/auto_clip.py` 的裁剪网格；`max_shrink` 用来限制缩放把权重压得过狠。
- **范围声明**：本题的归一化（几何平均）、裁剪阈值（层内全局 absmax）、误差权重（`mean|X|` 的一次方）与上游实现的具体写法并不逐项一致，题面公式即唯一口径；若要与某一版 AutoAWQ 逐位对齐，需回源该版本的 `auto_scale.py` / `auto_clip.py` 重核。

## 核心与易错点

**裁剪阈值是层内全局的。** \(t=\text{clip}\cdot\max_{o,j}|W_{oj}|\)。若按逐行 absmax 定阈值，`clip=0.5` 会把每一行都砍掉一半——没有离群值的行也被毁掉，示例里 loss 会从 1.4e-4 涨到 1e-2 量级。全局阈值只动真正接近全局最大值的元素。

**归一化用几何平均，且要在收缩之前。** \(s/\sqrt{\max(s)\min(s)}\) 让缩放不改变整体量级；之后再 `max(s, 1-max_shrink)`。顺序反了（先收缩后归一）会把下界重新缩掉，`max_shrink` 就失去意义。

**`max_shrink` 是下界。** \(s_j\ge 1-\text{max\_shrink}\)：限制权重被缩小的比例。`max_shrink=0` 表示权重一步都不许缩，此时所有 \(s_j\ge 1\)——不要把下界写成上界，也不要把它当成"允许收缩到 0"。

**误差要在逻辑域里算。** 量化的是 \(W^c\odot s\)，反量化后必须除以 \(s_j\) 才回到与 \(W^c\) 可比的逻辑权重；漏掉这一步会让误差随 \(s\) 整体漂移，最优 \(\beta\) 直接错位。

**分组边界与全零组。** 沿输入通道每 `group_size` 个一组，最后一组不能跨组借 scale；组内全 0 时 scale 取 1、码取 0，不能出现 `0/0`。

**并列取先出现者。** 外层 \(\beta\)、内层 `clip`，用严格 `<` 更新；网格允许重复元素。

**返回的码属于缩放后的权重。** `best_codes` 是 \(W^c\odot s\) 的码，不是 \(W\) 的码；`best_scales` 是归一化 + 收缩后的实际 \(s\)，不是未处理的 \(s\)。

**`xm` 是均值不是 absmax。** 题目用 \(\frac1T\sum_i|X_{ij}|\) 作为通道重要性；换成 absmax 会改变所有误差数值。

## 对拍口径

- 参考实现：`torch` 向量化，`s` 用布尔掩码算退化通道，`Wc = clamp(W, -clip*wmax, clip*wmax)`，分组用 `reshape(O, C//g, g)` + `amax(keepdim)`，误差用广播加权后取均值。
- 独立 oracle：纯 Python 嵌套循环，逐 (β, clip) 手写 clamp、逐组 `round()`、逐元素累加，与参考实现不共享控制流。
- 60 组随机实例（\(O\le 3\)、\(C\le 4\) 且偶数、\(T\le 4\)、随机放大一列激活、`max_shrink` 随机、分组与位宽随机）上五元组一致；另有 9 个定向用例（示例、裁剪单调改善、β 优于 0、收缩下界、分组量化边界、bits=8、全零权重、非法 bits、误差表与最优组合对应）。
- 不变量：`losses` 形状正确且最优组合取到最小值、`best_scales` 与 `_norm_s` 公式一致、码在 \([-q_{\max},q_{\max}]\) 内。

## 复杂度

- 每个 (β, clip)：\(O(O_\text{out}\cdot C)\) 次量化与误差计算；总 \(O(R\cdot K\cdot O_\text{out}\cdot C)\)，\(R\)、\(K\) 为网格大小。
- 真实 AWQ 用校准集前向做同样的搜索，本题用对角近似把前向压成按通道加权，保留网格与裁剪账。

## 相关题目

- `awq-activation-scale`：AWQ 的 \(\alpha\) 单参数搜索，per-tensor RTN + \(x_{bar}^2\) 加权的权重误差，无裁剪/收缩/分组。
- `smooth-quant-outlier-migration` / `quant-smoothquant-alpha-search`：把量迁到权重的一侧；本题只缩放权重、激活保持浮点（W4A16 口径）。
- `group-wise-quantize`：分组量化本身的数值口径。
- `quant-int4-marlin-layout`：量化出的 4-bit 码如何按 tile 布局落到显存。
