# 解题说明 · Dynatemp 动态温度

## 用熵当温度计

信息熵 $H=-\sum p\ln p$ 度量分布的不确定度。把它除以**最大可能熵** $\ln n$（均匀分布时的熵）
归一到 $C\in[0,1]$，再线性插值进温度窗口：

$$\text{dyn}=T_{\min}+(T_{\max}-T_{\min})\cdot C^{\text{exponent}},\quad
T_{\min}=\max(0,\text{temp}-\Delta),\ T_{\max}=\text{temp}+\Delta$$

- $C\to1$（几乎均匀、模型很犹豫）→ $\text{dyn}\to T_{\max}$，高温把 logits 压平、鼓励探索。
- $C\to0$（几乎 one-hot、模型已笃定）→ $\text{dyn}\to T_{\min}$，低温（甚至接近 0）保住把握。

> 注意与直觉一致：llama.cpp 的实现是用**熵本身**（不是其倒数）作插值系数，因此「越犹豫 → 越热」。

## exponent 调曲线

$C^{\text{exponent}}$ 决定在窗口里怎么移动：

- `exponent = 1`：线性，50% 熵 → 温度落窗口正中。
- `exponent > 1`：凸，把多数中等熵压向 $T_{\min}$（平时冷静，只在极端犹豫时才升温）。
- `0 < exponent < 1`：凹（如开方），偏向 $T_{\max}$（平时就热）。

## 易错点

1. **自然对数**：熵与 `max_entropy = -ln(1/n) = ln n` 都要用 `ln`，用 `log2` 会整体错一个常数比例。
2. **`clamp_min` 防 `log(0)`**：$p_i=0$ 时 $-p_i\ln p_i$ 取极限 0，实现里 `probs.clamp_min(1e-30)`
   避免 `0 * -inf = nan`。
3. **`delta <= 0` 早退**：此时动态关闭，`dyn` 恒等于 `temp`，且**不做**熵计算——直接
   `softmax(logits/temp)`。
4. 温度作用在 **logit 域**（`logits/dyn` 再 softmax），等价于概率域的对数缩放。

## 契约

与合集一致：`filtered` 原顺序、`chosen` 由原顺序 CDF 逆变换得到；额外返回每行的 `dyn`
（动态温度），单行为标量张量、批量为 `(...,)`。
