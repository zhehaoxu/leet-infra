# 详解：SDEdit strength 的截断起止步

## 出处

Meng et al., *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations*（arXiv:2108.01073）：先用强度 \(s\) 把输入加噪到中间时刻（\(t_0\) 随 \(s\) 单调增减），再从该时刻起跑反向 SDE/ODE；diffusers 的 `StableDiffusionImg2ImgPipeline` 里 `init_timestep = min(int(num_steps * strength), num_steps)`、`t_start = max(num_steps - init_timestep, 0)` 是同一件事的工程写法。

## 核心与易错点

1. **索引向下取整**：`floor` 而非 `round`，否则 \(s=1\) 可能落到 \(T-1\) 之外，或 \(s\) 略大于某个阈值时提前跳档。判题把 \(s=0.99\) 与 \(s=1\) 都取到，专门卡这一点。
2. **表的顺序**：`alpha_bars` 按噪声级升序（索引 0 最干净），这条约定必须与采样器遍历方向一致；反过来（索引 0 最噪）会让 \(s\) 越大反而越干净。
3. **\(\epsilon\) 是入参**：本题不现采随机数，所以「同一 \(\epsilon\) 反解再回代」必须逐位还原，这也是判题检验系数是否用 \(\sqrt{\bar\alpha_i}\) 而非 \(\sqrt{\bar\alpha_{T-1}}\) 的手段。
4. **\(s=0\) 的边界**：此时不加噪、步数为 0，输出应等于 \(x_0\)（前提 \(\bar\alpha_0=1\)）。很多实现会在这里出现「至少跑一步」的 off-by-one。
5. **剩余步数与索引相同**：\(i\) 既是起点索引也是要跑的步数（走到 0），两者不要各写一套公式。

## 对拍口径

参考解取 \(\bar\alpha_i\) 后用 `math.sqrt` 向量化乘加（float64）。判题 `_oracle` 逐元素展开成 \(a\cdot x_0+s\cdot\epsilon\)，另加四组结构断言：索引的 `floor` 语义与单调性、\(s=0/1\) 两端、`x_i ↔ ε` 往返、以及「把输出反解出的 \(\epsilon\) 与入参逐位比较」以锁定系数。阈值 `allclose(atol=1e-6)`。

## 复杂度

逐元素 \(O(N)\)，索引计算 \(O(1)\)。
