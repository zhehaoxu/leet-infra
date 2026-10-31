# 解题说明 · AdaRound 重构误差候选搜索

## 出处

- AdaRound（Nagel, Faronato, Diniz, Corradi, Galesso & Bassett, ICML 2020, **arXiv:2004.10568**, "Up or Down? Adaptive Rounding for Post-Training Quantization"；2026-10-30 复核纠正：此前记作 2004.10501，那个编号其实是 Graubohm 等的自动驾驶概念阶段论文）：把每个权重的"进/舍"写成可搜索的软化变量 \(h\)，用校准激活上的重构误差决定往哪个格点挪，并用退火指数把解推向两端。
- 本题的逐行公式转述自 `Saad-Fahmy/LLM-QAT` 的 `training_utils.py::AdaRound`（其实现沿用上游 `ideal/AdaRound` 的写法）：\(\text{floor}(w/\text{scale})\) 取格点、\(K_0=-0.1/K_1=1.1\) 的线性插值、\(\lfloor h\cdot\text{resolution}\rfloor/\text{resolution}\) 的阶梯解码、\(\alpha=\text{sigmoid}((\text{logit}(\bar h)+\log \text{resolution})/2)\)、候选偏移 \([0,+1/2^q,-1/2^q,+1,-1]\) 的贪心搜索与 \(|h-0.5|\le 0.5/\text{resolution}\) 的判定带。
- **转写仓库的复核记录**：`Saad-Fahmy/LLM-QAT` 的 `training_utils.py::AdaRound` 在本会话内复核过两轮，一轮**取到 raw 200 并把该类的 121 行与本节式子逐条对上**（候选表、判定带、`logit(bar_h)`、除以 2 的温标、\(p=1+(i+1)/T\)、`lam=5.0`、以及 `|alpha-0.5|` 惩罚那几句原文都在），另一轮同路径 404（LLM-QAT 现已挂在 `facebookresearch/LLM-QAT`，raw 抖动或改名导致）；因此上条的式子有源码依据，判题口径仍以题面为唯一规范。**可独立佐证的主干**：`yhhhli/BRECQ/quant/adaptive_rounding.py` 与 `quic/aimet` 的 `adaround_*` 确认了 \(\gamma=-0.1,\ \zeta=1.1\)、`x_floor = floor(x/delta)`、`rest = x/delta - x_floor`、\(\alpha=-\log\!\big(\frac{\zeta-\gamma}{\text{rest}-\gamma}-1\big)\) 的 logit 初始化、软化目标 `clamp(sigmoid(alpha)*(zeta-gamma)+gamma, 0, 1)` 与硬解码 `x_int = x_floor + (alpha >= 0)`；AIMET 侧重构项 \(\lVert \hat Y-Y\rVert_F^2\)、正则项 \(1-|2\alpha-1|^{\beta}\)、\(\beta\) 的余弦退火 `end + 0.5*(start-end)*(1+cos(rel_iter*pi))` 与 `warm_start` 之前不计正则。**在可达实现里找不到对应、按本题固化**的部分：`resolution` 阶梯与 `+log(resolution)`／除以 2 的温标组合、以及**逐元素五候选贪心搜索**（论文里叫 `loss_grid`，本轮 ar5iv 正文取不到）——它们只在上游转写里出现，本题把它们钉成显式契约。

## 来源与刻意简化

- **随机采样分支整体排除**。真实 AdaRound 每轮对 \(h\) 做 Bernoulli 采样（带温度退火）来产生候选权重，本题换成**确定性贪心候选搜索**：同一份输入逐位可复现，判题才能只看返回值。
- **逐层顺序 block reconstruction 排除**。上游把一层搜完冻结后再前向下一层、把误差逐级往下传；本题只有**单个** `W`，`X` 是显式输入的小矩阵，不做任何模型前向，也不依赖训练好的网络。
- **`betas_range = (log2h / zeta) / log(lam)` 中的 `zeta` 与 `log2h` 排除**。转述来源里 `zeta` 没有定义，`log2h`（\(\log_2 h\)）是**逐元素**量而不是**逐步**标量，无法直接当每轮的正则权重。本题不臆造读法：丢掉这两个符号，只保留未歧义的 \(1/\log(\text{lam})\) 与退火指数 \(p_i=1+(i+1)/T\)，合成 \(\beta_i=p_i/\log(\text{lam})\)，并把整条 `betas` 列表作为输出键暴露出来。
- **残差必须先归一化再插值**。转述写作 `h = clamp((r - K0)/(K1 - K0))`，但 \(r\in[0,\text{scale})\) 与 \(K_0,K_1\in[-0.1,1.1]\) 量纲不符，照抄会让除 \(r\) 极小以外的权重全部被夹到 \(h=1\)、搜索退化。上游真正的写法是对"以步长为单位"的残差插值，本题钉成 \(z=r/\text{scale}=\!W/\text{scale}-q\in[0,1)\)。
- **\(q=0\Rightarrow h=1\) 只钉公式，不钉注释**。转述旁白说这条覆盖是"强制向下取整"，但 \(\alpha\) 是**进位概率**、\(h=1\) 会把 \(\alpha\) 推到 \(\approx1\)（见下条），两种说法在这份代码路径里不自洽。题面固定的是覆盖本身与其数值效果，例二就专门让这条覆盖留在最终 \(h=1\) 上可见。
- **\(+\log(\text{resolution})\) 平移与除以 2 的温度照抄**，因此 \(\alpha\) 明显偏向 1：\(\bar h=0.5\) 时 \(\alpha=\text{sigmoid}(\tfrac12\log 1024)=32/33\approx0.9697\)，\(\bar h=\varepsilon\) 时 \(\alpha\approx0.031\)。候选之间的排序主要由这个"压缩到 1"的偏置驱动，不是笔误。
- **\(\varepsilon=10^{-6}\) 有两处固定用途**：解码前把 \(\bar h\) 夹到 \([\varepsilon,1-\varepsilon]\)（否则 logit 发散）、收尾 mask 的带宽容差 \(0.5/\text{resolution}+\varepsilon\)。两处同一个常数，不再引入别的 eps。
- **`searched_mse` 只含重构项**，不含正则项；正则项只参与搜索与 `soft_loss`。

## 核心与易错点

**插值用 \(z\) 不是 \(r\)。** \(h=((z+0.1)/1.2)\) 才落在 \([0,1]\)；直接用 \(r\) 会让 \(h\) 几乎全被夹到 1，候选搜索直接退化成"谁都别动"。

**阶梯解码在夹取之前。** \(\bar h=\lfloor h\cdot\text{resolution}\rfloor/\text{resolution}\)，之后才 `clamp(eps, 1-eps)`；先夹后取整会把 \(h=1\) 变成 \(\bar h=1-2^{-10}\)，\(\alpha\) 少掉一整档。

**冻结带和收尾 mask 是同一个 \(0.5/\text{resolution}\)，语义不同。** 前者决定"这个元素还参不参加下一轮"，后者决定"收尾用 \(\bar h\ge0.5\) 还是用 \(\operatorname{round}(\alpha)\)"。轮数 `steps` 用完时仍在带内的元素才会走 mask 分支。

**目标值是整个 `W` 的。** 试某个候选时，重构项对**所有**输出行求和、正则项对**所有**元素取均值；写成"只看当前元素"会在多列输入上改变胜出候选（正则项的均值分母从 \(On\) 变成 1）。

**并列取更早候选，且只在严格更小时更新。** 夹到 \([0,1]\) 之后，`0`、\(+1/2^{q}\)、\(+1\) 经常给出同一个 trial \(h\)，目标值逐位相同，此时必须是 `0` 胜出——例二的 `chosen_offset = 0` 就是这条。

**`chosen_offset` 是"最后一次提交"。** 带内元素会在多轮里被反复搜索，每轮覆盖一次；行优先遍历 + 每轮独立提交，顺序写反会得到不同的 \(h\)。

**\(q\) 与 \(r\) 是输入决定的，与搜索无关。** 返回的 `q`、`r` 在搜索前就已定死；`h`、`alpha`、`chosen_offset` 是搜索后的状态，别把初始化的 \(h\) 返回出去。

**`betas` 是 \((T,)\) 的标量列表**，\(\beta_i=p_i/\log(\text{lam})\)；\(\text{lam}<1\) 时 \(\log(\text{lam})<0\)，正则权重变负（等价于奖励两端），这也是合法输入，不要 `abs()`。

```
L_i = sum_o ||X * ((q + alpha)_o - W_o / scale) * scale||^2 / n + beta_i * mean(|alpha - 0.5| ** p_i)
candidates = [0, +1/resolution, -1/resolution, +1, -1]   # ties keep the earlier candidate
freeze     = abs(h - 0.5) > 0.5 / resolution             # after the commit
mask       = abs(h - 0.5) <= 0.5 / resolution + 1e-6     # before the final decode
```

## 对拍口径

- 参考解 `variants/pytorch/solution.py`：torch 张量、全程 float64，每轮行优先遍历未冻结元素，5 个候选各克隆一次 `h` 后整张量重解码、重算目标。
- 独立 oracle `test_cases.py::_oracle`：纯 Python 标量三层循环（`.tolist()` 出来，逐元素 `math.floor/log/exp`，重构项逐 \((o,t,j)\) 累加），与参考解不共享控制流；两边在 120 组随机实例（\(O,n,T_\text{cal}\le 4\)、`num_iters` 1..6、`steps` 1..4、`lam` 取 0.3/1.05/5.0/1000、`scale` 0.25/1.0/3.0，并随机放大一列激活）上九个键逐位一致（float32 容差 3e-5，int64 严格相等）。
- 定向用例覆盖：两个公开示例、候选并列取 `0`、\(q=0\) 覆盖、正好在格点上的元素（\(r=0\)）、收尾 mask 命中（\(h\) 落在带内、\(\bar h=0.5\) 进位）、多轮重复搜索、\(X=I\) 手算核对、dtype 扫（float32/float64 一致）、非连续输入、输入不被就地修改、形状/dtype 不变量、三类 `ValueError` 探针。

## 复杂度

- 每轮每个未冻结元素要算 5 次全张量目标：\(O(\text{steps}\cdot On\cdot 5\cdot(T_\text{cal}\cdot n+n))\)；\(O(\text{steps}\cdot T_\text{cal}\cdot O\cdot n^2)\) 量级。
- 题面规模刻意小（\(n\le 4\)、\(T_\text{cal}\le 4\)），暴力即过；向量化只是常数优化，不改变结果。

## 相关题目

- `quant-awq-clip-scale-search`：同样是"给定小矩阵 + 显式搜索"，但搜的是逐通道缩放与裁剪网格；本题搜的是每个权重的进/舍。
- `quant-smoothquant-alpha-search` / `awq-activation-scale`：单参数退火搜索，量迁到激活一侧；AdaRound 迁的是取整方向。
- `group-wise-quantize` / `weight-only-int8-matmul`：量化本身（scale/zero-point/码流）的数值口径；本题的 `scale` 是显式输入，不涉及分组定标。
