# 解题说明 · OmniQuant 的等价变换与权重裁剪

## 出处

- **转写出处是 `Saad-Fahmy/LLM-QAT` 的 `models/llama_linear.py`（本会话两次 raw 200 逐行核对），不是 OpenGVLab 官方 OmniQuant**。那段 `OmniQuantLinearWrapper` 给出逐通道等价变换的三行核心——\(s1\) 取权重逐输出通道 absmax、\(s2\) 取（中心化后）激活逐输入通道 absmax、\(s=\alpha s1+(1-\alpha)s2\)，shift 用 \(\beta(s1-s)/s1\) 且 **\(\beta\) 直接复用 \(\alpha\)**，再由 `diag(s) @ (I - 1*shift^T / n)` 组装 \(T\) 与它的近似逆；LWC 初值那行核到同文件第 91 行附近（源码把下界因子写成 `2 * sigmoid(可学习量)` 再与原始幅值界取 `min`，可学习量初始化为 `0.00`）。
- **OmniQuant 原版的口径（2026-10-30 回源 OpenGVLab/OmniQuant 核实）**：论文 2308.13137 与官方仓库里，LET 是**逐输入通道**的 \(s_j=\mathrm{clip}(s2_j^{\alpha}/s1_j^{1-\alpha},10^{-5})\)（几何插值，不是 \(\alpha s1+(1-\alpha)s2\) 的算术插值）加上 \(\delta_j=(\max X_j+\min X_j)/2\) 的平移，\(X\) 与 \(W\) 两侧分别是 `X/s` 与 `s*W`、偏置补 `shift @ W`，并且**没有 α 网格搜索**——`--alpha` 是单个浮点（默认 0.5），全部等价变换参数靠端到端梯度下降（`omniquant.py` 里被训练的是参数名含 `smooth`/`smooth_scale`/`bound_factor` 的量）。LWC 在官方 `quantize/quantizer.py` 里是 `xmax = sigmoid(upbound_factor)*xmax`、`xmin = sigmoid(lowbound_factor)*xmin`、`init_value = 4.`、`scale = range/(2**n_bits-1)`、`x_int = round(x/scale)+round(zero_point)` 后夹紧，**同样没有** `min(w, w*2*sigmoid(0))` 与 `ma/ma2/2/1` 候选表。`wiki-xx/OmniQuant` 这个 fork 名今日经 GitHub API 查询为 Not Found（其仓库描述实为 LLM-QAT）。
- 同一仓库的量化工具里给了裁剪的两段：WSC（用 `bmax` 的逐 block 幅值查一张固定表当初始阈值）与 LWC（把阈值变成可学习量，初值写成 \(w\cdot 2\sigma(\alpha)\)）。
- 本题的 \(\alpha\) 网格 `[0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]` 与「首个最小值」的选取（`min(range(len(scalars)), key=scalars.get)`）也来自那段搜索代码。

## 来源与刻意简化

- **无模型、无校准前向**：`Xc` 由调用方直接给成"已按列减过均值"的形状 \((T,n)\)，上游那句 `X = X - X.mean(0)` 不在本题里重做。
- **只覆盖方阵层**：\(s1\)（逐输出通道）与 \(s2\)（逐输入通道）要能凸组合，必须等长；\(T\) 与 \(T_{\text{inv}}\) 才同为 \(n\times n\)。上游对长方形层没有可直接照抄的写法，本题不臆造推广，非方阵直接 `ValueError`。
- **LWC 只考初值**：可学习参数的梯度步、它的上下界 clamp、以及每 block 的内层训练循环全部排除；`lwc_steps` 因此是契约里的占位输入，**不改变任何返回值**（有用例专门扫 0..8 钉住这一点）。
- **`wsc_w` 是输入不是重算**：上游用 `bmax(W)` 现算逐 block 幅值，本题要求调用方显式给一份，并且所有 \(\alpha\) 共用它——`lwc_bound` 同样只由这张表决定，与 \(\alpha\)、choice 无关。
- **LWC 常数的口径冲突按数值锚定**：源码那行写 `alpha = full_like(w, 0.00)`，但 \(2\sigma(0)=1\) 精确成立、什么常数也锚不出来；与它一起被引用的 `1.0049958…` 其实是 \(\alpha_0=0.01\) 时 \(2\sigma(\alpha_0)=1.00499996\) 的 float32 读数。题面按数值口径取 \(\alpha_0=0.01\)，并且返回的是**缩放后的候选界本身** \(|w_{\text{sc}}|\cdot 2\sigma(\alpha_0)\)：源码式子里的 \(\min(w,\cdot)\) 在 \(\alpha_0\) 处不生效（倍数 \(>1\)），它只在训练把 \(\alpha\) 压负之后才咬合。
- **源码路径的复核记录**：本会话对 `Saad-Fahmy/LLM-QAT` 的 `models/llama_linear.py` 做过两次独立 raw 抓取，两次都 200 且与本节式子逐行一致（AdaRound 那题同源、同法核对）。此前记作 `wiki-xx/OmniQuant/models/llama_patch.py` 的那条路径经 GitHub API 查询为 Not Found（该仓库名当前的描述是 LLM-QAT），已按 `LLM-QAT` 为准更正；判题口径仍只认题面。
- **与 OpenGVLab 官方 OmniQuant 的差异必须写明**：官方仓库（2308.13137 的实现，`models/transformation.py` + `quantize/quantizer.py` + `quantize/omniquant.py`）里 LET 是**逐输入通道的几何插值** \(s_j=\mathrm{clip}(s2_j^{\alpha}/s1_j^{1-\alpha},10^{-5})\) 加 \(\delta_j=(\max X_j+\min X_j)/2\) 的平移，\(X\)/\(W\) 两侧做 \(X/s\) 与 \(sW\)、偏置补 \(\delta W\)；**没有 α 网格搜索**（`--alpha` 是单个浮点，默认 0.5，等价变换参数靠端到端梯度下降，被训练的参数名含 `smooth`/`smooth_scale`/`bound_factor`）；LWC 是 `xmax = sigmoid(upbound_factor)*xmax`、`xmin = sigmoid(lowbound_factor)*xmin`、`init_value = 4.`、`scale = range/(2**n_bits-1)`、`x_int = round(x/scale)+round(zero_point)` 后夹紧，**没有** `min(w, w*2*sigmoid(alpha))`，也**没有** `min/ma/ma2/2/1` 这张 WSC 表。所以本题的名字组合是「OmniQuant 的等价变换 + WSC/LWC」这一**工程实现族**的共性，而不是官方版的逐字复现；题面公式是唯一判题口径。
- **\(T_{\text{inv}}\) 取带 \(\delta_{ij}\) 的那一对**：把转写里的 \((1+\text{shift}_j)/n/s_i\) 按字面实现（全 1 矩阵），乘积 \(T\,T_{\text{inv}}\) 是个秩一量级、\(\lVert T\,T_{\text{inv}}-I\rVert_\infty\approx 1\)，上游那条 `< 0.05` 断言永远不可能满足，\(\alpha=1\) 时也不可能是恒等对角。取 \((\delta_{ij}+\text{shift}_j/n)/s_i\) 之后 \(T\,T_{\text{inv}}=I-(\sum_k \text{shift}_k/n^2)\mathbf 1\,\text{shift}^\top\)，偏离只由 shift 的**和**控制，断言才有意义，\(\alpha\in\{0,1\}\) 两个退化点也恰好退化成互逆的对角矩阵、`norm_comp` 精确为 0。题面把这一对写成唯一口径。
- **量化只到"逐输出行对称"这一层**：分组 scale、非对称零点、真正的 int4/int8 kernel、以及 OmniQuant 的残差/低秩旁路都不在范围内；`n_bits` 只进 \(q_{\max},q_{\min}\)。

## 核心与易错点

**秩一因子的下标在列上。** `outer(ones, shift)` 给 \([i,j]=\text{shift}_j\)，\(T_{ij}=s_i(\delta_{ij}-\text{shift}_j/n)\)；写成 `outer(shift, ones)` 就变成 \(\text{shift}_i\)，是一张不同的矩阵，\(T\,T_{\text{inv}}\) 也不再靠近 \(I\)。

**shift 的分母是 \(s1\)。** \(\text{shift}=\alpha(s1-s)/s1\)，\(s1_i=0\)（整行权重为 0）时取 0——此时 \(s_i=(1-\alpha)s2_i\ne s1_i\)，若照除以 0 会得 NaN，所以这个退化分支必须显式处理；而 \(s_i=0\)（行全 0 **且**该列激活全 0）时 \(1/s\) 无定义，直接 `ValueError`。

**退化点很好用来自查**：\(\alpha=0\Rightarrow s=s2,\ \text{shift}=0\)；\(\alpha=1\Rightarrow s=s1,\ \text{shift}=0,\ T=\operatorname{diag}(s1),\ T_{\text{inv}}=\operatorname{diag}(1/s1)\)。两条都使 `norm_comp` 恰为 0，任何实现算出非零就是下标或 `/n` 放错了位置。

**`norm_comp` 是乘积的偏离，不是矩阵自身的范数。** \(\lVert T\,T_{\text{inv}}-I\rVert_\infty\) 取逐元素绝对值的最大值；它随 \(\alpha\) 单调变差（示例里 \(1.1\text e{-16}\to4.14\text e{-2}\)），所以"先最小化 loss、再检查恒等偏离"确实是两个独立动作。

**裁剪阈值是整表的标量。** \(w_2=\min_i w_i\) 用的是**带符号最小值**，\(w_3=\max_i|w_i|\) 用的是**绝对值最大**；五个候选按源码顺序

```
min -> w2      ma -> w3/4      ma2 -> w3/2      2 -> w2/8      1 -> w3/8
```

注意 `"2"` 与 `"1"` 是**字符串名字**、分别对应 \(w_2/8\) 与 \(w_3/8\)，不是"第 2 个/第 1 个候选"。

**loss 的基准是变换后的 \(W'\)，不是原始 \(W\)。** 变换本身免费，只有"裁剪 + 逐行整数化"进 loss；把基准写成 \(X_cW^\top\) 会让所有 \(\alpha\) 一起挨打，选出来的 \(\alpha\) 直接错位。

**摊平顺序与并列。** `losses[i * len(choices) + j]`，\(\alpha\) 外层；取首个最小值，因此 `alphas` 允许重复、并列时保留更靠前的下标（示例 2 专门造了这个情形）。choice 之间并列同理（`min` 永远排在 `ma2` 前）。

**逐行量化细节。** `scale = row_absmax / qmax`，整行为 0 时 `scale = 1`、码为 0（不许 `0/0`）；取整是「就近、`.5` 取偶」；截断区间不对称，\([q_{\min},q_{\max}]=[-2^{b-1},\,2^{b-1}-1]\)。

**返回 dtype 恒为 float32。** 内部一律 float64（否则 BLAS 分块会让示例里钉住的数字随线程数漂移），输入是 float64 时返回张量仍转 float32，`losses`/`norm_comp`/`best_alpha` 是 Python float。

## 对拍口径

- 参考实现：`torch` 向量化，float64 计算；`s1 = W.abs().amax(1)`、`s2 = Xc.abs().amax(0)`，秩一因子用 `torch.outer(ones, shift)`，\(T\) 用 `torch.diag(s) @ (eye - ...)`，裁剪用 `torch.clamp(Wt, -t, t)`，逐行 scale 用 `amax(keepdim=True)/qmax` 并把 0 换成 1，误差用 `(Xc@Wt.T - Xc@deq.T).pow(2).sum()/(T*n)`；选取用 `min(range(len(losses)), key=losses.get)`。
- 独立 oracle：纯 Python 嵌套循环——手写 \(s1/s2\) 的 max、手写 \(n\times n\) 矩阵乘（三次）、手写 `delta + shift/n` 的两个矩阵、逐 choice 求 \(w_2,w_3\)、逐行用 `round()`（同为 `.5` 取偶）做整数化、逐元素累加平方误差，与参考实现不共享控制流。
- 覆盖：示例 1（\(\alpha=0.25,\ t=1.94\) 的联合最优）、示例 2（三个 \(\alpha\) 完全并列）、\(\alpha=0/1\) 退化、\(s1\) 含零行、常数权重（\(w_2=w_3\)）、choice 逐个单独跑一遍以核对阈值表、\(2\sigma(0.01)\) 常数逐元素核对、dtype 扫描、非连续视图、输入不被修改、`lwc_steps` 惰性、非法输入（越界 alpha / 空网格 / 形状不符 / 非方阵 / 未知 choice / 越界位数），以及随机对拍扫描。
- 不变量：`norm_comp == max|T@T_inv - I|`（由返回的 \(T,T_{\text{inv}}\) 复算）、`clipped_weight == clamp(T@W@T_inv, -t, t)`（\(t\) 由 `best_wsc_choice` 复算）、`losses.min() == losses[best_alpha_index * C + best_choice_index]`、\(T\) 与 \(T_{\text{inv}}\) 都是对角占优。

## 复杂度

- 每个 \((\alpha,\text{choice})\)：两次 \(n\times n\) 乘 \((T W)\)、再乘 \(T_{\text{inv}}\) 共 \(O(n^3)\)，加两次 \((T,n)\times(n,n)\) 投影 \(O(Tn^2)\)。
- \(T\) 与 \(T_{\text{inv}}\) 只由 \(\alpha\) 决定，所以 \(T W T_{\text{inv}}\) 可按 \(\alpha\) 复用，choice 内层只剩 \(O(n^2)\) 的裁剪与量化。
- 总计 \(O(|A|\cdot(n^3+Tn^2)+|A||C|\cdot n^2)\)；真实 OmniQuant 在这里还要乘上校准 batch 数与内层迭代步数——那部分正是本题排除掉的训练开销。

## 相关题目

- `quant-awq-clip-scale-search`：AWQ 的 scale-clip 联合网格搜索，逐通道对角缩放 + 分组对称量化；本题把对角缩放换成"对角 + 秩一"的等价变换，并把裁剪阈值改成查表。
- `awq-activation-scale` / `quant-smoothquant-alpha-search`：单参数搜索与"把量迁到权重一侧"的口径对照。
- `quant-hadamard-transform`：另一种（正交的）变换矩阵；本题的 \(T\) 不是正交的，靠 \(T_{\text{inv}}\) 近似抵消。
- `group-wise-quantize` / `per-channel-quantize`：本题只用到逐输出行的对称 scale，分组与非对称零点在这些题里。
