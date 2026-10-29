# 解题说明 · SmoothQuant α 搜索

## 出处

- SmoothQuant（Xiao et al., ICML 2023, arXiv:2211.10438）§3：逐输入通道缩放 \(s_j=\max|X_j|^\alpha/\max|W_j|^{1-\alpha}\)，把激活离群迁移到权重；论文里 \(\alpha\) 由网格搜索按精度选（原实现给 LLM 用的经验值 0.5），本题把这个搜索本身作为考点。
- 迁移公式与 `smooth-quant-outlier-migration` 同源，但那题只算**给定 α** 的一次迁移；本题补上"怎么选 α"以及两侧量化之后的端到端误差。
- per-token 激活量化、per-output-channel 权重量化是 W8A8 推理的标准配置（如 TensorRT-LLM / vLLM 的 int8 GEMM）。

## 核心与易错点

**α 的两端都是坑。** α=0 时 \(s_j=1/\max|W_j|\)，激活几乎不缩，离群通道仍会撑大量程；α=1 时 \(s_j=\max|X_j|\)，权重被乘上激活的幅值，权重侧反而更难量。示例里 α=0 的 MSE 是 1.95e-4、α=1 是 1.91e-4，中间 0.5 只有 9.45e-6——两个方向的误差都比最优值大一个量级以上。所以不能"只试两端"。

**逐 token / 逐通道，不是 per-tensor。** 激活按行定 scale（每个 token 一行），权重按输出通道定 scale；如果两边都退化成 per-tensor，示例里的离群会把整行压扁，α 的差异就消失了。

**对称范围是 \([-q_{\max},q_{\max}]\)。** 不要把负侧扩到 \(-q_{\max}-1\)：这不只是差一个码，示例的 \(q_{\max}=127\) 下会把"就近取整"的边界整体挪一格，MSE 与最优 α 都可能变。取整用 `torch.round`（`.5` 取偶），与 `numpy.round` 一致。

**零通道 / 零行要短路。** \(sx_j\) 或 \(sw_j\) 为 0 时 \(s_j\) 会变成 0 或 `inf`；约定取 1。整行为 0 的 token 反量化后必须仍是 0，不能出现 `0/0`。

**MSE 要除以 \(T\cdot O\)。** 只求和会让不同形状的校准集之间的数值不可比；题目规定了均值。

**并列取先出现者。** 网格允许重复，用严格 `<` 更新最优即可；`<=` 会悄悄把并列变成"取最后"。

**误差来自两侧量化。** \(\hat X\hat W^\top\) 里两个因子都是反量化结果；漏掉激活量化（只量权重）会得到完全不同的 α 曲线——那是对角近似的 W-only 口径（见 `awq-activation-scale`），不是本题。

## 对拍口径

- 参考实现：`torch` 向量化，逐 α 计算 `sx`/`sw` 广播缩放，`_quant_sym` 用 `abs().amax(dim=1)` 定 scale，`torch.round` 取整，float64 累加。
- 独立 oracle：纯 Python 嵌套 list，逐 token / 逐通道手写 `round()`（Python 的银行家舍入与 `torch.round` 一致）、逐元素点积求和，和参考实现不共享控制流。
- 60 组随机实例（\(T\le 5\)、\(C\le 6\)、\(O\le 4\)、随机放大一个通道、α 网格随机子集、bits 4–8）上三元组逐位一致；另有 9 个定向用例（示例、全 1 并列、激活离群、权重离群、零通道、零行、bits=4、非法 bits、最优 α 的 scale 复核）。
- 不变量：`mses` 全有限、`best_alpha` ∈ 网格、`best_scales` 与最优 α 的公式值一致。

## 复杂度

- 每个 α：\(O(TC + OC + TOC)\)，网格大小 \(G\) 时总计 \(O(G\cdot TOC)\)；示例与测试规模下是毫秒级。
- 真实实现还会把校准集分 batch、按层搜 α；本题只保留单层的误差口径。

## 相关题目

- `smooth-quant-outlier-migration`：给定 α 的迁移本身（\(s\)、\(\hat X\)、\(\hat W\)），是本题的内层操作。
- `awq-activation-scale`：W-only 的 α 搜索，用激活加权的**权重**误差（对角近似），不量化激活。
- `quant-rtn-gptq-error`：RTN 与 GPTQ 的输出 MSE 对比，是"量化误差怎么算"的另一种口径。
- `quant-int4-marlin-layout`：搜索/量化之后，4-bit 码如何按 tile 布局进显存。
