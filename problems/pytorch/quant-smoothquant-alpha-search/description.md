# 量化：按 W8A8 输出误差搜索 SmoothQuant 的 α

## 题目描述

SmoothQuant 用逐输入通道的等价缩放 \(s_j\) 把激活的离群量迁到权重上：\(X/s\)、\(W\cdot s\)，数学上 \(XW^\top\) 不变。迁移多少由一个平衡系数 \(\alpha\) 控制：

\[s_j=\frac{\left(\max_i|X_{ij}|\right)^{\alpha}}{\left(\max_o|W_{oj}|\right)^{1-\alpha}}.\]

\(\alpha=0\) 让激活独自承担离群（\(s_j=1/\max|W_{\cdot j}|\)），\(\alpha=1\) 全推给权重。**哪一端更好不是公式能直接回答的**：它取决于两侧各自的量化误差，只能把完整的 W8A8 伪量化跑一遍、按输出误差挑。本题给定校准激活、权重和一个 \(\alpha\) 网格，要求对每个 \(\alpha\) 做量化-反量化、计算输出 MSE，并返回最优的 \(\alpha\)、对应的缩放因子和整条误差曲线。

## 规则

对网格中每个 \(\alpha\)：

1. **求缩放因子**：逐输入通道 \(s_j=\text{sx}_j^{\alpha}/\text{sw}_j^{1-\alpha}\)，其中 \(\text{sx}_j=\max_i|X_{ij}|\)、\(\text{sw}_j=\max_o|W_{oj}|\)。若 \(\text{sx}_j=0\) 或 \(\text{sw}_j=0\)（该通道退化），取 \(s_j=1\)。
2. **迁移**：\(X'=X/s\)（按列除）、\(W'=W\cdot s\)（按列乘）。
3. **量化激活（逐 token）**：每个 token 行用该行绝对值最大元素定对称 scale，\(\text{scale}_i=\max_j|X'_{ij}|/q_{\max}\)，
   \[
   q_{ij}=\operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{X'_{ij}}{\text{scale}_i}\right),-q_{\max},q_{\max}\right),\qquad \hat X_{ij}=q_{ij}\,\text{scale}_i.
   \]
   整行为 0 时 \(\hat X\) 整行为 0（不产生 NaN）。
4. **量化权重（逐输出通道）**：每个输出通道行同样处理，\(\text{scale}_o=\max_j|W'_{oj}|/q_{\max}\)，得到 \(\hat W\)。
5. **误差**：\(\text{MSE}=\frac{1}{T\cdot O}\sum_{i,o}\left[(XW^\top)_{io}-(\hat X\hat W^\top)_{io}\right]^2\)，内部用 float64。

在网格上取 MSE 最小的 \(\alpha\)；并列时取**先出现**者（网格允许重复元素）。

返回三元组：

- `best_alpha`：float，最优 \(\alpha\)；
- `best_scales`：float32 \((C,)\)，最优 \(\alpha\) 下的 \(s\)；
- `mses`：float32 \((\text{len(alphas)},)\)，按输入顺序的每个 \(\alpha\) 的输出 MSE。

其中 \(q_{\max}=2^{\text{bits}-1}-1\)，取整用 `torch.round` 的「就近、`.5` 取偶」语义。

## 任务

- 输入：`X` float32 \((T,C)\)、`W` float32 \((O,C)\)、`alphas`（非空 float 列表，每个在 \([0,1]\)，允许重复）、`bits` 为 2～8 的整数。
- 输出：`(best_alpha, best_scales, mses)`，`best_scales`、`mses` 为 float32，容差 `1e-5`（MSE 可用相对容差 \(10^{-4}\)）。
- `bits` 越界抛 `ValueError`；其余输入按题面保证取值，不要求额外校验。

## 示例

```
X = [[ 0.9, 1.1, 55.0],        W = [[ 0.4, -0.6,  0.05],
     [ 1.2, 0.7, 62.0],             [-0.5,  0.3, -0.04]]
     [-0.8, 1.4, 70.0]]
alphas = [0.0, 0.25, 0.5, 0.75, 1.0],  bits = 8
```

逐通道 absmax：`sx = [1.2, 1.4, 70.0]`、`sw = [0.5, 0.6, 0.05]`。各 α 的输出 MSE：

| α | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| MSE | 1.9476e-4 | 1.6137e-4 | **9.4523e-6** | 3.9517e-5 | 1.9066e-4 |

```
best_alpha  = 0.5
best_scales = [1.549193, 1.527525, 37.416573]   # sqrt(1.2/0.5), sqrt(1.4/0.6), sqrt(70/0.05)
```

（离群通道 2 在 \(\alpha=0\) 时压不干净、\(\alpha=1\) 时又把权重推得过猛，中间值最好。）

## 约束

- 量化是**对称**的：码范围是 \([-q_{\max},q_{\max}]\)，不要用 \([-q_{\max}-1,q_{\max}]\)。
- 激活按 **token**（行）定 scale，权重按**输出通道**（行）定 scale；两侧都用各自行的 absmax，不要退化成 per-tensor。
- 求 \(s_j\) 时用 float64；`sx`、`sw` 为 0 的通道固定 \(s_j=1\)，避免 0 除与 `inf`。
- 判题只看返回值，不限定实现方式。
