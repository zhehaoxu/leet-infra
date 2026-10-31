# 量化：OmniQuant 的等价变换与权重裁剪联合搜索

## 题目描述

把一层权重量化到 4 bit，光靠"旋转/缩放"或光靠"裁剪"都不够：OmniQuant 的做法是把两者放进**同一个搜索**里。第一个旋钮是**逐通道等价变换**——用一个可学习的对角量级 \(s\) 加上一个秩一改动 \(\mathbf 1\,\text{shift}^\top/n\) 左乘权重、用它的近似逆右乘权重，把激发的量级"抹平"到权重这一侧；第二个旋钮是**权重裁剪**（WSC，Learnable Weight Clipping 的初始化）——把变换后权重里最大的那几个值裁到阈值内，牺牲幅值换码数。

本题给定一层方阵线性权重、一份**已经按列减过均值**的校准激活、alpha 网格与 WSC 候选表，要求对每个 \((\alpha,\text{choice})\) 组合执行「定 \(s\) → 定 shift → 组装 \(T\) 与 \(T_{\text{inv}}\) → 变换 → 裁剪 → 逐行对称整数化 → 层重构误差」，返回最优组合、该组合下的变换矩阵与裁剪后权重、整张误差表，以及 LWC 的初值界。

## 规则

记 \(n\) 为方阵边长、\(T\) 为激活 token 数、\(\mathbf 1\) 为长度 \(n\) 的全 1 列向量、\(\delta\) 为 Kronecker 记号。逐通道的量级统计为

\[
s1_i=\max_j|W_{ij}|\ (\text{逐输出通道}),\qquad s2_j=\max_t|X_{c,tj}|\ (\text{逐输入通道}).
\]

对 `alphas` 里每个 \(\alpha\)（外层，按输入顺序）：

1. **混合尺度**：\(s=\alpha\,s1+(1-\alpha)\,s2\)。上游把 \(\beta\) 直接复用为 \(\alpha\)，于是
   \(\text{shift}=\alpha\,(s1-s)/s1\)；\(s1_i=0\) 的位置取 \(\text{shift}_i=0\)。
2. **两个变换矩阵**（都是 \(n\times n\)）：
   \[
   T=\operatorname{diag}(s)\,(I-\mathbf 1\,\text{shift}^\top/n),\qquad
   T_{\text{inv}}=\operatorname{diag}(1/s)\,(I+\mathbf 1\,\text{shift}^\top/n),
   \]
   即按元素写 \(T_{ij}=s_i(\delta_{ij}-\text{shift}_j/n)\)、\((T_{\text{inv}})_{ij}=(\delta_{ij}+\text{shift}_j/n)/s_i\)。
   \(T\) 左乘权重、\(T_{\text{inv}}\) 右乘权重：\(W'=T\,W\,T_{\text{inv}}\)。
3. **恒等偏离**：\(\text{norm\_comp}=\lVert T\,T_{\text{inv}}-I\rVert_\infty=\max_{ij}|(T\,T_{\text{inv}}-I)_{ij}|\)。
   两式相乘得 \(T\,T_{\text{inv}}=I-\big(\sum_k\text{shift}_k/n^2\big)\mathbf 1\,\text{shift}^\top\)，
   偏离完全由 shift 的分量和决定，所以 \(\alpha=0\) 与 \(\alpha=1\) 时它精确为 0。
4. **权重裁剪（WSC）**：把 `wsc_w`（调用方给的逐 block \(\max|W|\)，形状任意）摊平，
   \(w_2=\min_i w_i\)、\(w_3=\max_i|w_i|\)，`wsc_choices` 的元素**按源码顺序**映射成标量阈值：

   ```
   min -> w2      ma -> w3/4      ma2 -> w3/2      2 -> w2/8      1 -> w3/8
   ```

   裁剪对整张权重生效：\(W^c=\operatorname{clamp}(W',-t,t)\)。
5. **逐输出行对称整数化**：\(q_{\max}=2^{\text{bits}-1}-1\)、\(q_{\min}=-2^{\text{bits}-1}\)，
   第 \(r\) 行的 scale 为该行 \(W^c\) 绝对值最大者除以 \(q_{\max}\)（整行为 0 时 scale 取 1、码取 0），
   \(\text{deq}_{rj}=\operatorname{clamp}(\operatorname{round}(W^c_{rj}/\text{scale}_r),q_{\min},q_{\max})\cdot\text{scale}_r\)。
6. **层重构误差**：基准是**变换后、未量化**的 \(W'\)，量化噪声是唯一的惩罚来源：
   \[
   \mathcal L(\alpha,\text{choice})=\frac{1}{T\cdot n}\sum_{t,j}\Big((X_cW'^\top)_{tj}-(X_c\,\text{deq}^\top)_{tj}\Big)^2 .
   \]

`losses` 把 \((\alpha,\text{choice})\) 摊平成一维：`losses[i * len(wsc_choices) + j]` 对应 `(alphas[i], wsc_choices[j])`。取最小者，**并列取先出现者**（先比 \(\alpha\) 下标、再比 choice 下标，严格小于才更新）。内部全程 float64，取整用「就近、`.5` 取偶」；返回的浮点张量一律 float32。

LWC 只考它的**初值**：记 \(\alpha_0=0.01\)（题面固定的常数，不是输入），
\(c=|w_{\text{sc},i}|\cdot 2\sigma(\alpha_0)\)，逐元素返回为 `lwc_bound`；摊平顺序与 `wsc_w` 一致。
上游那行是 \(\min(w,\,w\cdot 2\sigma(\alpha))\)，在 \(\alpha_0\) 处 \(2\sigma(\alpha_0)>1\)，\(\min\) 不生效——它要等梯度把 \(\alpha\) 压成负数才咬合，而迭代不在本题范围内（`lwc_steps` 因此不改变任何返回值）。

## 任务

- 输入：`W` \((n,n)\)、`Xc` \((T,n)\)（已按列减均值，本题不再中心化）、`alphas`（非空，元素在 \([0,1]\)）、`wsc_choices`（非空，元素属于 \(\{\text{min},\text{ma},\text{ma2},\text{2},\text{1}\}\) 这五个字符串）、`wsc_w`（任意形状、非空）、`n_bits`（2～16）、`lwc_steps`（非负）。
- 输出：一个 dict，键恰为 `best_alpha_index, best_alpha, best_wsc_choice, losses, s, shift_vec, transform_T, transform_T_inv, norm_comp, clipped_weight, lwc_bound`；浮点张量为 float32、数值容差 \(10^{-5}\)（loss 可用相对容差 \(10^{-4}\)）。
- 非法输入抛 `ValueError`：`W` 不是二维方阵、`W` 与 `Xc` 的输入通道数不一致、`alphas` 或 `wsc_choices` 为空、alpha 越出 \([0,1]\)、choice 不在五元集合内、`n_bits` 越出 \([2,16]\)、`lwc_steps` 为负、`wsc_w` 为空、某个 \(s_i\) 恰为 0（其倒数无定义）。

## 示例

```
W  = [[ 3.88,  1.75,  1.45, -0.85],        Xc = [[-0.27,  0.39,  0.96,  1.06],
     [-2.00,  1.35,  1.60,  1.30],               [ 0.12, -0.06, -0.09,  0.46],
     [ 0.88,  0.85, -0.35,  0.15],               [-0.03, -0.11, -0.84,  0.46],
     [ 1.75, -0.95,  1.10, -1.50]]               [ 0.10,  0.24,  0.31, -0.29],
                                                  [ 0.08, -0.46, -0.34, -1.69]]
alphas      = [0.00, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
wsc_choices = ["min", "ma", "ma2", "2", "1"]
wsc_w       = [3.88, 1.75, 1.60, 1.50]
n_bits = 4,  lwc_steps = 0
```

\(s1=[3.88,2.00,0.88,1.75]\)、\(s2=[0.27,0.46,0.96,1.69]\)、\(w_2=1.50\)、\(w_3=3.88\)。误差表（行 = \(\alpha\)，列 = choice）：

```
                 min         ma         ma2           2           1
a=0.00    6.86674e-01  1.10736e+00  7.63157e-01  2.26830e+00  1.71558e+00
a=0.15    6.76201e-02  2.67329e-01  9.47780e-02  1.32119e+00  7.73547e-01
a=0.20    4.80141e-02  2.64371e-01  3.53595e-02  1.35942e+00  7.97988e-01
a=0.25    4.17612e-02  2.71005e-01  2.82965e-02  1.44052e+00  8.50101e-01   <- best
a=0.30    5.31183e-02  3.29183e-01  2.95982e-02  1.55287e+00  9.30576e-01
a=0.35    9.69807e-02  3.90868e-01  2.85752e-02  1.69158e+00  1.03530e+00
a=0.40    1.34912e-01  4.78205e-01  4.18000e-02  1.85496e+00  1.16291e+00
```

```
best_alpha_index = 3      best_alpha = 0.25      best_wsc_choice = "ma2"
s         = [1.172500, 0.845000, 0.940000, 1.705000]
shift_vec = [0.174452, 0.144375, -0.017045, 0.006429]
norm_comp = 4.143121e-02
clipped_weight = [[ 1.940000,  1.940000,  1.623360, -0.577780],
                  [-1.437042,  1.290990,  1.319418,  0.643622],
                  [ 0.674053,  0.849912, -0.478703,  0.080629],
                  [ 1.940000, -1.940000,  1.767677, -1.505982]]
lwc_bound = [3.899400, 1.758750, 1.608000, 1.507500]
```

阈值 \(t=w_3/2=1.94\) 只把第 0 列那个 \(3.88\) 量级的离群通道削下来，\(\alpha=0.25\) 再把各行 absmax 拉到接近一致，逐行 scale 随之变小；\(\alpha=0\)（纯激活尺度）与更宽的裁剪（`min`、`ma2` 之外）都更差。

第二个例子里权重是单位阵、\(s1=s2\)，于是每个 \(\alpha\) 的 \(T=I\)、\(\text{shift}=0\)、三个 \(\alpha\) 的 loss 完全相同——并列取先出现者：

```
W = [[1.0, 0.0],   Xc = [[ 1.0, -1.0],
     [0.0, 1.0]]         [-1.0,  1.0]]
alphas = [0.0, 0.2, 0.4]
wsc_choices = ["min", "ma", "ma2", "2", "1"]
wsc_w = [1.0, 1.0],  n_bits = 4,  lwc_steps = 0
```

```
best_alpha_index = 0   best_alpha = 0.0   best_wsc_choice = "min"
losses = [0.0, 5.625e-01, 2.5e-01, 7.65625e-01, 7.65625e-01]   (same for all three alphas)
s = [1.0, 1.0]   shift_vec = [0.0, 0.0]   norm_comp = 0.0
clipped_weight = [[1.0, 0.0], [0.0, 1.0]]
lwc_bound = [1.004999995231628, 1.004999995231628]
```

## 约束

- 变换只覆盖**方阵层**（`in_features == out_features`）：\(s1\) 与 \(s2\) 同长才能凸组合，\(T\) 与 \(T_{\text{inv}}\) 才同为 \(n\times n\)。
- `wsc_w` 是输入、不是重算：所有 \(\alpha\) 共用同一份逐 block 幅值，`lwc_bound` 也只由它和 `n_bits` 之外的常数决定。
- 阈值是**标量**（整个 block 表算出的 \(w_2,w_3\)），不是逐行 absmax；\(2\sigma(\alpha_0)=1.00499996\) 这个倍数必须由 `lwc_bound` 显式可见。
- 判题只看返回值，不限定实现方式。
