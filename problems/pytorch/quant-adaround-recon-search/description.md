# 量化：AdaRound 的重构误差候选搜索

## 题目描述

权重量化时"进一位还是舍一位"并不总是就近取整最好：把一个权重往上抬一点点，可能被同一行的其它权重抵消掉。AdaRound（Nagel et al., ICML 2020, arXiv:2004.10501）把这件事写成一次**重构误差搜索**——先给每个权重一个"进位概率" \(p\)，再用校准激活上的输出误差去搜该概率该往哪个格点挪。

本题给定一个小权重矩阵、一小段**显式给出的**校准激活 `X` 和 per-tensor 步长 `scale`，要求按下面的确定性口径完成 AdaRound 的软化、**逐元素贪心候选搜索**与硬解码收尾。没有梯度、没有反向传播、没有随机采样：同一份输入必须给出逐位一致的答案。

## 规则

记 \(\text{resolution}=2^{\text{num\_iters}}\)、\(n\) 为 `W` 的列数、\(z=W/\text{scale}-q\) 为归一化残差、\(K_0=-0.1\)、\(K_1=1.1\)、\(\varepsilon=10^{-6}\)。

1. **格点与残差**：\(q=\lfloor W/\text{scale}\rfloor\)，\(r=W-q\cdot\text{scale}\in[0,\text{scale})\)，\(z=r/\text{scale}\in[0,1)\)。
2. **软化初值**：\(h=\operatorname{clamp}\!\big((z-K_0)/(K_1-K_0),\,0,\,1\big)\)，随后**硬覆盖** \(q=0\Rightarrow h=1\)（不管残差多大，只要下格点正好是 0 就把 \(h\) 钉成 1）。
3. **软解码**：\(\bar h=\operatorname{clamp}\!\big(\lfloor h\cdot\text{resolution}\rfloor/\text{resolution},\ \varepsilon,\ 1-\varepsilon\big)\)，
   \[
   \alpha=\operatorname{sigmoid}\!\left(\frac{\operatorname{logit}(\bar h)+\log(\text{resolution})}{2}\right),\qquad
   \operatorname{logit}(u)=\log u-\log(1-u).
   \]
   夹到 \([\varepsilon,1-\varepsilon]\) 是为了让 \(\bar h=0/1\) 时 logit 不发散；\(+\log(\text{resolution})\) 的平移与除以 2 的温度都照抄上游。
4. **目标函数**：第 \(i\) 轮（\(i=0..T-1\)，\(T=\text{steps}\)）用
   \[
   \mathcal L_i=\frac1n\sum_{o}\bigl\|X\,\bigl((q+\alpha)_{o}- (W/\text{scale})_{o}\bigr)\bigr\|^2\cdot\text{scale}^2
   \;+\;\beta_i\cdot\frac1{On}\sum_{o,j}\bigl|\alpha_{oj}-0.5\bigr|^{p_i},
   \]
   其中 \(p_i=1+(i+1)/T\)、\(\beta_i=p_i/\log(\text{lam})\)。第一项就是重构误差 \(\sum_o\|X(\hat w_o-w_o)\|^2/n\)（\(\hat w=(q+\alpha)\text{scale}\)）。
5. **逐元素贪心搜索**：每一轮按**行优先**（先第 0 行第 0 列、再第 0 行第 1 列……）遍历**尚未冻结**的元素。对每个元素，按这个**固定顺序**试 5 个候选偏移：
   \[
   \bigl[\,0,\ +1/\text{resolution},\ -1/\text{resolution},\ +1,\ -1\,\bigr].
   \]
   每个候选把 \(h_k\) 换成 \(\operatorname{clamp}(h_k+\text{off},0,1)\)，其余元素保持当前值，重解码后算**整个** `W` 的 \(\mathcal L_i\)，取目标值最小的候选；只在严格更小时更新，因此**并列一律保留列表里更早的候选**（例如夹到 1 之后 `0` 与 `+1` 给出同一个 trial \(h\)，胜者记作 `0`）。
6. **提交与冻结**：胜出者提交为 \(h_k\leftarrow\operatorname{clamp}(h_k+\text{off},0,1)\)，`chosen_offset[k]` 写成这个 off（后一轮会覆盖前一轮）。提交后若 \(|h_k-0.5|>0.5/\text{resolution}\)，该元素**冻结**，后续轮次不再搜索；仍在这一条带里的元素下一轮继续被搜索，且新的偏移**累加**在已提交的 \(h_k\) 上。
7. **逐轮记录**：一轮结束后，用当时的 \(\alpha\) 再算一次 \(\mathcal L_i\)，写进 `soft_loss[i]`。
8. **硬解码收尾**：所有轮次跑完后 \(\text{mask}=\bigl(|h-0.5|\le 0.5/\text{resolution}+\varepsilon\bigr)\)。mask 命中的元素落回最近格点：\(\alpha_{\text{hard}}=1\) 若 \(\lfloor h\cdot\text{resolution}\rfloor/\text{resolution}\ge 0.5\)（正好等于 0.5 时进位），否则 0；未命中的元素 \(\alpha_{\text{hard}}=\operatorname{round}(\alpha)\)。最后 \(\hat W=(q+\alpha_{\text{hard}})\cdot\text{scale}\)，`searched_mse` \(=\sum_o\|X(\hat W_o-w_o)\|^2/n\)（**不含**正则项）。

内部计算一律 float64；返回的浮点张量按 `W` 的 dtype（float32 或 float64）给出，\(q\) 是 int64，`searched_mse` 是 0 维张量。

返回一个字典，键固定为 9 个：

- `q` int64 \((O,n)\)：下格点；
- `r` float \((O,n)\)：绝对残差 \(W-q\cdot\text{scale}\)；
- `h` float \((O,n)\)：搜索结束后的软化变量；
- `alpha` float \((O,n)\)：搜索结束后的软进位概率（收尾前的 \(\alpha\)）；
- `betas` float \((T,)\)：逐轮正则权重 \(\beta_i\)；
- `soft_loss` float \((T,)\)：逐轮结束时的目标值 \(\mathcal L_i\)；
- `searched_weight` float \((O,n)\)：硬解码后的量化权重；
- `chosen_offset` float \((O,n)\)：该元素最后一次提交时用的候选偏移（从未被搜索过的元素不会存在——第一轮就会把所有元素各搜一次）；
- `searched_mse` float 0 维：硬解码权重的纯重构误差。

## 任务

- 输入：`W` float32/float64 \((O,n)\)、`X` 同 dtype \((T_\text{cal},n)\)、`scale>0`、`num_iters>=1`、`steps>=1`、`lam>0` 且 `lam != 1`。
- 输出：上述 9 键字典；浮点容差 float32 用 `1e-5`、float64 用 `1e-9`。
- 下列非法输入抛 `ValueError`：`num_iters < 1`、`steps < 1`、`scale <= 0`、`lam <= 0` 或 `lam == 1`、`W` 或 `X` 不是二维、`W.shape[1] != X.shape[1]`、`W` 与 `X` 的 dtype 不同。

## 示例

示例一：\(\text{scale}=1.0\)、\(\text{num\_iters}=10\)（\(\text{resolution}=1024\)、微调步 \(1/1024\approx0.00097656\)）、\(\text{steps}=3\)、\(\text{lam}=5.0\)，`X` 是单位阵，于是重构项退化成"逐元素平方误差之和除以列数"，可手算核对。数值取自 float64 输入。

```
W = [[ 0.34000,  0.90000,  1.55000],
     [ 0.99998, -0.72000,  2.00000]]
X = identity(3)

q               = [[0, 0, 1], [0, -1, 2]]
r               = [[0.34000, 0.90000, 0.55000], [0.99998, 0.28000, 0.00000]]
h               = [[0.00000000, 0.99902344, 0.54069010], [0.99902344, 0.00000000, 0.00000000]]
alpha           = [[0.03100777, 0.99902391, 0.97196826], [0.99902391, 0.03100777, 0.03100777]]
betas           = [0.82844658, 1.03555822, 1.24266987]
soft_loss       = [0.42640970, 0.41976967, 0.40138468]
searched_weight = [[0.00000, 1.00000, 2.00000], [1.00000, -1.00000, 2.00000]]
chosen_offset   = [[-1.00000000, -0.00097656, -0.00097656], [-0.00097656, -1.00000000, -1.00000000]]
searched_mse    = 0.13550000
```

（\(w=2.0\) 正好落在格点上：\(r=0\)、\(q=2\)，候选 \(-1\) 把 \(h\) 打到 0、\(\alpha\) 掉到 0.03 一档，收尾回 \(q+0=2\)，重构误差 0；候选 `0` 的 \(\alpha\) 更接近 1、收尾会进位到 3。\(w=0.34\) 因 \(q=0\) 触发 \(h=1\) 覆盖，随后仍被候选 \(-1\) 拉回 0。）

示例二：单列、\(\text{scale}=1.0\)、\(\text{num\_iters}=10\)、\(\text{steps}=3\)、\(\text{lam}=5.0\)、`X = [[20.0], [-10.0]]`。第一个元素 \(q=0\) 触发 \(h=1\) 覆盖，候选 `0`、\(+1/1024\)、\(+1\) 夹回 \([0,1]\) 后是同一个 trial \(h=1\)，目标值逐位相同，并列取最早的 `0`。

```
W = [[0.99998], [0.05000]]
X = [[20.0], [-10.0]]

q               = [[0], [0]]
r               = [[0.99998], [0.05000]]
h               = [[1.00000000], [0.00000000]]
alpha           = [[0.99996875], [0.03100777]]
betas           = [0.82844658, 1.03555822, 1.24266987]
soft_loss       = [0.49565801, 0.49000997, 0.47233177]
searched_weight = [[1.00000], [0.00000]]
chosen_offset   = [[0.00000000], [-1.00000000]]
searched_mse    = 1.25000020
```

## 约束

- \(\text{resolution}\) 只由 `num_iters` 决定：微调候选步是 \(1/\text{resolution}\)，冻结带与收尾 mask 的半宽都是 \(0.5/\text{resolution}\)；`steps` 只影响 \(p_i/\beta_i\) 调度与搜索轮数。
- 候选顺序就是 \([0,+1/\text{resolution},-1/\text{resolution},+1,-1]\)，遍历顺序是行优先，并列取更早候选——这三条定死了，返回值才唯一。
- 目标值按**整个** `W` 计算：重构项对所有输出行求和、正则项对所有元素取均值，而不是只看当前元素所在行。
- 冻结只由"提交后是否离开 \([0.5-0.5/\text{resolution},\,0.5+0.5/\text{resolution}]\)"决定，与目标值的大小无关。
- 判题只看返回字典，不限定实现方式。
