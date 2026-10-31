# 离散扩散采样：VQ-Diffusion 的反扩散单步在 log 空间拼转移矩阵

## 题目描述

VQ-Diffusion（*Vector Quantized Diffusion Model for Text-to-Image Synthesis*，arXiv:2111.14822）的前向过程不加高斯噪声，而是把每个隐像素以 \(\gamma_t\) 的概率改写成**掩码类**，且掩码是吸收态——一旦盖上就不再揭开。于是反向的一步是一个离散贝叶斯反转：

\[ p(x_{t-1}\mid x_t)=\sum_{x_0} q(x_t\mid x_{t-1})\,\frac{q(x_{t-1}\mid x_0)\,p(x_0)}{q(x_t\mid x_0)} \]

diffusers 的 `VQDiffusionScheduler` 全程在 log 空间做这件事，而且不用矩阵乘法：它拿 `sample` 造出 log one-hot，再用一句 `(onehot + a).logaddexp(b)` 一次性生成整列转移概率——命中行给出 \(\log(a+b)\)，其余行给出 \(\log b\)（源码注释里自称 "this is a cheeky trick"）。代价是一连串形状与下标的机关：单步表长 \(T\) 而累积表长 \(T+1\)、\(t=0\) 有一条完全不走后验的捷径、掩码行要「先保存再丢弃」、\(\log 0\) 被钳成 \(\log 10^{-30}\) 而不是 \(-\infty\)。

本题实现这一个算子。所有本来由调度器自己生成的东西（\(\alpha/\gamma\) 两张表、Gumbel 随机数）都摊成显式入参；没有码本、没有网络前向，也没有随机数生成。

## 规则

记 \(C=\)`num_vec_classes`、已知类个数 \(k=C-1\)（掩码类固定是下标 \(k\)）、\(T=\)`len(at)`。\(b\) 表由 \(a,c\) 推出，\(t\)-1 上标指**数组下标**而不是数学时刻：

\[ b_t=\frac{1-a_t-c_t}{k},\qquad \bar b_t=\frac{1-\bar a_t-\bar c_t}{k}\qquad\Longrightarrow\qquad \bar a_t+k\,\bar b_t+\bar c_t=1 \]

转移矩阵（对某个隐像素，观测值 \(j=x_t\)）：

\[ q(x_t=j\mid x_0=i)=\begin{cases}\bar c_t & j=k\\ \bar a_t+\bar b_t & i=j\\ \bar b_t & \text{否则}\end{cases}\qquad q(x_t=j\mid x_{t-1}=i)=\begin{cases}c_t & j=k\\ a_t+b_t & i=j\\ b_t & \text{否则}\end{cases} \]

而 \(q(x_t=j\mid x_{t-1}=k)=1\) 当 \(j=k\)、否则 \(0\)（吸收态）。log one-hot 记作

\[ E[i]=\begin{cases}0 & x_t=i\\ \log(10^{-30})=-69.07755278982137 & \text{否则}\end{cases} \]

\(t=0\) 是捷径：\(\text{logits}=\) `model_output` **逐字返回**，只有 \(k\) 个类别，后验整段跳过。\(t>0\) 时按下面五步（都在 log 空间，\(\operatorname{LSE}\) 是沿类别维的 `logsumexp`，`logaddexp` 逐元素）：

\[ V[i]=\operatorname{logaddexp}(E[i]+\log \bar a_t,\ \log \bar b_t)\ \xrightarrow{\ x_t=k\ \text{时整列}\ \log \bar c_t\ }\ \text{，}i=0..k-1 \]

\[ U[i]=\operatorname{logaddexp}(E[i]+\log a_t,\ \log b_t)\ \xrightarrow{\ x_t=k\ \text{时整列}\ \log c_t\ }\ \text{，}U[k]=E[k] \]

\[ q=\text{model\_output}-V,\qquad S=\operatorname{LSE}_{\text{cls}}(q),\qquad q\leftarrow q-S \]

\[ W[i]=\operatorname{logaddexp}(q[i]+\log \bar a_{t-1},\ \log \bar b_{t-1})\ (i<k),\qquad W[k]=\log \bar c_{t-1} \]

\[ \text{logits}=W+U+S \]

注意三处下标：\(V\) 用**累积表下标 \(t\)**，\(U\) 用**单步表下标 \(t\)**，\(W\) 用**累积表下标 \(t-1\)**。

采样按上游的 Gumbel 口径做，但随机数由入参给出：

\[ \text{noise}=-\log\!\big(-\log(u+10^{-30})+10^{-30}\big),\qquad \text{prev}=\arg\max_{\text{cls}}(\text{logits}+\text{noise}) \]

返回 `(prev, logits)`：`prev` 是 int64 的 \((B,L)\)，`logits` 是 float64，\(t>0\) 时 \((B,C,L)\)、\(t=0\) 时 \((B,C-1,L)\)。

六个容易写错的地方：

1. **两张表的长度不同**。单步表长 \(T\)，累积表长 \(T+1\)，末位是永远读不到的哨兵（上游写 \(1.0\) 与 \(0.0\)，因此 \(\bar b_T=0\)）。累积表读下标 \(t+1\) 在 \(t=T-1\) 上就会撞到哨兵，给出 \(-\infty\)。
2. **\(t=0\) 时 `model_output` 只有 \(k\) 个类别**，`noise` 的形状也必须是 \((B,k,L)\)；返回的 logits 同样少一行。多写一行掩码类会把整个形状链条带崩。
3. **掩码行要在 `[:, :-1, :]` 丢弃它之前保存**。\(U[k]=E[k]\) 扮演 \(q(x_t\mid x_{t-1}=\text{掩码})\)：观测为掩码时是 \(0\)（概率 1），否则是那个钳制值。
4. **\(\log 0\) 是钳制值而不是 \(-\infty\)**。它不光出现在 one-hot 的非命中项里（那里被 \(\log(a+b)\) 盖掉，看不出差别），还作为 \(U[k]\) 一路活到最终 logits：例子里未掩码像素的掩码行是 \(-68.755728\)，而不是 \(-\infty\)。
5. **\(b\) 的分母是 \(k=C-1\)**，不是 \(C\) 也不是 1。若不除 \(k\) 而写成 \(b=1-a-c\)，转移矩阵的行和就变成 \(\bar a_t+k(1-\bar a_t-\bar c_t)+\bar c_t=1+(k-1)(1-\bar a_t-\bar c_t)\neq1\)，下一条的守恒立刻破掉。
6. **中间那次归一化不能省，末尾的 \(S\) 也不能忘**：\(\operatorname{logaddexp}\) 不是线性的，先减 \(S\) 再走 \(W\)、然后把 \(S\) 加回来才是上游的式子。它有个可验证的守恒后果——逐像素 `exp(logits).sum(cls)` 恒等于 `exp(model_output).sum(cls)`，与 \(t\) 无关。

## 示例

```
num_vec_classes = 3          # classes 0,1 are known; 2 is the mask class
at  = [0.80, 0.60, 0.40, 0.20]              # T = 4
ct  = [0.10, 0.20, 0.30, 0.40]
att = [0.80, 0.48, 0.192, 0.0384, 1.0]      # trailing sentinel is never read
ctt = [0.10, 0.28, 0.496, 0.6976, 0.0]
bt  = (1 - at - ct) / 2 = [0.05, 0.10, 0.15, 0.20]
btt = (1 - att - ctt) / 2 = [0.05, 0.12, 0.156, 0.132, 0.0]

model_output = [[[-0.2, 0.3, -1.0, 0.5],
                 [ 0.4, -0.7,  0.1, -0.3]]]     # (B, C-1, L)
sample       = [[0, 2, 1, 2]]                    # pixels 1 and 3 are masked
```

`gumbel_uniform` 全取 `exp(-1)` 时噪声恰为 0，`prev` 就是裸 argmax（平票取下标最小者）：

```
t=0 prev=[[1, 0, 1, 0]]   shape (1, 2, 4)
   [[-0.2, 0.3, -1.0, 0.5],
    [ 0.4, -0.7,  0.1, -0.3]]

t=1 prev=[[0, 0, 1, 0]]   shape (1, 3, 4)
   [[  0.22076,  -0.177582,  -1.310108,   0.027097],
    [  0.061595,  -1.050658,   0.185034,  -0.675965],
    [-68.755728,   -0.416358, -69.78935,   -0.158519]]

t=2 prev=[[0, 2, 1, 2]]
   [[ 0.341843,  -0.642624,  -1.31155,  -0.427565],
    [-0.102015,  -1.279465,   0.185357,  -0.945435],
    [-67.872666,   0.041475, -68.639611,   0.299314]]

t=3 prev=[[0, 2, 1, 2]]
   [[ 0.317762,  -1.159089,  -0.925513,  -0.928227],
    [-0.065598,  -1.514885,   0.073922,  -1.219759],
    [-66.999512,   0.272192, -67.551656,   0.530031]]

exp(logits).sum(cls) = [2.310555, 1.846444, 1.473050, 2.389539]   # same for all four t
```

`t=1` 的第一格（观测 \(x_t=0\)）手算一遍，可以核对三个下标：\(p=(e^{-0.2},e^{0.4})=(0.818731,1.491825)\)；\(\bar a_1=0.48,\bar c_1=0.28,\bar b_1=0.12\) 给出 \(V=(0.60,0.12)\)、\(r=(1.364551,12.431872)\)、\(S=13.796424\)、\(w=(0.098906,0.901094)\)；累积表下标 \(t-1=0\) 给出 \(\bar a_0=0.8,\bar b_0=0.05\)，于是 \(W_0=0.05+0.8w_0=0.129125\)；单步表 \(a_1+b_1=0.7\)（观测命中）。\(0.7\times0.129125\times13.796424=1.247024\)，\(\log=0.22076\) ✓。掩码行则是 \(10^{-30}\times\bar c_0\times S=1.38\times10^{-30}\)，\(\log=-68.755728\) ✓。

换成真随机数：

```
gumbel_uniform = [[[0.50, 0.10, 0.90, 0.20],
                   [0.30, 0.70, 0.40, 0.60],
                   [0.80, 0.05, 0.25, 0.65]]]

t=0 (has no mask row) prev=[[1, 1, 0, 1]]
t=1, 2, 3           prev=[[0, 1, 0, 2]]     # logits are byte-identical to the table above
```

噪声只改采样、不改返回的分布；但 \(t=0\) 用这张 \((1,3,4)\) 的噪声会直接报错，因为那一步只有两层。

## 约束

- `model_output` 必须是三维浮点张量 \((B, C-1, L)\)、全为有限值；`sample` 必须是整数张量（`int32`/`int64` 皆可，`bool` 与浮点非法）\((B, L)\)，取值落在 \([0, C-1]\)。
- `at`、`ct` 是一维同长张量，长度 \(T\ge 2\)，元素落在 \((0,1)\) 且 \(a_t+c_t<1\)；`att`、`ctt` 长度必须是 \(T+1\)，**前 \(T\) 个**元素满足同样的条件（末位哨兵不校验，因为读不到）。
- `timestep` 是 Python `int`（`bool` 非法），\(0\le t<T\)；`num_vec_classes` 是 Python `int` 且 \(\ge 2\)。
- `gumbel_uniform` 是浮点张量，取值落在 \([0,1)\)，形状是 \((B,C,L)\)（\(t>0\)）或 \((B,C-1,L)\)（\(t=0\)）。违反任一约束抛 `ValueError`。
- 内部计算在 float64 下完成，`logits` 以 float64 返回，`prev` 以 int64 返回。禁止 `import diffusers`。
