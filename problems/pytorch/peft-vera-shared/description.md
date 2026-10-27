# PEFT：VeRA 共享冻结基矩阵与可学缩放

## 题目描述

**VeRA**（Kopiczko et al., ICLR 2024, arXiv:2310.11454）把 LoRA 的参数量再砍一个量级：所有层**共享**一对随机冻结的低秩矩阵 \(A\in\mathbb R^{r\times d_\text{in}}\)、\(B\in\mathbb R^{d_\text{out}\times r}\)，每层只训练两个**缩放向量**——输入侧 \(\lambda_d\in\mathbb R^r\) 与输出侧 \(\lambda_b\in\mathbb R^{d_\text{out}}\)：

\[\Delta W=(\text{diag}(\lambda_b)\,B)\,(\text{diag}(\lambda_d)\,A),\qquad
y=xW_0^{\!\top}+\lambda_b\odot\Bigl(\bigl(\lambda_d\odot(xA^{\!\top})\bigr)B^{\!\top}\Bigr)\]

前向按 PEFT `vera/layer.py`：`result + lambda_b * F.linear(lambda_d * F.linear(x, A), B)`。每层可训参数只有 \(d_\text{out}+r\) 个，对比 LoRA 的 \(r(d_\text{out}+d_\text{in})\)。

PEFT 的初始化：\(\lambda_b=0\)（起点是纯基座）、\(\lambda_d=d_\text{initial}=0.1\)。

## 任务

实现 VeRA 前向并报告每层可训参数量。

- 输入：\(W_0\in\mathbb R^{d_\text{out}\times d_\text{in}}\)，\(\lambda_b\in\mathbb R^{d_\text{out}}\)，\(\lambda_d\in\mathbb R^{r}\)，\(A\in\mathbb R^{r\times d_\text{in}}\)，\(B\in\mathbb R^{d_\text{out}\times r}\)，\(x\in\mathbb R^{N\times d_\text{in}}\)。
- 输出 \((y,\ \texttt{n\_trainable})\)：\(y\in\mathbb R^{N\times d_\text{out}}\)（float32），\(\texttt{n\_trainable}=d_\text{out}+r\)。

**约定**：内部用 float64 计算、输出 float32；广播按上式；不使用任何现成 PEFT 库。容差 \(10^{-5}\)。

## 提示

- 两个缩放作用在不同位置：\(\lambda_d\) 缩放低秩**隐藏维**（r 维），\(\lambda_b\) 缩放**输出维**（d_out 维）。
- \(\lambda_b=0\) 时输出应等于 \(xW_0^\top\)（初始化语义）。
- 可先用 \(\Delta W=(\text{diag}(\lambda_b)B)(\text{diag}(\lambda_d)A)\) 重建再验证——函数式前向与矩阵式重建必须一致。
