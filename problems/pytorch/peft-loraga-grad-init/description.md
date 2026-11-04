# PEFT：LoRA-GA 梯度近似的初始化

## 题目描述

**LoRA-GA**（arXiv:2407.05000，Outsider565/LoRA-GA）让 LoRA 的**第一步梯度**与全微调对齐：对小样本累积梯度做**随机化低秩 SVD**，从 top-\(2r\) 奇异向量银行按四种模式分配 A/B 方向，按四种尺度缩放因子，最后把适配器初始输出**吸收进基座权重**（\(W \leftarrow W-\eta\,BA\)）——训练起点精确回到预训练模型，而第一步更新方向与全微调一致，收敛速度提升 2-4 倍。

本题复刻官方实现（自定义 peft fork，`tuners/lora/layer.py`）的确定性账：

- **grad_svd**：`svd_lowrank(grad.float(), q=min(4r, min(shape)), niter=4)`，\(V^\top=V.\mathrm{T}\)；返回 \(U\ (d_\text{out},q)\)、\(S\ (q,)\)、\(V^\top\ (q,d_\text{in})\)，出域转 float64。
- **direction 四态**（方向分配互斥切片）：
  - `ArBr`：\(B=U[:,\,0{:}2r{:}2]\)（**偶数列**）、\(A=V^\top[1{:}2r{:}2,\,:]\)（**奇数行**）——交错选取；
  - `A2rBr`：\(B=U[:,:r]\)、\(A=V^\top[r{:}2r,\,:]\)；
  - `ArB2r`：\(B=U[:,r{:}2r]\)、\(A=V^\top[:r,\,:]\)；
  - `random`：Python `random.sample(range(2r), 2r)`，**先抽 indexes_A（行）后抽 indexes_B（列）**。
- **scale 四态**：`gd`（\(A\mathrel{/}=\eta,\ B\mathrel{/}=\eta\)）；`unit`（A/B 正交不缩放）；`stable`（\(B,A \mathrel{*}= m^{0.25}/\sqrt\gamma\)，\(m=d_\text{out}=B.\mathrm{size}(0)\)）；`weightS`（\(S(w)\mathrel{/}=\eta\)，\(avg_s=\sqrt{S[:r]}.\mathrm{mean}()\)，\(B,A \mathrel{*}= avg_s\)——weight 的奇异值由调用方入参）。
- **offset 吸收**：\(offset = B@A\cdot\eta\)，\(W' = W - offset\)；`norm_clip` 时 \(ratio=\max|W|/\max|offset|\)，\(ratio<1\) 则 \(offset \mathrel{*}= ratio\)、\(A,B \mathrel{*}= \sqrt{ratio}\)——**不变量** \((B\sqrt{ratio})@(A\sqrt{ratio})\cdot\eta = offset\cdot ratio\) 恒成立。

## 任务

- `mode="grad_svd"`：grad (out,in)、r、niter、seed 入参，返回 (U, S, Vt)。
- `mode="direction"`：U、Vt、r、direction（及 random 的 seed）入参，返回 (B, A)。
- `mode="scale"`：B、A、scaling、scale（及 stable_gamma / weight_singulars）入参，返回缩放后的 (B, A)，**不原地修改**。
- `mode="offset_apply"`：B、A、scaling、weight、norm_clip 入参，返回 (new_weight, a_w, b_w)。
- 全程 float64 进出；SVD 管线内部按官方 float32。

## 示例

\(d_\text{out}=6, d_\text{in}=8, r=2\)，grad 为 randn(6,8)（seed 0），\(\eta=4\)：

```
grad_svd(grad, 2, niter=4, seed=42) -> U:(6,6), S:(6,), Vt:(6,8)
  S = [4.716037750244141, 3.28708815574646, 3.007770538330078,
       1.8632816076278687, 1.1207000017166138, 0.9001481533050537]（float32 管线逐位锚定）
direction(U, Vt, 2, "ArBr") -> B:(6,2)=U[:, [0,2]], A:(2,8)=Vt[[1,3], :]
scale(B, A, 4.0, "gd") -> B/4, A/4
offset_apply(B, A, 4.0, W, norm_clip=True) -> (W - B@A*4, A, B)（ratio<1 时按不变量收缩）
```

## 约束

- `svd_lowrank` 用官方 q=min(4r, min(shape))、niter 入参；r<=0 抛 ValueError。
- direction / scale 的非法枚举值抛 ValueError。
- weightS 的 weight 奇异值显式入参（判题不重跑 weight SVD）。
