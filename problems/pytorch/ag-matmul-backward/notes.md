# 详解：linear 反向的维度账

## 出处

任何写自定义 `autograd.Function`（fused MLP、量化 linear、LoRA 分支）的岗位都会先问这条：给定 dY 怎么写 dX 与 dW。真正区分候选人层次的不是公式，而是「dW 为什么要对 batch 求和」以及「多卡/多层时这份求和发生在哪一步」。

## 为什么 dW 必然带一次折叠求和

把损失记作 \(L\)，\(Y_{bnk}\) 换成标量形式 \(Y_{b,m,n}=\sum_k X_{b,m,k}W_{n,k}\)。对某个权重分量求导：

$$\frac{\partial L}{\partial W_{n,k}}=\sum_b\sum_m \frac{\partial L}{\partial Y_{b,m,n}}\,X_{b,m,k}$$

\(W\) 被所有 batch、所有位置共享，所以梯度必须沿**所有非 (N,K) 的轴**累加。用 einsum 表达最不容易错：`dW = einsum('...mn,...mk->nk', dY, X)`；写成 reshape 成二维再乘时，如果 \(M\) 与 \(B\) 不可交换（非连续内存）就会静默错位。dX 则相反：`einsum('...mn,nk->...mk', dY, W)`，没有任何归约。

## 核心与易错点

- **忘记折叠**：`(dY.T @ X)` 在 3-D 输入上会直接报错或给出 `(...,N,M)@(...,M,K)`，看似算得出、维度却是 \(B\) 份而非 1 份。
- **转置对象**：`W` 是 `(out_features, in_features)`，前向用的是 `W.T`，因此 dX 乘的是 `W` 本身、dW 是 `dY.T @ X`，两个转置极易写反。
- **dtype 归属**：dX 跟着 X、dW 跟着 W；混合精度下若两者都返回 float32 会破坏与 autograd 的逐位比对。
- **零尺寸**：`M=0` 时矩阵乘给出全 0 而不是空，与 autograd 行为一致，不要提前 return。

## 与邻近题的区别

`ag-attention-qkv-backward` 也是矩阵乘反向，但三份梯度互相耦合（Q/K/V 共享同一批 dP）；`linear-layer` 只要求正向、不涉及归约。本题刻意保持单算子，把考察点集中在 batch 折叠与 dtype 归属上。

## 对拍口径与复杂度

oracle 用 `torch.autograd`：把 X、W 设为 requires_grad，`Y=X@W.T`、`loss=(Y*dY).sum()`，取 grad。复杂度 \(O(BMK N)\)（折叠求和与两次乘法同阶），额外空间 \(O(NK)\) 的 dW 累加缓冲。
