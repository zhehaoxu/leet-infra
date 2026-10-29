# 解题说明 · HRA 的 Householder 反射与紧凑 WY

## 出处

- `huggingface/peft` 的 `src/peft/tuners/hra/layer.py`（2024 年并入的 HRA 适配器）：`_get_householder_vt` 产出 `(V, T)`，`get_delta_weight` 组装 `H = I - (V T) Vᵀ`，`_apply_delta_weight_to_input` 在输入侧算 `x Hᵀ`。本课的口径逐行对齐这份实现；HRA 论文编号未回源，以源码为准。
- 紧凑 WY 表示本身出自 Schreiber & Van Loan, *A Storage-Efficient WY Representation for Products of Householder Transformations*（SIAM J. Sci. Stat. Comput., 1989）：把 r 个反射压成 `I - V T Vᵀ`。

## 核心与易错点

**乘积顺序是 `P_0 @ P_1 @ … @ P_{r-1}`，不是反过来。** 紧凑表示只对这一个顺序成立。写成 `P_{r-1} @ … @ P_0`，示例里 `H` 的符号会整块变掉，而且正交性检查照样通过——只看 `HᵀH = I` 发现不了。

**`T_inv` 的三角部分取严格上三角。** `triu(VᵀV, diagonal=1)` 去掉对角；对角恒为 1（列已单位化）由 `0.5 I` 那一项补齐。用 `triu(·)` 含对角会把对角写成 `0.5 + 1 = 1.5`，`T` 就全错。

**`T` 是解三角方程得到的，不是 `T_inv` 的转置。** `solve_triangular(T_inv, I, upper=True)` 才是 `T = T_inv^{-1}`；有的实现写 `T = (I/2 + triu(VᵀV,1))^{-1}` 时顺手用 `torch.inverse`，数值上会有 1e-7 级差异，本题按三角求解的口径对拍。

**GS 分支不是「先正交化再走同一套 T」。** 正交化后 `VᵀV = I`，交叉项消失，`T` 直接是 `2I`；如果仍去解三角方程，会得到 `T = 2I` 之外的数值噪声。

**合并是 `W0 @ H` 而不是 `H @ W0`。** HRA 作用在输入侧（`x Hᵀ`），所以权重右乘；写成左乘，输出维度对不上时直接报错，维度恰好相等时（方阵）会静默算错。

**反向不是 `H` 取逆。** `H` 正交，`H⁻¹ = Hᵀ`，而源码用 `T.T` 直接得到 `Hᵀ`；两者数值等价，但按 `T.T` 的口径实现才能保证与 unmerge 的逐位一致。

**QR 的符号不唯一。** GS 分支的 `V = Q` 依赖 `torch.linalg.qr` 的约定；判题里只断言 `H = I - 2QQᵀ` 与 `H u = -u`（对 `Q` 的列符号不变），不断言 `V` 的逐元素符号。

## 与其他题的分工

- `peft-oft-cayley`：OFT 用 Cayley 变换 `(I+Q)(I−Q)⁻¹` 构造**分块对角**正交阵并左乘权重；本题是 Householder 反射乘积 + 紧凑 WY，作用在输入侧（右乘）。
- `peft-loha-hadamard`：LoHa 是 Hadamard 积的低秩分解；本题没有 Hadamard 积。
- `peft-lokr-factorization` / `peft-lokr-kron-merge`：Kronecker 分解；本题是正交反射。
