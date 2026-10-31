# 解题说明 · BOFT 的蝴蝶分解正交变换

## 出处

- `huggingface/peft` 的 `src/peft/tuners/boft/layer.py`：`cayley_batch(data)` 先 `skew_mat = 0.5 * (data - data.transpose(1, 2))`，再 `Q = torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat, left=False)`；`update_layer` 用 `block_butterfly_perm(self.in_features, int(boft_block_num / 2**i), int(boft_block_size / 2), boft_n_butterfly_factor)` 加 `perm2mat`（`perm_mat[i, idx] = 1`）预先算出缓冲区 `boft_P`，形状 `(nbf+1, d_in, d_in)`，并把 `boft_R` 注册成 `(nbf+1, block_num, block_size, block_size)`、`boft_s` 注册成 `(out_features, 1)`；`get_delta_weight` 里 `butterfly_oft_mat = batch[0]` 后 `for i in range(1, N): butterfly_oft_mat = batch[i] @ butterfly_oft_mat`；`Linear.forward` 的未合并路径把权重转置成 `(in, out)`、`torch.mm(boft_rotation, orig_weight)`、转置回来、`* boft_scale`。本课逐段对齐这四段。
- 论文：Liu 等，*Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization*，arXiv:2311.06243（ICLR 2024）；其稠密单级版本是同一作者的 OFT，arXiv:2306.07280。公式与下标以源码为准。
- 与源码的两处**刻意收窄**，都写进了题面约束：其一，peft 的 `boft_n_butterfly_factor` 是 1 起的（内部立刻 `-1`，为 1 时置换直接返回 `torch.arange(n)`），题面的 `F` 是真实级数、与 `boft_R` 的第 0 维相等；其二，peft 在第 `i` 级用 `int(boft_block_num / 2**i)` 做**向下取整**，当 `D` 不能被 `2**i` 整除时段长 `n // b` 就不再等于 `H * 2**i`（例如 `n=12, H=4, D=3` 的第 1 级会得到长度 12 的唯一一段），本题用 `D % 2**(F-1) == 0` 把这类配置挡在门外，使"段长 = `H * 2**i`、组大小 = `H // 2`"这条简化规则处处成立。此外 peft 只在开启蝴蝶时要求块尺寸为偶数，`F = 1` 时允许奇数 `H`；题面统一要求 `H` 为正偶数，好让"第 0 级置换恒为单位阵"这一条不用分情况。

## 核心与易错点

**只有反对称部分起作用。** `S = (R - R.T) / 2` 会把 `R` 的对称部分整体抹掉，因此给 `R` 加任意对称阵，`Q`、`M`、`y` 都逐位不变（`check_only_skew_part_matters`）。这也是 Cayley 参数化的全部意义：用一个无约束方阵换掉一个正交阵的 `H(H-1)/2` 个自由度，多出来的对称分量是冗余参数。

**`left=False` 是右解，别写成左解。** `torch.linalg.solve(A, B, left=False)` 求的是 `X` 使 `X @ A = B`，即 `Q = (I+S)^(-1) @ (I-S)`；默认的左解给出 `(I-S) @ (I+S)^(-1)`。两者在 2×2 反对称阵上恰好相等（`I+S` 与 `I-S` 都可写成 `a*I + b*Ŝ`，交换），所以只用 `H = 2` 的用例**测不出这个 bug**；`H >= 4` 的一般反对称阵上不交换，`check_block_size_four` 用的正是随机 4×4 块。

**`P_i` 是共轭用的，不是左乘。** `M_i = P_i @ B_i @ P_i.T`，即先把输入轴按蝴蝶重排、在重排后的坐标里做分块旋转、再重排回来。只写 `P_i @ B_i` 会破坏正交性以外的所有结构（它仍是正交阵，但块的位置全错）。组大小是 `H // 2` 而不是 `H`：段长 `H * 2**i` 里有 `2**(i+1)` 组，"偶数组在前、奇数组在后"是针对这些半块组的，用 `H` 当组大小得到的是另一个置换（`check_block_size_four` 把 `n=8, H=4` 第 1 级的置换钉成 `[0,1,4,5,2,3,6,7]`）。

**第 0 级的置换恒等。** `i = 0` 时段长就是 `H`、只有 0/1 两组，重排后 `[0,1]` 次序不变，所以 `M_0 = B_0` 是严格分块对角（`check_single_stage_block_diagonal` 逐块验证非对角块为 0，且对角块等于对应 Cayley 结果）。蝴蝶的"混合"从第 1 级才开始：`F = 2, H = 2` 时第 1 级置换是 `[0,2,1,3]`，非对角 2×2 块立刻出现非零元（`check_butterfly_mixes_blocks`）。

**级间连乘的顺序。** `M = M_(F-1) @ ... @ M_0`，小级号在右先作用；peft 的 `for i in range(1, N): mat = batch[i] @ mat` 就是这个左乘循环。各级一般不可交换，`check_stage_product_order` 同时断言 `M == M_1 @ M_0` 与 `M != M_0 @ M_1`。

**`M` 作用在输入的列侧。** `W' = (M @ W0.T).T` 就是 `W0 @ M.T`：旋转发生在 `d_in` 轴上，这也正是 BOFT 能处理 `d_in != d_out` 的原因（若把 `M` 左乘到 `W0` 上，非方阵第一步就形状错误）。两个等价写法都应当成立，`check_rotation_on_input_side` 用 `y = x @ M @ W0ᵀ` 复核，并断言 `M @ W0` 那一版结果不同。

**`boft_s` 逐输出行。** 它的形状是 `(d_out, 1)`，在转置回 `(d_out, d_in)` 之后广播，所以缩的是第 `r` 个输出神经元；把它当成输入列缩放（形状 `(1, d_in)`）或当成对角阵左乘都是错的。`check_scale_applied_after_rotation` 用"只放大某一行、其余行逐位不变"和"按输入列缩放结果不同"双向卡住；`check_scale_orientation_shapes` 则取 `R = 0` 这个干净退化点：此时 `M = I`，`y` 就是基座输出按行乘 `s`。缩放与旋转是**可交换**的（一个作用在行、一个作用在列），所以"先旋转后缩放"在数值上没有唯一性，题面按源码的权重布局写死为 `W' = (M @ W0.T).T * s`。

**正交性与保范数。** 每个 `Q` 是 `det = 1` 的正交阵，置换阵也正交，故 `M @ M.T = I`（`check_orthogonal_rotation`）；当 `W0 = I`、`s = 1` 时前向就是 `y = x @ M`，样本 2-范数不变（`check_norm_preserving`）。`R = 0` 时 `S = 0`、`Q = I`，整条链路退化成单位阵（`check_zero_R_is_identity`），这是 peft 的默认初始化（`init_weights=True` 用 `nn.init.zeros_` 存 `boft_R`），意味着适配器初始时不改变基座行为。

## 对拍口径

- `_oracle` 用纯 numpy 重算：`np.linalg.solve(eye + skew, eye - skew)` 做 Cayley、Python 循环按"段 → 偶组 → 奇组 → 组内元素"逐项列出 `perm`、`P[k, perm[k]] = 1` 手动铺置换阵、`np.zeros((dim, dim))` 切片填分块对角，最后 `M = P @ B @ P.T @ M` 连乘；与参考实现不共享任何算子。
- `check_against_oracle` 覆盖 8 组配置（`F` 从 1 到 3、`H` 取 2 与 4、含非方阵、带偏置、纯 float32 输入），对 `M` 与 `y` 同时比对；`check_example` 用手推的 2×2 Cayley（`t = 2` 时 `Q = 1/5 * [[-3,-4],[4,-3]]`）钉死最小用例。
- 编写时用真实 peft 复核：`tuners.boft.layer.Linear(nn.Linear(in, out), 'd', boft_block_size=H, boft_n_butterfly_factor=F, boft_dropout=0.0)`，覆盖 `boft_R`/`boft_s` 后与 `solution.solve` 比 10 组 `(in, out, H, F)`（`F` 到 4、`in` 到 16），`M` 最大偏差 `2.9e-8`、前向最大偏差 `1.7e-7`（都是参考实现返回 float32 造成的舍入）。多级用例必须 monkeypatch `FastBlockDiag` 与 `get_fbd_cuda`：`get_fbd_cuda()` 返回假时 `update_layer` 会把 `boft_n_butterfly_factor` 强设为 1（并告警"Setting boft_n_butterfly_factor to 1"），而未打补丁的 CPU 分支用 `squeeze(0)` + `torch.block_diag`，只对 `N == 1` 的单级正确。另外 `boft_P` 是无条件 `torch.empty(...)` 建的 fp32 缓冲区，`get_delta_weight` 只把它 `.to(device)`，所以 float64 对拍时要手动升精度。单级路径另用未打补丁的 peft 复核 6 组，偏差 `1.2e-7` 以内。判题文件不导入 peft。

## 复杂度

可训练参数 `F * D * H * H + d_out`（对比稠密 OFT 的 `d_in²` 个）：由于 `D * H = d_in`，参数量是 `F * d_in * H + d_out`，随 `d_in` 只是线性、代价是块尺寸 `H` 与级数 `F` 这两个常数超参。物化 `M` 需要 `O(d_in²)` 时间与空间（题面要返回 `M`，参考实现就走这条稠密路径，每级一次 `(d_in, d_in)` 的三重乘）。训练里不必物化：把 `M_i = P_i B_i P_iᵀ` 依次作用到向量上，每级只需 `D` 次 `H×H` 乘与一次重排，`O(d_in * H)` 时间、`O(d_in)` 空间，总计 `O(F * d_in * H)`，这就是论文用 butterfly 快乘的理由。前向的 `y = x @ W'.T` 仍是 `O(N * d_in * d_out)`，与全量微调的基座算子同量级——BOFT 省的是优化器状态与梯度显存，不是前向算力。
