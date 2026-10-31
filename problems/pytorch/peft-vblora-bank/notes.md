# 解题说明 · VB-LoRA 的向量库选择

## 出处

- `huggingface/peft` 的 `src/peft/tuners/vblora/layer.py`：`_get_low_rank_matrix(logits, bank, topk)` 先 `logits.topk(topk, dim=-1)`、再 `F.softmax(top_k_logits, dim=-1)`、最后 `(weights.unsqueeze(-1) * bank[indices]).sum(-2)`；`_get_lora_matrices` 把 A 侧结果 `reshape(r, -1)`、把 B 侧结果 `transpose(1, 2).reshape(-1, r)`；`get_delta_weight` 直接返回 `transpose(B @ A, fan_in_fan_out)`；`Linear.forward` 的未合并路径是 `result + F.linear(F.linear(dropout(x), A), B)`。本课逐行对齐这四段。
- 同一文件里 `update_layer` 的校验：`r > 0`、`topk > 0`、`in_features % vector_length == 0`、`out_features % vector_length == 0`，并规定 `vblora_logits_A` 形状 `(r, in//vl, V)`、`vblora_logits_B` 形状 `(out//vl, r, V)`。
- 论文：Li 等，*VB-LoRA: Extreme Parameter Efficient Fine-Tuning with Vector Banks*，arXiv:2405.15179（NeurIPS 2024）。公式以源码为准。

## 核心与易错点

**softmax 只在选中的 `topk` 个分数上归一化。** 这是源码里最容易被"顺手写对"的一句话：`F.softmax(top_k_logits)` 的输入是 `topk` 之后的那一小段，而不是整条长度 `V` 的 logits。用全量 softmax 时，被丢弃的行仍会分到权重，`A` 的每一段就不再是选中行的凸组合；`check_selected_only_softmax` 故意放一条指向 `[100, 100]` 的大权重落选行，全量归一化会让分量飙到 ~7.6，一眼可见。反过来这也带来一个恒等式：权重和恒为 1，所以库行全 1 时 `A`、`B` 必为全 1（`check_bank_is_shared` 末段验证）。

**并列取谁在源码里是不确定的，题面必须自己钉死。** `torch.topk` 不承诺并列元素的选择顺序（本机 CPU 上，当并列组横跨截断边界时它取的是下标更大的那个，例如 `[0.5, 0.5, 9.0, -5.0]` 的 `topk=2` 给出 `[2, 1]`）。本题把口径固定成**值降序、同值按下标升序**的稳定选择，判题与参考实现都用它，换后端也不会漂；notes 里保留这个差异，是为了让读者知道这不是 peft 的语义承诺，而是一道题的可复现性承诺。

**`B` 侧必须先 `transpose(1, 2)` 再展平。** 选择结果是 `(out_tile, r, vl)`，目标形状是 `(d_out, r)`，所以行号要走 `o*vl + s`、列号才是 `j`；直接 `reshape(-1, r)` 得到的是 `(out_tile*r*vl/r, r)`，尺寸碰巧也对得上，但 `r` 会被塞进行号里，`check_b_transpose_layout` 专门把这一版错误展平拿来对照。A 侧反而**不能**转置：`(r, in_tile, vl)` 直接展平就是 tile 主序。

**`delta = B @ A`，不是 `A @ B`。** 形状是 `(d_out, r) @ (r, d_in)`；写反在方阵且 `r` 恰当时才可能看不出来，非方阵立刻形状报错。前向的 `F.linear(F.linear(x, A), B)` 展开就是 `x @ Aᵀ @ Bᵀ = x @ (B@A)ᵀ`，两条路径必须同时成立（`check_merge_equivalence`）。

**没有任何 `alpha / r` 缩放。** LoRA 的 `scaling = lora_alpha / r` 在 VB-LoRA 源码里不存在，`get_delta_weight` 就是 `B @ A`；`check_delta_no_extra_scaling` 用 `delta * r` 的反例卡住这一条。

**库是共享的，`A`、`B` 用同一张。** `num_vectors` 与 `vector_length` 是全模型级的超参，两侧 logits 的最后一维必须等于库的行数；`check_bank_is_shared` 改一行库向量，验证 `A`、`B` 都可能被改动，而 `delta` 仍等于 `B @ A`。

**两个退化端值得单独跑。** `topk = V` 时选择退化为全量 softmax 加权和（`check_topk_full` 与显式 `torch.softmax(logits)` 对齐），这是训练里"软"的一极；`topk = 1` 时权重恒 1、就是硬 `argmax` 取一整行（`check_topk_one`），此时梯度只能流进被选中的那条库行，logits 本身拿不到梯度——这正是原论文要用 top-k 软加权而不是纯硬选择的理由。

**整除与一致性校验。** `d_in % vl`、`d_out % vl` 必须为 0，`logits_A`、`logits_B` 的 `r`、`V` 必须与库对齐，`0 < topk <= V`（`topk > V` 在 torch 里是 `RuntimeError`，题面统一成 `ValueError`）。

## 对拍口径

- `_oracle` 用纯 numpy 实现选择与拼装：`np.argsort(-logits, kind="stable")` 取前 `topk`、`np.take_along_axis` 取分数、手写带最大值平移的 softmax、`bank[idx]` 高级索引取行后求和，再分别做 `reshape(r, -1)` 与 `transpose(0,2,1).reshape(-1, r)`；与参考实现不共享任何算子。
- `check_against_oracle` 覆盖 6 组配置（含 `1×1×1` 极小、`topk=1`、`topk=V`、非方阵、带偏置、纯 float32 输入），逐一对 `A`、`B`、`delta`、`y` 四个返回值比对；其余检查各守一个上文的坑。
- 编写时用真实 peft 复核：构造 `tuners.vblora.layer.Linear(nn.Linear(in, out), {'def': bank}, ...)`，覆盖 `vblora_logits_A/B` 后与 `solution.solve` 比 5 组 `(in, out, r, V, vl, topk)`，`A`/`B`/`delta` 最大偏差 `6e-8`、前向 `2.4e-7`（来自 peft 的 fp32 路径）。判题文件不导入 peft。

## 复杂度

选择阶段对 `r × (d_in/vl)` 与 `(d_out/vl) × r` 个长度 `V` 的 logit 向量各做一次 top-k 与加权求和，`O((r·d_in + r·d_out)/vl · V)` 时间（同值稳定的整序实现是 `O(V log V)`，堆选择是 `O(V log topk)`）；拼 `delta` 是 `O(d_out · d_in)`；前向若走未合并路径只需 `O(N · r · (d_in + d_out))`，比物化 `delta` 后乘一遍更省。可训练参数量是 `2` 侧 logits 的 `r·(d_in + d_out)·V/vl` 加上共享库 `V·vl`，与 `d_out × d_in` 无关——这就是"极端压缩"的来源。
