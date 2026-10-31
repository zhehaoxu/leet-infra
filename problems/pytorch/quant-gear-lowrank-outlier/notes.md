# 解题说明 · GEAR 低秩幂迭代误差补偿

## 一句话

先把每行的两端极值换成行均值送去量化、量化完再原样写回，然后对量化残差跑一个「QR 只放在最后一轮」的幂迭代，`output = quantized + low_rank`；补偿的数学身份是正交投影，误差只会变小。

## 为什么 `low_rank` 是投影

离群元换成行均值（而不是 0 或截尾均值）只是给量化器一个不会撑爆组内动态范围的占位，原值最后写回，这些位置等于免费精确表示、残差恒为 0；真正进幂迭代的是剩下的误差矩阵 \(A=X-\hat X\)。

循环体里两次 QR 都写成 `if i == loop - 1`，其余轮次只是裸的子空间迭代 \(Q\leftarrow AP\)、\(P\leftarrow A^{\top}Q\)；收尾时 \(Q\) 的列两两正交且 \(P=A^{\top}Q\)，于是 `low_rank` \(=QP^{\top}=QQ^{\top}A\) 是 \(A\) 到 \(\operatorname{span}(Q)\) 的正交投影，\(A-\text{low\_rank}=(I-QQ^{\top})A\) 与 \(\operatorname{span}(Q)\) 正交，\(\lVert A-\text{low\_rank}\rVert\le\lVert A\rVert\) 恒成立。`loop=1` 时子空间是 \(\operatorname{span}(A\,\mathrm{qr}(P_0).Q)\)，`loop` 越大越贴近 \(A\) 的主右奇异子空间，误差单调下降（示例形状 0.6166 → 0.3873 → 0.1925 → 0.1857 → 0.1856）；也正因如此，`q_base` 在第一轮就被 `A @ P` 覆盖，永远不影响结果。

## 三个易错口径

1. **名额是整行的**：`n_out = int(int(B*H*L*D*sparsity)/B/L/2)` 双重截断，除式里没有 `H`，两端各取 `n_out` 个；写成「每个 head 各 `n_out` 个」或漏掉外层 `int` 都会错。
2. **均值含离群元**：替换值是原始行的均值，不是摘掉两端之后的截尾均值；`[-100, 0, ..., 0, 100]` 整行被换成 0。
3. **级数写反**：分母是 `2**bits - 1` 个间隔（上游 `gears_tokenQ` 笔误成 `quantize_bit ** 2 - 1`，只在 bits=2/4 时巧合相同）；常数组不做除法、反量化成 `mn`。

## 判分点

- 掩码：合并行两端各 `n_out` 个候选、同分取更小下标；下标重合时 True 少于 \(2n_{\text{out}}\)（整行常数就只剩 `n_out` 个）；
- `quantized` 的离群位置等于原值、`residual` 的离群位置等于 0；组内其余点走仿射格点，取整是半偶（题面示例 row0 的 head0 组里 \((2-1)/(2/3)=1.5\) 落到偶数 2）；
- `low_rank` 逐 \((b,h)\) 独立、秩不超过 `rank`，且 `||kv - output|| <= ||kv - quantized||`；换 `q_base` 或把 `p_base` 取负/缩放都不动它，换 `p_base` 的方向才会换子空间；
- 六个键齐全、dtype 跟随输入、非连续视图与连续副本同解、三个输入张量不被就地修改。

## 边界

- `sparsity = 0` 时 `n_out = 0`，退化成「纯量化 + 低秩补偿」、掩码全 False（示例形状仍把 2.108 的误差降到 0.745）；`sparsity` 再大也只到半行，不会越界。
- `group_size = H*D` 时整行合成一组；`bits = 1` 只剩 `mn`/`mx` 两个格点，组极值仍精确往返。
- 题面示例的 `p_base` 全 1、`rank = 1`，QR 给的方向就是 \((1,\dots,1)/\sqrt{D}\)；`rank >= 2` 的常数基其 QR 第二列属于补空间、跨实现符号不定，测试因此改用满秩随机基。

## 复杂度

设合并行宽 \(C=H\cdot D\)：每行两次 stable argsort \(O(C\log C)\)、量化 \(O(BLC)\)；幂迭代每轮每个 \((b,h)\) 两次矩阵乘 \(O(LD\cdot\text{rank})\)，最后一轮再加两次 QR，合计 \(O(\text{loop}\cdot B\,H\,L\,D\,\text{rank})\)。

## 出处

`opengear-project/GEAR`（arXiv 2403.05527），本地副本 `/tmp/qsrc/GEAR/GenerationBench/GenerationTest/GEARLM/Simulated/compress_function.py`：

- `gears_tokenQ` L297-333：元素数与名额 L300-303（`sparsity_pertoken = int(sparsity_num / batch / seq_len/2)`）、合并成 token 行 L304-306、两端 `torch.topk` L309/L311、行均值 L312、两次 `scatter_` 替换 L316/L319、量化调用 L321-322、原值写回 L327-328；
- `fake_groupwise_token_asymmetric_quantization_cluster` L132-160：切组 L142、组内极值 L144、`scale = (mx - mn) / cluster_num` L147、`(x - mn)/scale` L148、`F.relu` L149、`round_()` L150、反量化 L151；
- `fake_poweriteration_group` L69-98：`for i in range(loop)` L86、最后一轮的两次 `torch.linalg.qr(...).Q` L87-88 与 L90-91、`q_base[0] = input @ p_base[0]` L89、`p_base[0] = transpose(input, 2, 3) @ q_base[0]` L92、`input = q_base[0] @ transpose(p_base[0], 2, 3)` L93；
- `gearslkivi_tokenQ_new` L204-211：L207 调 `gears_tokenQ`、L209 `error = cloned_input - output`、L210 幂迭代、L211 `return output + error_lr`；
- `compress_insert_function` L421-584：GEAR 分支 L466-495——K 走 channel-wise 的 `gearslkivi_channelQ_new`（L477）再 `.half()`（L486），V 走 token-wise 的 `gearslkivi_tokenQ_new`（L487）再 `.half()`（L495）。

**本题自己的补充**（与上游的偏差，别记到论文头上）：

1. 上游 `gears_tokenQ` L322 把级数写成 `quantize_bit ** 2 - 1`（`gears_channelQ` L286 同错），它与意图中的 `2 ** bits - 1`（同文件 L24、L55 的非 cluster 版本）只在 `bits ∈ {2, 4}` 时相等；本题统一用 `2 ** bits - 1`。
2. 上游在函数内部用 `torch.rand` 现生成两个基（L83-84，L72-73 是被注释掉的旧写法），本题改成显式入参 `p_base`/`q_base`，结果可复现，且可证 `loop >= 1` 时 `q_base` 不起作用。
3. 上游常数组直接除以 0 得到 NaN（L147 没有兜底），本题 `scale == 0` 时 code 恒 0、反量化成组 `mn`。
4. 上游靠 `torch.topk`（L309/L311），同分顺序未定义；本题钉成「同分取更小下标」作为确定性契约。
5. 上游末尾把结果 `.half()`（L332）；本题输出 dtype 跟随输入，内部统一 float64。
6. 只考 token-wise（V 侧）变体：channel-wise 的 K 变体沿另一根轴切组（`gears_channelQ` L269-271 先 `permute(0, 1, 3, 2)`），不在本题范围。

## 相关题目

- `quant-kivi-kv-asym`：本题第 2 段的量化器就是 KIVI 式 token-wise 非对称分组量化，但 KIVI 到此为止，不管剩下的误差怎么补。
- `quant-qaq-error-bitwidth`：同样摘 outlier，QAQ 让它们原值透传、不做补偿，GEAR 还要用低秩项把非 outlier 那部分误差也吃掉一块。
- `quant-rtn-gptq-error`：GPTQ 用二阶信息在量化**当下**逐列吸收误差，本题在量化**之后**用秩-`rank` 投影做事后补偿。
