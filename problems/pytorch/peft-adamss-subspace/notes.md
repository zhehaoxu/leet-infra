# 详解：AdaMSS 自适应多子空间散射适配

## 出处

论文 AdaMSS（NeurIPS 2025，Zheng 等，官方仓 `jzheng20/NeurIPS2025-AdaMSS`，已入 HF PEFT `tuners/adamss`）。出发点：orthogonal fine-tuning 一系（BoFT/OFT/diagonal-FFT 谱系）把 \(\Delta W\) 约束在单一低秩正交子空间里，对所有奇异方向一视同仁。AdaMSS 的观察是 SVD 谱成分的重要度不均——于是把 \(r\) 个成分按 KMeans 聚成若干组，每组独立配低秩因子、独立定秩（甚至整组冻结），在总参数预算内把容量按谱结构自适应分配。冻结的 \(\mathrm{newB} = V_r'^\top\) 保证 \(\Delta W\) 始终落在预选子空间的直和里。

## 官方实现的逐行翻译

`layer.py` 的确定性主干（去训练/设备细节）：

```python
# update_layer 末尾: scatter 索引 = 按训练子空间顺序拼接各段
self.scatter_index[adapter_name] = torch.cat(
    [self.seg_result[adapter_name][self.train_subspace_index[adapter_name][i]]
     for i in range(self.num_subspaces[adapter_name])], dim=0).long()

# forward: bias 存在时输入拼全 1 列（输出维不加一）
if self.get_base_layer().bias is not None:
    ones = torch.ones(*x.shape[:-1], 1, ...)
    newx = torch.cat((x, ones), dim=-1)
projected = F.linear(newx, self.adamss_newB[...])          # (..., r)
for i in trained_subspaces:
    h = projected @ A_i.T
    outs.append(h @ B_i.T)                                  # (..., seg_i)
delta = torch.zeros(..., out_features).scatter(1, index, cat(outs))

# merge 主干 (get_delta_weight / get_delta_bias 共用):
blocks = [B_i @ A_i for ...]
combined = torch.cat(blocks, dim=0) @ self.adamss_newB[...]  # (Σ seg_i, in+1)
delta_weight = torch.zeros(out_features, in_features_plus_1)
delta_weight[scatter_idx, :] = combined
# get_delta_weight: output_tensor = delta_weight[:, :-1]
# get_delta_bias:   output_tensor = delta_weight[:, -1]
```

三个关键口径：

1. **seg 是 out 维 SVD 成分的划分**（覆盖 \(0..r{-}1\)），不是 in 维分割；\(r \le out\)，未覆盖的输出行（成分不足处）delta 恒零。
2. **rank 截断双护栏**（固定 rank 模式）：`actual_rank = min(len(seg_indices), subspace_rank)`，为 0 时置 1——空段或 subspace_rank=0 都不会产出零秩子空间。（动态 rank 模式另有 Gram 矩阵 SVD 阈值筛，不在本题账内。）
3. **bias 取列不取行**：bias 并入 SVD 空间的方式是输入侧拼全 1 列（\(W'\) 的最后一列乘 1），所以 merge 出的 \((out, in{+}1)\) 增广 delta 要按**列**拆分；`get_delta_bias` 与 `get_delta_weight` 全流程相同，仅最后一步剥列不同。

**forward 与 merge 同源**：forward 是 \(\mathrm{cat}[\,(B_iA_i)\,]\cdot \mathrm{newB}^\top\) 的分步形式（先投影再逐段上升），merge 是先升维再一次性乘 \(\mathrm{newB}\) 后散射——两者由 \(x\, @\, W_\Delta^\top\ (+b_\Delta)\) 互证。

## 易错点

1. **输出维不 +1**：bias 折进输入侧全 1 列，forward 输出仍是 out_features；把 +1 加到输出侧（dual +1）是本题埋的经典 shape 错（m-fwd-outp1 钉）。
2. **scatter 顺序 = 训练子空间顺序**：`build_scatter_index` 按 train_idx 顺序拼接，反序或忽略 train_idx 全选都会错位（m-scatter-rev/m-scatter-all 钉）；按位置而非组号取段也错（m-scatter-pos 钉）。
3. **rank_clamp 双界**：去下限（空段返 0）、去上限（超 subspace_rank）、下限提到 2 都可观测（m-rank-nofloor/m-rank-nocap/m-rank-floor2 三钉）。
4. **merge 各 block 各自乘 newB**：`cat([B_i@A_i]) @ newB` 与「先 cat 后乘」同、与漏乘或块间交换 A/B 因子都数值可钉（m-dw-nonewb/m-dw-ba 钉）；blocks 反序拼接 = scatter 行错位（m-dw-catrev 钉）。
5. **散射写行，不是左对齐写**：`dW_full[scatter_index, :] = combined`；左对齐赋值在部分训练 fixture（scatter=[0,1,4]）下立刻错位（m-dw-rowassign/m-fwd-noscatter 双钉）。
6. **split 是列拆分**：bias 取 `[:, -1]` 不是 `[-1, :]`（m-split-row 钉）；weight 剥 `[:, :-1]` 不是 `[:, 1:]`（m-split-off 钉）；with_bias 分支整体短路也会被 forward_bias 的互证抓（m-split-nobias 钉）。
7. **with_bias 的新投影维度耦合**：newB \((r, in{+}1)\) 只能配 \([x\,|\,1]\) 输入；no-bias 输入直接配含 bias 列的 newB 是 shape 违规（测试钉 ValueError 护栏）。
8. **未训练组行恒零**：部分训练时 forward 与 merge 在未训练行全零——fixture 特意用「跳过中间组」钉散射击中语义（forward_partial/dW_partial 锚点）。

## 对拍口径

锚点由沙箱 dump（numpy float64，seed 31）：out=6/in=5/r=5、三组 (2,2,1)、subspace_rank=2。三轨互证——① no-bias：forward \(= x\, @\, \mathrm{dW}^\top\)；② with-bias：\(x\, @\, W^\top + b =\) forward，且 \(x_{aug}\, @\, \mathrm{dW}_{full}^\top\) 等价形态复核；③ 部分训练：未训练行/列双侧零断言。rank_clamp 四配置（sr=1/2/5 + 空段）。23 变异全 CAUGHT，无等价剔除。

## 复杂度

forward \(O(BN\cdot r + \Sigma_i\,\mathrm{rank}_i\cdot \mathrm{seg}_i)\) 主项在冻结投影 \(x_{aug}\ @\ \mathrm{newB}^\top\)；merge 多一次 \((\Sigma \mathrm{seg}_i, r)\ @\ (r, in{+}1)\)。可训练参数 \(\Sigma_i\,\mathrm{rank}_i(\mathrm{seg}_i + r) < r\cdot(out+r)\)——比整体训练左因子 UU 更省，且按谱结构分配。
