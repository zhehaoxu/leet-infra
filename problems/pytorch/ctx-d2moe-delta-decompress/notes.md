# 详解：D2-MoE Delta 解压缩

## 出处

论文 D²-MoE（arXiv:2502.17298，Hao Gu 等，HKUST + Birmingham + AISpeech，官方仓 `lliai/D2MoE`）。动机：MoE LLM 的专家权重相关性极高——把每个专家表示为「共享 base + 独有 delta」，base 一次存储、delta 低秩压缩，存储从 \(m \cdot out \cdot in\) 降到 \((out \cdot in) + m \cdot k(r{+}c)\)。三段法：① Fisher 信息加权合并出 base；② delta 白化 SVD 低秩压缩（可跨专家共享 V）；③ base 半动态结构化剪枝（本题不含——打分依赖动态 probe 输入，公式在 pruning_module.py，属动态路径）。在 Mixtral/Phi-3.5/DeepSeek/Qwen2-MoE 上 40~60% 压缩率下超基线 13%+。

## 官方实现的逐行翻译

```python
# merge_mixtral.py: merge_experts, fisher 分支（w1 示例）
fisher_scale_w1[j] = hessian[key_w1] * scaling_factors_ft_list[j]   # fisher × expert_freq
fisher_scale_w1 = [f / sum(fisher_scale_w1) for f in fisher_scale_w1]
self.Wmean1.weight.data = sum(experts[j].w1.weight * fisher_scale_w1[j] for j in range(8))
# delta: module.experts[j].w1.weight - self.Wmean1.weight

# svd_delta: 参数预算秩 + svdllm 白化
num_s_after_trunc = int(W.shape[0]*W.shape[1]*ratio / (W.shape[0]+W.shape[1]))
W_scale = W @ svd_scale                      # 右乘白化
U, S, VT = torch.linalg.svd(W_scale)
truc_v = VT[:k] @ cal_scale_inv(svd_scale)   # 去白化
svd_u = U @ sqrt(Σ); svd_v = sqrt(Σ) @ truc_v  # sqrt(Σ) 两侧拆分

# delta_share_V: 行堆叠共享 V
# 堆叠 (8*d, n) 一次 SVD -> 共享 V，各专家 U 行切片；
# 共享 U 时按 freq 加权平均；共享 scale 时按 freq 加权平均 scale
```

三个关键口径：

1. **Fisher×freq 双加权**：合并权重是 fisher 与专家路由频率的乘积再归一化——高频且重要的专家主导 base；fisher=0 的专家系数为 0（base 不含它，信息完整保留在其 delta 里，delta 不丢）。
2. **参数预算秩不是满秩**：\(k = \lfloor rc \cdot ratio/(r+c) \rfloor\) 把「压缩后参数量占比」映射成秩——ratio=1 时 \(k = rc/(r+c) < \min(r,c)\)，所以任何 ratio 下都是有损压缩，没有「满秩重建」锚点可言（互证改为与白化域截断 SVD 重建的恒等）。
3. **√Σ 两侧拆分 + 去白化右乘**：`svd_u = U√Σ`、`svd_v = √Σ·Vt·scale⁻¹`，乘积 = \((M \cdot scale)_k \cdot scale^{-1}\)——白化只改变 SVD 的度量，重建时右乘 inv(scale) 回原域。共享 V 时 V 侧同样带 √Σ，U 行切片也带 √Σ，\(U_j V\) 恰为堆叠低秩重建 \(Us\,\mathrm{diag}(\Sigma)\,vts\) 的行切片。

**合并前向**（shared_infer）：base 路径对全部 token 算一次，共享 V 也一次，输出 = base SwiGLU + delta 低秩 SwiGLU 同位相加；路由 softmax/top-k/加权在 MoE 层原样保留。

## 易错点

1. **双加权缺一**：漏 freq（m-merge-nofreq）或漏归一化（m-merge-nonorm）都改变 base 与全部 delta；fisher 与 freq 写成除法（m-merge-swap 钉）。
2. **delta 方向**：\(W - base\) 不是 \(base - W\)（m-delta-rev 钉——方向反了前向立刻数值错）。
3. **rank 账两个坑**：int 截断不是 round（m-rank-round 钉）；分子是 \(r \cdot c \cdot ratio\)、分母是 \(r+c\)（写反成 \((r+c) \cdot ratio / rc\) 得到 0（m-rank-flip 钉））。
4. **√Σ 拆分双侧**：只放一侧则乘积少乘 Σ（m-svd-oneside 钉）；去白化漏掉（m-svd-noinv）或多乘一次 scale（m-svd-scale 钉）。
5. **共享 V 三件**：V 带 √Σ（m-shv-nosq 钉）、U 行切片起点 \(j \cdot r\)（m-shv-offset 钉——j=0 时错位不可见，必须钉 j≥1 段）、U 切片也带 √Σ（m-shv-unosq 钉）。
6. **SwiGLU 结构**：ReLU 作用于 gate 不作用于 up（m-fwd-actswap 钉）；双通道是逐元素乘不是加（m-fwd-add 钉）；delta 低秩路径与 base 同位相加、缺一路径即错（m-fwd-nodelta 钉）。
7. **等价变异剔除记录**：dispatch 里对已 int 的返回值再做 int 恒等（m-disp-rank 实测 SURVIVED 剔除）。

## 对拍口径

锚点由沙箱 dump（numpy float64，seed 808）：in=6/h=8、2 专家、freq=[0.7,0.3]、w1/w3/w2 各配 fisher。互证四路——① 白化抵消：√Σ 拆分乘积 ≡ \((M \cdot scale)_k \cdot scale^{-1}\)；② 共享 V 恒等：\(U_j V\) ≡ 堆叠矩阵 \(Us\,\mathrm{diag}(\Sigma)\,vts\) 低秩重建的行切片；③ 合并前向 ≡ base+delta 重建权重的普通 SwiGLU；④ 零 delta 时 forward ≡ 纯 base SwiGLU。rank 账 8 配置（两种形状 × ratio 0.2/0.5/0.9/1.0）。16 变异 15 CAUGHT + 1 int 恒等等价剔除。

## 复杂度

合并离线一次 \(O(m \cdot out \cdot in)\)；压缩后存储 \(out \cdot in + m \cdot k(r{+}c)\)，delta_ratio=0.8 时约省 \(m(1-ratio) \cdot out \cdot in\)；共享 V 再省 \((m{-}1) \cdot k \cdot c\)。前向 base 路径对全 token 共享、delta 低秩路径 \(O(BN(rk + kc))\) 仅对选中专家生效。
