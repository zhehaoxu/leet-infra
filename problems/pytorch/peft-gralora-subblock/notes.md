# 详解：GraLoRA 子块低秩前向与分块合并

## 出处

官方仓 `SqueezeBits/GraLoRA`（arXiv:2505.20355，NeurIPS 2025 Spotlight）。`model.py`/`config.py` 均为 PEFT 标准脚手架（merge 明确 `raise NotImplementedError`），确定性核心只有 `peft/src/peft/tuners/gralora/layer.py` 两处：`update_layer` 的布局/初始化账与 `Linear.forward` 的双 einsum 链。该实现已进入 HuggingFace PEFT 官方 `tuners/gralora`。官方训练配置（`scripts/train/llm.llama-3_1-8b+data.code.sh`）：Llama-3.1-8B，r∈{16,32,64,128}，gralora_k=2（r≤32）否则 4，hybrid_r=8（仅 r=16）、其余 0——即默认配置就是"纯 granular、无旁路"。

## 官方 einsum 链的逐行翻译

```python
h = einsum("blni,nir->blnr", x.view(B,L,N,in//N), A)          # A: (N, in//N, rank)
h = h.view(B,L,N,N,sub).permute(0,1,3,2,4).reshape(B,L,N,N*sub)
o = einsum("bljr,jro->bljo", h, B)                            # B: (N, rank, out//N)
result += scaling * o.reshape(B,L,-1)
```

- 第一个 einsum：输入第 \(n\) 段 \(x_n\) 过 \(A_n\)，输出 \((B,L,N,r_\text{gra})\)。注意 A 存储时 transpose(1,2) 过：每块是 \((d_\text{in}/k, r_\text{gra})\)——**每段的 A 覆盖完整 \(r_\text{gra}\) 秩**。
- view/permute/reshape 三连：rank 轴切成 \((N, \text{sub})\)（行优先，`rank_idx = 段*sub + 偏移`），permute 把"段"轴换到前面、"输入块"轴与"偏移"轴复合 reshape 成新的 rank 轴。结果 `mid[bl, 段j, n*sub+off] = x_n·A[n][:, j*sub+off]`。
- 第二个 einsum：`y[j,o] = Σ_r mid[j,r]·B[j,r,o]`，r 求和（mid 的复合索引 ↔ B 的 rank 行）、o 自由（\(d_\text{out}/k\)）。B 每块是 \((r_\text{gra}, d_\text{out}/k)\)。

## 等价分块公式（结构 intuitively 可见）

\[\Delta W^{(j,n)} = B_j[n\cdot s{:}(n{+}1)s,:]^{\top} A_n[:,\, j\cdot s{:}(j{+}1)s]\]

块 \((j,n)\) 的秩 \(=s=r_\text{gra}/k\)。B 的第 \(n\) **行段**配 A 的第 \(j\) **列段**——A 在输入块方向共享、B 在输出段方向共享。参数量 \(N\cdot r_\text{gra}\cdot\frac{d_\text{in}}{N}+N\cdot r_\text{gra}\cdot\frac{d_\text{out}}{N}=r_\text{gra}(d_\text{in}+d_\text{out})\)，与同秩普通 LoRA 相同。这正是论文卖点：同参数预算下 bottleneck 宽度从 \(r_\text{gra}\) 变成 \(\text{sub}\)，每块更"瘦"且独立。

## 易错点

1. **scaling 用总秩 \(r\) 而不是 \(r_\text{gra}\)**。官方 `self.scaling[adapter_name] = gralora_alpha / r`——hybrid_r>0 时（官方 r=16、hybrid=8、\(\alpha=32\) 的配置）scaling=2 而不是 4。这与普通 LoRA 的 \(\alpha/r\) 口径保持一致。
2. **permute 交换的是"段"与"输入块"**。漏掉 permute 时 reshape 合并的是 \((n,\text{off})\) 保持原序，mid 的复合索引变成 \(n\cdot\text{sub}+\text{off}\) 不再与 B 的行段语义配对——数值全错（变异 m-permute/m-noview 钉死）。
3. **B 的零初始化**。B 与 \(B_g\) 全零 ⇒ 训练起点 \(\Delta W=0\)（A kaiming 非零）。`param_layout` 的 `zero_init` 集合把这一点钉成可测账。
4. **校验是 `r > 0` 与 `r_\text{gra} % k == 0` 两条**（后者官方是 assert）。注意校验的是 **gralora_rank**（\(r-r_\text{hyb}\)）能否被 k 整除，不是 \(r\) 本身——hybrid 旁路的秩不参与分块。
5. **hybrid 旁路同乘 scaling**。官方 forward 里 hybrid 分支 `result += scaling * Bg(Ag(dropout(x)))`，与 gralora 分支共用同一个 scaling。

## 对拍口径

锚点由沙箱三轨互证后 dump（numpy float64）：

- **Track A**：官方 einsum 链逐字直译（np.einsum 与 torch.einsum 下标语义一致）。
- **Track C**：显式四重循环按 `y[j,o] = Σ_{n,b} (x_n·A[n][:,j·sub+b])·B[j][n·sub+b, o]` 展开——不依赖 einsum 引擎，纯下标语义。
- **Track D**：分块公式 `ΔW[j,n] = B[j][n·sub:(n+1)·sub,:].T @ A[n][:, j·sub:(j+1)·sub]` 组装稠密 \(\Delta W\) 后走 \(x\Delta W^\top\)。

三轨在 4 组尺寸（含 hybrid>0、in≠out、k=3/4）全 1e-14 一致后，以 Track A 输出写死 EXPECTED。测试里 `check_block_delta` 再做一次**测试内互证**（x@ΔWᵀ·scaling ≈ forward 锚点），`check_forward_delta` 钉 einsum 链数值，`check_full_forward` 钉 hybrid 分支及其两条 ValueError 守卫，`check_param_layout` 钉 shape 与 zero_init 集合，`check_split_dims` 钉五元组与两条校验。

## 复杂度

forward_delta 与官方同为两次低秩 contraction：\(O(BL\cdot d_\text{in}\cdot r_\text{gra} + BL\cdot r_\text{gra}\cdot d_\text{out})\)，与普通 LoRA 前向同阶；block_delta 组装 \(\Delta W\) 是 \(O(d_\text{out}d_\text{in})\) 的稠密写出（仅在合并/分析时需要，官方 merge 未实现，训练态永远走 einsum 链）。
