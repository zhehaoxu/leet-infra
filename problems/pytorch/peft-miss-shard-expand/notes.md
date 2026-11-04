# 详解：MiSS 分片共享扩展与前向合并

## 出处

论文 MiSS: Revisiting the Trade-off in LoRA with an Efficient Shard-Sharing Structure（arXiv:2409.15371 v10，ICLR 2026；前身 Bone→DiSHA→MiSS 三代演化）。官方仓 `Joluck/MiSS`（同 JL-er/bone），2024.11 合入 HF PEFT（`tuners/miss`）。官方仓的 `flashop/miss.py` 是 TileLang GPU kernel（不出题），但其 `__main__` 里的 PyTorch 参考语义（`a.reshape(…, D//R, R).sum(dim=-2) @ b`）与本题主干账一致。README 的效率表：MiSS 空间 \(O(dr)\)、前向 \(O(bldk)\)，均严格小于同秩 LoRA 的 \(O((d+k)r)\)——**参数更少且算得更快**，这是它相对 VeRA/DoRA 系的核心卖点。

## 官方实现的逐行翻译

**前向（MissLinear.forward，非 bat）**：

```python
miss = self.miss_block[adapter]            # (r, out) 零初始化
if self.miss_fn[adapter] == "mini":
    miss = miss.repeat(1, out_features // self.miss_mini_r[adapter])
r = miss.size(0)
if x.size(-1) % r != 0:
    x = F.pad(x, (0, (r - x.size(-1) % r) % r))
result += torch.sum(dropout(x).reshape(*x.shape[:-1], x.size(-1)//r, r), dim=-2) @ miss
```

关键在 reshape 后 `sum(dim=-2)`：\((\dots,n_\text{blocks},r)\) 里 dim=-2 是**片索引**、dim=-1 是**片内偏移**，求和消掉片索引 → 输出 \(r\) 维（片内偏移对齐的跨片求和）。这与 flashop kernel 的 `sum_A[s,j] = Σ_i A[s, i*R+j]` 完全同构。

**merge（get_delta_weight_miss，非 bat）**：

```python
W_blocks = W.reshape(-1, W.size(1)//r, r).permute(1, 2, 0)   # (nb, r, out)
output = (W_blocks + sign * miss_B).permute(2, 0, 1).reshape(W.shape)
```

`miss_B (r, out)` 广播加到每个块——等价于 \(\Delta W[:,\, i\cdot r+j] = B[j,:]\) 对每个块 \(i\)，即 \(\Delta W=\text{tile}(B^\top,(1,nb))\)。两形态一致性可直接验证：\(\Delta W\cdot x = B\cdot\sum_i x_i\)（\(x_i\) 为第 \(i\) 片）恰好等于前向的 `shard_sum(x) @ B`。

**bat（get_delta_weight）**：

```python
W = orig_weight.reshape(out//r, r, in//r, r).permute(2, 0, 1, 3)  # (nb_in, nb_out, r, r)
result = W @ miss_B + miss_B          # miss_B[j]: (r, r)，逐输出块
dW = result.permute(1, 2, 0, 3).reshape(out, in)
```

unmerge 走 \((W-B)(I+B)^{-1}\)（精确逆），本题不取。

## 易错点

1. **聚合轴**：`sum(dim=-2)` 消掉的是片索引，输出保留片内偏移维 \(r\)。写成 `sum(dim=-1)` 得到 \((\dots,n_\text{blocks})\) 维——形状对不上 miss_B \((r,out)\) 时崩，碰巧对上时数值全错（变异 m-sum-axis/m-reshape-swap 双双钉死）。
2. **pad 账**：pad 是右侧零填充到 `n_blocks*r`，零元不改变跨片求和的聚合值（只影响形状）；`pad_size=(r-in\%r)\%r` 在整除时为 0。
3. **mini 的 repeat 方向**：沿**输出维** repeat（`(r, mini_r) -> (r, out)`），且 repeat 次数是 `out//mini_r`；错成 `+1` 会在维度或数值上暴露（m-mini-repeat 钉）。
4. **bat 的加性项**：\(W_\text{块}B_j+B_j\) 的第二个 \(+B_j\) 是 MiSS 相对 Bone 的结构修正（破共线），漏掉后 \(\Delta W\) 块整体偏移（m-bat-noadd 钉）；bat 的 reshape 四段顺序 \((nb_\text{out},r,nb_\text{in},r)\) 与 permute 后的块归属必须配对——方阵 fixture（in=out）下 reshape 顺序变异不可观测，fixture 必须取非方阵（本题主锚 F5 取 \(in=12,out=8\)）。
5. **零初始化**：三种形态的 `miss_block` 全部零初始化（`zero_init=True`）——训练起点 \(\Delta W=0\)，与 LoRA 的「B 零 A 随机」不同，MiSS 连 A 都没有。

## 对拍口径

锚点由沙箱三轨互证后 dump（numpy float64）：Track A=PEFT forward 直译（pad+reshape+sum+matmul）、Track B=merge 的 ΔW tile 展开走 \(x\Delta W^\top\)、Track C=显式循环 \(dW[b,l,o]=\sum_{n,j}x[b,l,n\cdot r+j]\cdot B[j,o]\)；bat 块账另以 \((j{=}0,i{=}0)\) 块抽查核对。三轨 1e-14 一致后写死 EXPECTED。测试内再做**形态互证**（F1/F2/F4：`x @ dW.T ≈ forward`）。五组 fixture 钉：F1 整除、F2 k=3、F3 触发 pad、F4 mini、F5 bat（非方阵）。校验路径共四条 raise（r≤0 / r>in / bat 不整除 / mini 不整除）+ mini 缺 out_features 两条。

## 复杂度

前向 \(\sum\) 聚合 \(O(in)\)、matmul \(O(r\cdot out)\) 每位置——总 \(O(BL\cdot(in+r\cdot out))\)，严格低于普通 LoRA 的 \(O(BL\cdot(in+out)\cdot r)\)；expand \(\Delta W\) 是 \(O(out\cdot in)\) 的 tile 写出（仅 merge/分析需要，训练态走前向 delta）。bat 账为逐块 \(r\times r\) matmul，共 \(nb_\text{in}\cdot nb_\text{out}\) 次。
