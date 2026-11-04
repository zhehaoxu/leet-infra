# 详解：TinyLoRA 冻结 SVD 因子与随机投影组合

## 出处

论文 Learning to Reason in 13 Parameters（arXiv:2602.04118，Meta，Morris/Mireshghallah/Ibrahim/Mahloujifar）。结论：RL（GRPO）微调的本质是**激活**而非学习——Qwen2.5-8B 用 13 个 bf16 参数（26 字节）达 GSM8K 91%，SFT 则需要 100–1000× 的更新量。实现谱系：LoRA-XS（arXiv:2405.17604，ΔW=B·R·A、R r×r 可训）→ TinyLoRA 把 R 再分解为固定随机投影的线性组合。官方参考实现 ANRGUSC/TinyLoRA（2026-02 public research pack）；HF PEFT PR #3024（2026-02 合并，`tuners/tinylora`）是本题口径来源。

## 官方实现的逐行翻译

**SVD 分摊（`_init_svd`）**：

```python
U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
actual_r = min(r, min(weight.shape))
sqrt_S = torch.sqrt(S[:actual_r])
tinylora_A = (sqrt_S.unsqueeze(1) * Vh[:actual_r, :]).contiguous()   # (r, in)
tinylora_B = (U[:, :actual_r] * sqrt_S.unsqueeze(0)).contiguous()    # (out, r)
```

√S 均分两侧（对比 LoRA-XS 论文的同款惯例）：\(B=A^\top\) 结构上都是 \(\sqrt S\) 缩放的因子，第 \(i\) 对因子范数恰为 \(\sqrt{S_i}\)——避免一侧因子范数失衡。SVD 在 float32 下算（数值稳定），锚点 dump 时同样走 float32 再转 float64。

**投影初始化（`_init_projection`）**：

```python
seed = base_seed + self._layer_idx          # 注入顺序的确定层号
gen = torch.Generator().manual_seed(seed)
P = torch.normal(0.0, 1.0 / r**0.5, size=(u, r, r), generator=gen)
```

P 存 `BufferDict`（冻结，不进 optimizer）；`save_projection=False` 时 checkpoint 不存 P、加载时按 seed 重放。分布 N(0, 1/r) 是 JL 引理惯例（注释原文）。

**R 组合与前向**：

```python
R = torch.einsum("i,ijk->jk", v, P)         # (r, r)
h = F.linear(x, A); h = F.linear(h, R); delta = F.linear(h, B)
```

即 delta \(=x A^\top R^\top B^\top\)，与 \(\Delta W=BRA\) 的 \(x\Delta W^\top\) 精确一致（测试互证）。v 存模型级 `nn.ModuleDict`（同组层共享同一 Parameter 对象，PyTorch 不重复注册），层内只存 `_tinylora_v_key` 字符串键。

**weight_tying 分组（model.py `_create_and_replace`）**：

```python
num_groups = max(1, round(num_target_layers * (1.0 - weight_tying)))
group_size = max(1, num_target_layers // num_groups)
group_idx = min(layer_idx // group_size, num_groups - 1)
```

三重钳制各司其职：`max(1,·)` 保 wt=1 时组数为 1（`round(0)=0` 会 0 组）；`max(1, n//groups)` 保 group_size 至少 1；`min(·, groups-1)` 保 group_idx 不越界（n 不整除组数时尾组收纳余数层）。锚点抽查：n=5, wt=0.4 → round(3.0)=3 组、size=5//3=1、indices=[0,1,2,2,2]；n=7, wt=0.5 → round(3.5)=4（Python banker's rounding：3.5→4）、size=1、indices=[0,1,2,3,3,3,3]。

## 易错点

1. **√S 均分两侧，不是全给一边**。只给 B（或只给 A）时重构 \(B@A\neq U_r\mathrm{diag}(S_r)V_r\) 且范数抽查立刻失败（m-split-side-a/b 双钉）。
2. **秩截断在 SVD 短边**：\(r\) 可以大于 \(\min(out,in)\)（F2 给 r=8、5×4 权重），`actual_r=min(r, min(shape))` 截断——漏掉后 S 切片越界或 A/B 形状错。
3. **P 的分布是 N(0, 1/r)**（std=1/√r 不是 1/r），且数值路径是 **float32 生成 → 转 v 的 dtype**——锚点按官方路径 dump，float64 直接生成的高斯样本逐位不同。
4. **R 组合用 einsum "i,ijk->jk"**：v[i] 加权第 i 个 P[i] 后对 i 求和；输出下标顺序 (j,k) 写反形状虽同为 (r,r) 但数值转置（m-r-einsum 钉）。
5. **分组三重钳制**：`round` 不能换 `int`（round(3.5)=4 vs int=3，m-tie-round 钉）；`max(1,·)` 去掉后 wt=1 直接 0 组除零（m-tie-max 钉）；`min(·, groups-1)` 去掉后尾组越界（m-tie-clip 钉）。
6. **forward 链顺序 A→R→B**，每步都是 `@·.T`（F.linear 语义）；R 的转置漏掉在 \(R\) 非对称时立刻数值错（m-fwd-trans 钉）。

## 对拍口径

锚点由沙箱 dump（.venv-eval torch float64）：F1 6×8 权重 r=4 满分解 + u=6/seed=7 投影 + v 显式向量；F2 5×4 权重 r=8 截断。互证两式：`x@dW.T == forward`（LoRA-XS 链 vs 合并形态）、`B@A == U_r diag(S_r) V_r`（分摊无损）。tie 账五组配置（wt∈{0,0.4,0.5,1}、n∈{5,6,7}）含钳制边界。测试另做：同 seed 投影重放 `torch.equal`、B 列范数/A 行范数 = √S_i 抽查、R 线性组合显式求和对照。

## 复杂度

compute_R 是 \(O(u\cdot r^2)\) 的加权求和；delta \(B@R@A\) 两连乘 \(O(out\cdot r^2+r^2\cdot in)\)；前向三次 linear 与 LoRA 同阶 \(O(BL(in\cdot r+r^2+r\cdot out))\)。可训练参数恰为 groups×u——论文 budget=13 即靠 u 与 tie 分组凑出。
