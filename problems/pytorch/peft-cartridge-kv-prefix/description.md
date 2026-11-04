# PEFT：Cartridge 直接参数化的 KV 前缀适配器

## 题目描述

**Cartridge**（arXiv:2506.06266，Stanford Hazy Research，已入 HF PEFT `tuners/cartridge`，PR #2953）把**可学习的压缩 KV cache** 直接存成参数：与 prefix-tuning 用 MLP 重参数化生成前缀不同，Cartridge 的前缀 KV 就是一张可训练的 embedding 表，按 prefix 语义直接注入注意力；论文推荐冻结第一个 token 作为 **attention sink** 保训练稳定。

官方 PEFT 实现的确定性账（锚点全部可离线重算）：

```
# 1) 从 legacy KV cache 打包 prompt embeddings（utils.py）
packed[:, 2*l]     = k_l[0, :, :NVT, :].transpose(0, 1)   # (NVT, H, D)
packed[:, 2*l + 1] = v_l[0, :, :NVT, :].transpose(0, 1)
pe = packed.reshape(NVT, L*2*H*D)                          # 每行 = L*2 段 K/V 交错

# 2) CartridgeEncoder（model.py）：frozen 段 + trainable 段
frozen_embedding   ~ N(0, 1)   # requires_grad=False，RNG 先抽
trainable_embedding ~ N(0, 1)  # requires_grad=True，RNG 后抽
weight = cat([frozen, trainable], dim=0)                   # (NVT, L*2*token_dim)

# 3) load_prompt_embeddings：按 num_frozen_tokens 分段拷贝
# 4) forward：忽略 token ids，weight.unsqueeze(0).expand(batch, -1, -1)
# 5) compose：cat(prompt_embeddings, dim=0)，frozen 数取第一个 adapter
```

三个关键口径：

1. **交错布局**：扁平化后每行的列下标 \(2l\cdot HD + h\cdot D + d\) 对应第 \(l\) 层第 \(h\) 头第 \(t\) 个虚拟 token 的 **K** 分量，\(2l+1\) 对应 **V**——K/V 层级交错、头在前、head_dim 最后（C 序）。
2. **frozen 在前**：冻结段是表格**开头**的 `num_frozen_tokens` 行（attention sink），`requires_grad=False`；`reset_parameters` 按官方顺序**先 frozen 后 trainable** 用 `normal_`（std=1）初始化——同一 RNG 流上顺序不可换。
3. **compose 的 frozen 数取第一个**：多个 adapter 拼接时 `num_frozen_tokens` 只取第一个 adapter 的配置（对应单个 attention-sink 前缀），兼容性要求 `peft_type="CARTRIDGE"` 且 task_type / token_dim / num_layers / num_attention_heads / num_transformer_submodules 五属性一致。

## 任务

实现打包、编码器与组合三组入口：

- `prompt_embeddings_from_past_key_values(past_key_values, *, num_virtual_tokens)`：入参是 per-layer `(k, v)` 元组列表（每个 `[1, H, S, D]`）。校验（全部抛 `ValueError`）：空缓存、rank≠4、batch≠1、`seq_len < NVT`、逐层头数/头维不一致、逐层缓存长度不足。返回 `(NVT, L*2*H*D)`。
- `CartridgeEncoder(num_virtual_tokens, num_layers, token_dim, num_frozen_tokens=1, dtype=torch.float32)`：校验 `0 <= num_frozen_tokens <= num_virtual_tokens`（越界抛 `ValueError`）；`weight` 属性（frozen 在前拼接；无冻结段时就是 trainable 参数本身）；`load_prompt_embeddings`（行数/维度不符抛 `ValueError`，按 frozen 数分段拷贝）；`forward(prefix_tokens)`（忽略 ids，扩 batch）。
- `compose_cartridge_adapters(adapters)`：入参为 `{"prompt_embeddings": 张量, "config": dict}` 列表。少于 2 个、`peft_type` 非 `"CARTRIDGE"`、兼容属性不一致（抛 `ValueError`）；返回 `(拼接后的 embeddings, 第一个 adapter 的 num_frozen_tokens)`。

## 示例

\(L=3, H=4, D=5, S=10, NVT=6\)：

```
prompt_embeddings_from_past_key_values(cache, num_virtual_tokens=6).shape -> (6, 120)
pe[3, 2*0*4*5 + 1*5 + 4] == cache[0][0][0, 1, 3, 4]        # 层0 K、头1、token3、维4
enc = CartridgeEncoder(6, 3, 20, num_frozen_tokens=1)       # hidden = 3*2*20 = 120
enc.weight.shape -> (6, 120)；enc.forward(空 ids (8,6)).shape -> (8, 6, 120)
compose([pa(4,120), pb(4,120)]) -> ((8,120), 1)
```

## 约束

- 打包是纯布局重排：不修改数值、只做切片/转置/拼接（`contiguous` 与否不影响结果）。
- 打包取缓存**前** `num_virtual_tokens` 个 token（截断，不从尾部取）。
- encoder 的 `dtype` 参数只影响存储精度；RNG 锚定用 float32 官方同路径（`torch.empty(...).normal_()`）。
- compose 的拼接顺序 = adapters 列表顺序；frozen 数即使后续 adapter 配置不同也以第一个为准。
