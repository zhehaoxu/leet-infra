# 详解：Cartridge 直接参数化的 KV 前缀适配器

## 出处

Cartridges（arXiv:2506.06266，Stanford Hazy Research，博客 2025-06-08）：用**合成自学语料**训练一段压缩 KV cache（cartridge），推理时作为前缀注入，让模型"带着记忆"但不需要把整个语料放进上下文。PEFT PR #2953（2026-01）把其中**适配器侧**的账落成 `tuners/cartridge/`（config.py + model.py + utils.py，共 455 行）——base 实现等价 prefix tuning，但**直接存 KV 而非 MLP 重参数化**，且新增 frozen 首 token（attention sink）与 KV 初始化工具。

## 官方实现的逐行翻译

**打包（`utils.py::prompt_embeddings_from_past_key_values`）**：

```python
legacy = _to_legacy_past_key_values(past_key_values)   # 判题域已剔除，入参即 [(k,v),...]
if len(legacy) == 0: raise ValueError(...)
num_layers = len(legacy); key0, _ = legacy[0]
# rank 4 / batch 1 / seq_len >= nvt 校验（均 ValueError）
packed = torch.empty(num_virtual_tokens, num_layers*2, num_heads, head_dim, ...)
for layer_idx, (k, v) in enumerate(legacy):
    # 逐层头数 k/v.shape[:2] == (1, num_heads)、长度 k/v.shape[2] >= nvt、
    # 头维 k/v.shape[3] == head_dim 校验
    packed[:, 2*layer_idx]     = k[0, :, :nvt, :].transpose(0, 1).contiguous()
    packed[:, 2*layer_idx + 1] = v[0, :, :nvt, :].transpose(0, 1).contiguous()
return packed.reshape(num_virtual_tokens, -1)
```

元素账：`pe[t, (2l)*H*D + h*D + d] == k_l[0, h, t, d]`、`pe[t, (2l+1)*H*D + h*D + d] == v_l[0, h, t, d]`。**K/V 层级交错 + 头在前 + 截前 NVT 个 token** 是三处最容易写反的位置。

**编码器（`model.py::CartridgeEncoder`）**：

```python
num_frozen_tokens = int(config.num_frozen_tokens)
if num_frozen_tokens < 0 or num_frozen_tokens > num_virtual_tokens: raise ValueError(...)
self.num_trainable_tokens = num_virtual_tokens - num_frozen_tokens
if self.num_frozen_tokens:
    frozen = torch.empty(nft, hidden)
    self.frozen_embedding = torch.nn.Parameter(frozen, requires_grad=False)
trainable = torch.empty(ntr, hidden)
self.trainable_embedding = torch.nn.Parameter(trainable, requires_grad=not config.inference_mode)
self.reset_parameters()          # Embedding 式：frozen 先 normal_、trainable 后 normal_（std=1）

weight: frozen_embedding is None ? trainable_embedding : cat([frozen, trainable], dim=0)
load_prompt_embeddings: 行数与 (nft+ntr, hidden) 不符 raise；[0:nft] → frozen、[nft:] → trainable
forward(prefix_tokens): 忽略 ids；weight.unsqueeze(0).expand(batch, -1, -1)
```

config 层（`CartridgeConfig.__post_init__`）还有一道 `num_frozen_tokens < 0` / `> num_virtual_tokens` 校验（`num_frozen_tokens` 默认 1，论文推荐冻首个 token 当 attention sink）——判题域并入 encoder 构造。

**组合（`utils.py::compose_cartridge_adapters`）**：`len < 2` raise；每个 config `peft_type == CARTRIDGE`；五兼容属性（task_type / token_dim / num_layers / num_attention_heads / num_transformer_submodules）与第一个一致，`num_frozen_tokens` **只取第一个**（注释原话："matches a single attention-sink prefix"）；`torch.cat(prompt_embeddings, dim=0)`。

**初始化工具（`initialize_kv_prefix_from_past_key_values` / `from_text`）**：对 CARTRIDGE 与 PREFIX_TUNING 都可用（`prefix_projection=True` 拒绝）；`from_text` 要跑真实模型 prefill——**触真实前向红线不入题**；`load_prompt_embeddings` 是它们的落点（已入题）。

## 判题域剔除与红线

- `_to_legacy_past_key_values`（transformers Cache 对象兼容分支）：`.venv-eval` 无 transformers，入参固定 legacy list。
- `initialize_kv_prefix_from_text`（tokenizer + 真实模型 prefill）与 `model.disable_adapter()` 上下文：真实前向红线。
- safetensors/torch.save 序列化（save_file/load_peft_weights）与磁盘路径账：IO 红线。
- `PeftConfig.from_pretrained` / config 序列化：判题域用 dict 直传。
- GQA 的 `token_dim` 调整（`_prepare_prompt_learning_config`）：全局 PEFT 机制，判题域 token_dim 显式入参。

## 与已有题的划界

- `peft-prefix-tuning`（765 期）：那里考**注意力前向**（前缀在前的 KV 拼接）+ PrefixEncoder 两形态参数量账（MLP 重参数化）；本题考 **KV cache → 扁平参数的打包布局** + frozen/trainable 分段 + compose，无 MLP、无注意力前向。
- kvcache 合集（snapkv/h2o/quest 等）：全是**驱逐/量化/合并**账；本题是**训练侧参数布局**账，不重叠。
- `kv-streaming-llm-sink`：sink 是**驱逐保护位**；本题 frozen 首 token 是**冻结参数段**，机制不同。

## 变异测试（35/36，1 等价）

- 打包：不转置 / K-V 互换 / 列槽覆盖 / transpose(0,2) / batch 反转 / rank 反转 / 头数校验放松 / 头维索引错 / 长度校验放松 / 空校验 / head 来源错——全 CAUGHT。
- 编码器：frozen 可训 / RNG 顺序 / std / cat 维度 / 丢 frozen / 拼接反转 / load 段错 / load frozen 段错 / ndim 校验 / 行数校验 / expand 形状 / unsqueeze 轴 / trainable 数 / hidden 公式 / frozen 分支恒真 / frozen 上界 / 负数校验——全 CAUGHT。
- 组合：len 门 / peft_type / cat 维度 / 拼接反转 / frozen 取尾 / 属性校验反转 / 属性表截断——全 CAUGHT。
- **m-pk-seq 真等价剔除**：前置 `seq_len < nvt` 校验（第一层）与循环内逐层 `k/v.shape[2] < nvt` 校验对判题域完全冗余——任何触发前置的输入都会先被循环内校验以 ValueError 拒绝（测试只断言 ValueError 类型），故前置校验的松紧不可观测。

## RNG 锚定

`torch.nn.init.normal_` 与 `torch.randn` **不是同一条 RNG 流**（804 期 uniform_ 教训同款）；`torch.manual_seed(42)` 后按官方顺序 `empty(nft, h).normal_()` → `empty(ntr, h).normal_()` 可逐位复现。float32 与 float64 的流也不同，锚点固定 float32。
