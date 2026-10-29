# 详解：Gemma2 整模型装配

## 出处

- Gemma Team, *Gemma 2: Improving Open Language Models at a Practical Size*, Google DeepMind, 2024, arXiv **2408.00118**。整模型层面的三处标志性设计：偶数层/奇数层交替的局部-全局注意力、注意力 logits 与末端 logits 的 tanh soft-capping、以及用于蒸馏的训练流程。本题只取前两项加上装配规则。
- 可执行规范取 `transformers==5.20.2` 的 `transformers/models/gemma2/modeling_gemma2.py` 与 `configuration_gemma2.py`：`Gemma2Config` 缺省 `layer_types[i] = "sliding_attention" if (i+1) 为奇数 else "full_attention"`、`tie_word_embeddings=True`、`attn_logit_softcapping=50.0`、`final_logit_softcapping=30.0`、`query_pre_attn_scalar=256`、`head_dim=256`；`Gemma2Model` 用 `embed_scale=hidden_size**0.5` 缩放嵌入，注意力缩放取 `query_pre_attn_scalar**-0.5`，softcap 在加掩码之前。
- 与库内其他题的分工：`llm-gemma2-layer`（order 525）只做**单层**（输入 cos/sin，返回该层输出与注意力权重），不涉及嵌入缩放、层型序列、词表头与末端 softcap；`llm-qwen2-model`（order 113）是 Qwen2 的整模型装配，但 `head_dim` 由调用方直接传入、`lm_head` 显式不共享、没有 softcap 与层型交替，也不接受 config 推导。本题的独有内容是**从 config 推导维度/层型 + 词表头绑定**这条装配链。
- 与 `llm-yarn-rope-scaling`（order 524）的分工：那题只产 `inv_freq` / 缩放因子 / 整张 cos-sin，本题用最朴素的 RoPE（`rope_theta`），不涉及外推。

## 核心与易错点

1. **`head_dim` 要推导，但不能"凑"**。本题缺省 `hidden // num_attention_heads`；真实 Gemma2Config 里 `head_dim` 是显式字段（默认 256，9B 模型 `hidden/heads = 224 ≠ 256`），所以业务代码里两者都可能出现。推导值必须与 `q_weight` 行数吻合，否则报错——HF 的 `getattr(config, "head_dim", hidden // heads)` 会静默取到错误维度，这里不允许。`head_dim` 必须为偶数，否则 RoPE 的 `rotate_half` 没有配对维度。
2. **KV 头数不能整除时必须报错**。HF 用 `num_attention_heads // num_key_value_heads` 做整数除法，3 头配 2 组会被悄悄截成 1 组；本题直接拒绝。
3. **层型交替从 sliding 开始**。`layer_types[i] = sliding if (i+1) % 2 else full`，第 0 层就是滑窗层；写成从 full 开始会让所有层用错掩码。
4. **滑窗含当前 token**。条件是 \(\text{kv} > \text{q} - \text{window}\)，窗口 1 时每个位置只看自己；写成 \(\ge\) 会多放一个位置进去，`window_off_by_one` 变异被 5 个检查同时抓住。
5. **softcap 在加掩码之前**。HF eager 路径是：`logits * scale` → `cap*tanh(logits/cap)` → `logits + mask`（mask 处为 `finfo.min`）。若先加掩码再 softcap，被屏蔽位置的 \(-\infty\) 会变成 \(-cap\)，softmax 后拿到非零权重，整层结果全错。
6. **注意力缩放不是 \(1/\sqrt{h_d}\)**，而是 `query_pre_attn_scalar**-0.5`。Gemma2 用这个字段把缩放与 head 维度解耦；照抄 Llama 的 `head_dim**-0.5` 会得到不同的权重。
7. **双 post-norm 的加法顺序**：先对**子层输出**做 RMSNorm，再加残差（`x = residual + RMSNorm(sublayer(x))`），不是先 norm 再相加。pre-norm 的写法在这里是错的。
8. **norm gain 是 \(1+w\)**。Gemma2 的 norm 权重零中心化，`w=0` 表示恒等；直接用 \(w\) 会让 gain 全变 0。
9. **嵌入缩放只作用在入口**：`X = embed[ids] * sqrt(hidden)`；末端 RMSNorm 之后不再乘。少了这一项所有 logits 量级都不对。
10. **词表头绑定**：tie 时用 `embed_weight` 当词表头（同一个矩阵被读两次），并把独立的 `lm_head_weight` 判为非法；不 tie 时缺词表头必须报错。绑定语义与"传了一个恰好等于 embed 的独立矩阵"在数值上等价，但契约上必须二选一。
11. **末端 softcap 在词表头之后**：先用隐藏态算 logits，再 `cap*tanh(logits/cap)`；把隐藏态先压一遍再投影是错的。
12. **GeGLU 用 tanh 近似**：`gelu(gate, approximate="tanh") * up`，与精确 erf 版本有可见差距。
13. **精度口径**：内部 float64、返回 float32；与 HF（全程 float32）对拍时容差取 2e-5。

## 对拍口径

- `_oracle` 用 numpy 逐 batch/头/位置三重循环重写整模型：手写 RoPE 旋转、逐位置枚举合法 kv 下标做 softmax、手写 tanh-GELU，和 torch 实现没有共享代码；`config` 的推导规则也独立重写一遍。
- 10 个 `check_*`：手算小样例、缺省推导等价性（省略全部可推导键后逐位相同）、词表头绑定、滑窗收窄与"窗口 ≥ S 退化为因果全注意力"、因果前缀稳定（更长序列不改写前缀 logits）、两级 softcap、GQA 分组映射、随机对拍（4 组配置 + 独立词表头 + float32 输入）、对拍 HF 完整模型（3 组显式配置 + 1 组全缺省推导 + 1 组不绑定）、53 种非法输入。
- 与 HF 的对拍直接调用 `Gemma2ForCausalLM`（eager）**且不传 attention_mask**，让 HF 自己用 `create_causal_mask` / `create_sliding_window_causal_mask` 建掩码——这样层型交替与滑窗边界是拿 HF 的掩码工具验证的，而不是自己造一遍。
- 变异测试 18 项全部被抓住：略去嵌入缩放、k 不做 RoPE、GQA 交错复制、softcap 放到掩码之后、滑窗差一、去掉因果掩码、post-norm 作用在残差上、softcap 放在词表头之前、忽略绑定、无视 layer_types、gain 用 \(w\)、换精确 GELU、去掉末端 norm、忽略显式 head_dim、去掉末端 softcap、去掉注意力缩放、去掉 FF 输出 norm。
- 非法输入只捕获 `ValueError`：config 非字典/未知键/缺 `num_attention_heads`、`hidden_size` 与 `vocab_size` 不匹配、头数为 0/bool/float、`head_dim` 奇数或 0、KV 头数不整除、层数不匹配、`layer_types` 长度/取值/类型错误、窗口为 0/bool/float、`query_pre_attn_scalar` 为 0/NaN/bool、softcap 为 0/负/字符串/`inf`、`rms_norm_eps` 或 `rope_theta` 非正、`tie` 非 bool、绑定状态与词表头不符、各权重形状或 dtype 错误、输入 id 非张量/浮点/一维/空/越界。

## 复杂度

- 每层：q/k/v 投影 \(O(B S h (H_q+2H_{kv}) d)\)、注意力 \(O(B H_q S^2 d)\)、o_proj \(O(B S h H_q d)\)、GeGLU 三段 \(O(2 B S h I)\)。
- 总时间 \(O\!\left(L\cdot\left(B S h (H_q+2H_{kv})d + B H_q S^2 d + 2 B S h I\right)\right)\)；额外空间主要是注意力权重，全注意力层为 \(O(B H_q S^2)\)，滑窗层可压到 \(O(B H_q S \cdot \text{window})\)。
