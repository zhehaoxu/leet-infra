# leet-infra 待办清单

> 按建议实施顺序编号；⭐ = 收益/成本比最高，优先做。
> **本清单只保留待办**：已完成项做完即分批提交，由 `git log` 追溯，不再在此归档。
> 当前题库状态：operators 86 + 框架 6 + perf-systems 9 + distributed 21 + llm-models 7 + optimization 17 + data-tokenization 8 + quantization 13 + sampling 12 + moe 7 + inference 7 + kvcache 14 + training-objectives 10 + speculative-decoding 6 + peft 6 + diffusion 6 + sparsity 6 + multimodal 6 + retrieval 6 ＝ **253 题**，全部含 notes.md 详解、示例值实跑 dump 对拍；判题环境 torch+numpy（+ llm-models 合集 oracle 用离线 `transformers==5.20.2`）、强制 CPU、无 GPU/torchvision。全量后端 `-n auto` 785 passed（含题库完整性护栏 test_problem_integrity）+ 前端 36 vitest + build 全绿。

## 题库与内容

> 选题标准：面试高频 + 判题确定性好 + 不与现有重复。AI-infra 题均按「算法/数据流/显存账可在 CPU 模拟」为前提设计；真 GPU kernel 归 item 7。

### operators 零散承接（线代/损失/归一化缺口）

- [ ] 38. ⭐ 补 einsum / matmul·bmm / vector_norm·matrix_norm / logsumexp / bce_with_logits / kl_div / diff / repeat_interleave——均可直接对拍原生 torch，确定性最强，先做这批垫底座
- [ ] 46. operators 缺口：loss 前向（focal/dice/kl/jsd/kd/ctc，order235–240）已由 #62 落地、group_norm·instance_norm 前向旧题已有。仍可补 **label_smoothing_loss**（前向）；各类 loss / 归一化 / 注意力的 **backward（局部梯度）** 统一归下面的 #65 autograd 新合集，以明确「前向 vs 反向」分工、避免重复

### AI-infra 新合集候选（每个 = collections.json 新增一个 collection）

- [ ] 45. ⭐ 新合集 `numerics`（数值稳定性 / 混合精度显式成题）：softmax·logsumexp 减最大值、Welford/两-pass 方差、NaN·Inf 传播与归约；再加 **FP8 E4M3/E5M2 格式转换与 block 缩放**、**动态 loss scaling**（overflow 计数 + scale 升降，回源 NVIDIA/apex）、**bf16·fp16 舍入误差**、**Kahan/成对求和**、**随机舍入（stochastic rounding）**、**数值 vs 解析梯度检查**——纯数值确定、CPU 可对拍。新 domain tag→族「数值稳定」需登记。编号从 **249** 起顺延
- [ ] 51. ⭐ `llm-models` 合集续批（复用 HF-eager-oracle 逐组件对拍范式，order 从 **182** 起顺延）：**DeepSeek-V3 MLA（低秩潜在压缩 + 解耦 RoPE，order181）已落地**；Qwen2 骨干 6 题（order108–113）已覆盖、勿重复。经核对，#51 原列候选多与 operator 合集撞车——**滑窗注意力(order48) / Gemma softcap(order49) / QK-Norm(order37) 均已作 operator 题**、Llama·Mistral 去 qkv bias 仅组件微差不单列。仍可做的新机制：① DeepSeek **noaux_tc MoE 路由**（sigmoid + 分组 top-k + e_score_correction_bias，异于现有 softmax `moe-topk-gating` order137，tag `moe`）② Llama/Mistral **RoPE 长度外推 scaling**（位置插值 / linear·dynamic NTK，`llm-rope-apply` 未含；与 #62 attention 子族的 YaRN/NTK 协调防撞）③ Mistral/DeepSeek **整模型装配 capstone**（端到端对拍 HF XxxModel、含 head_dim 推导）④ **Gemma (1+w) RMSNorm**（区别于标准 RMSNorm 的增益偏移技巧）。暂缓需真实前向/训练或重复机制
- [ ] 52. `kvcache` 合集第三批候选（kvpress@main 尚未入题的算法）：CriticalKV / Finch / Q-FILTER / SimLayerKV / Random（需外部给定噪声保确定性）；以及 per_layer_composed 等 Wrapper 类。已覆盖 14 题（含第二批 ChunkPress/ChunkKV/KeyRerotation/DuoAttention，order165–168），勿重复；注：LengthPress/L2Press/Vee/Head/Block/Query/Recurrent 系旧版或衍生仓库方法，当前 kvpress@main 已无对应文件

### quantization 合集下一批候选（LightCompress 已支持、尚未入题）

> 以下候选仍满足「CPU 可对拍 + 确定性 + 原创中文题面 + 独立纯 Python oracle」选题标准，续做时 order 从 **182** 起顺延（与 45 numerics 共用后续编号池，按实际落地先后取号）。已覆盖 SmoothQuant/AWQ/LLM.int8/QuaRot/GPTQ/SpQR 6 题（order169–174），勿重复。

- [ ] 53. quantization 第二批候选算法：① **NF4/QLoRA 正态分位数量化**（information-theoretically optimal，按半正态 CDF 求分位格点 + dequant scale，格点非均匀、公式干净）② **KIVI KV cache 非对称量化**（Key 逐通道 / Value 逐 token 的非对称 zero-point+absmax，与 per-channel-quantize 呼应但更贴 LLM 缓存）③ **HQQ half-quadratic 压缩**（Wiener 滤波闭式求最优 scale，与 RTN/GPTQ 目标不同）④ **SqueezeLLM 非均匀敏感量化**（Fisher 敏感度加权 + 类 k-means 非均匀格点）⑤ **双重量化（double quantization）**（对量化 scale 再量化，QLoRA 落地技巧，纯数值可精确对拍）。暂缓：OmniQuant/ learnable 类需反向训练、不利 CPU 确定性

### speculative-decoding 合集下一批候选（SpeculativeDecodingPapers 带源码链接、尚未入题）

> `speculative-decoding` 合集现有 6 题（order175–180，tag `speculative`→族「推测解码」）。注：Leviathan/Chen 的**线性 draft 修改版拒绝采样**已由旧题 `inf-spec-decoding-rejection`（order155, tag inference）覆盖，勿重复入题。续做 order 从 **182** 起顺延。

- [ ] 54. speculative-decoding 第二批候选（均带源码链接、CPU 可对拍；与现有 155/176/177/179/180 各有分工，勿重复）：① **Sequoia 有界深度最优树 DP**（2402.12374 Algorithm 4，Infini-AI-Lab/Sequoia：在 177 的 T[n,b] 上叠加深度上限 D、并把接受向量扩展为按深度索引的 2D 接受率矩阵以兼容 self-speculation；纯 DP，可暴力枚举有界深度树作 oracle）② **EAGLE-2 动态草稿树重排**（2406.16858 SafeAILab/EAGLE：draft 树草拟后第二遍前向估各节点接受率、据此重算累积接受分数并动态剪枝/扩展到验证预算，选实际送入 target 的节点集；给定 toy logits/接受率可确定对拍）③ **SpecInfer 通用 token-tree 并行验证**（2305.09781 flexflow-ai/SpecInfer：tree-attention 一次性并行验证整棵草稿树、沿接受前缀逐层传播、拒绝处回退取最长存活路径；区别于 176 Sequoia Algo2 的单位置残差收缩）④ **LayerSkip 自投机 early-exit 解码**（2404.16710 facebookresearch/LayerSkip：以逐层 exit/skip 阈值决定用浅层出 draft、末层做 speculative verification 的自投机；给定逐层 logits 可确定性模拟）⑤ **Multi-Candidate Speculative Decoding**（2401.06706：每个位置从 draft 采多个候选、成批组织后按目标分布验证，是 155 单序列拒绝采样向多候选的扩展）。暂缓：EAGLE 特征自回归、Medusa 头训练等需真实前向/训练（不利 CPU 确定性）；vLLM N-gram proposer 与 179/180 检索式草稿高度重叠

### AI-infra 待选新方向（2026-10 题库盘点，按需择做）

> 现状：训练算子/分布式(含 ZeRO·重算·序列并行)/推理 Serving/KV 压缩/量化/采样/推测解码/对齐(PPO·DPO·RM·GAE·GRPO)/PEFT/扩散/稀疏/多模态/检索/优化器 已密集。**核对确认的空白或明显缺口**见 #45/#51/#52/#53/#54 与下方 #63–69（含 #68 推理引擎 serving 内核续批、#69 `agentic-systems` agent 新合集）；注意 #62 续批剩 attention/长上下文与 inference serving 两子族。均满足「CPU 可对拍 + 确定性 + 不与现有重复」，oracle 走原生 torch 对拍 / 独立纯 Python 重算 /（涉模型时）HF transformers。新合集落地时 order 从 **249** 起（各候选共用编号池，按实际落地先后取号）；**每开一个新合集需同步** problemFilter.ts 新 domain tag→TAG_FAMILY→GROUP_ORDER + collections.json 条目 + 护栏测试。

- [ ] 62. 现有合集续批（补题、非新集）——**loss 家族 / SFT 数据 / distributed 三子族已落地**（order235–248：operators 追加 6、data-tokenization 追加 4、distributed 追加 4，详见 git log）。**仍待办两子族**：① **attention/长上下文**——FlashAttention **backward**（前向已有 `flash-attention-tile`，反向用 autograd 对拍、归 #65 协同）、**YaRN/NTK RoPE 长度外推**（与 #51② 协调防撞）、Ring/Blockwise 分布式注意力（做真实环形 KV 交换逐步模拟，区别于 #distributed `distributed-sequence-parallel` 的显存-通信账）、Gemma2 **sliding+global 层交替掩码**；② **inference serving**——**Radix-tree 前缀缓存**（token 字典树 LRU 复用，异于现有 `inf-prefix-cache-block-hit` 的块哈希命中）、**priority/anti-starvation 调度**（现有 `inf-continuous-batching-scheduler` 之外）、**swap vs recompute 抢占决策**、**disaggregated prefill/decode** 显存-带宽分配账。**续做 order 从 249 起**
- [ ] 63. ⭐ 新合集 `evaluation`（LLM 评测 / 指标基建，全空白）——训后评估与数据筛选的确定性核心计算，纯数值、CPU float32 可对拍。候选：① **perplexity from logits**（next-token NLL：右移对齐、`-logsoftmax(logits)` 取 label 位、掩 pad 求均再 `exp`，回源 Mikolov / HF `evaluate`）② **token-level accuracy**（`ignore_index` / prompt 位不计）③ **BLEU / ROUGE / F1 / exact-match**（n-gram precision + brevity penalty、ROUGE-N 召回与 ROUGE-L LCS，回源 Papineni/Lin）④ **pass@k**（Codex 2107.03374 组合式 1−C(n−c,k)/C(n,k)，整数无偏、确定性最强）⑤ **few-shot prompt 组装与截断**（示例拼接顺序 + tokenizer 计数后按预算裁剪）。每题独立纯 Python oracle + 退化不变量。新 domain tag→族「评测指标」需登记
- [ ] 64. ⭐ 新合集 `audio`（语音 / 音频前端 DSP，全空白）——Whisper/Hubert 等语音模型输入特征，纯确定性信号处理、CPU float32 可精确对拍。候选：① **预加重 + 分帧 + 加窗**（帧长/步长、Hann·Hamming 窗）② **STFT**（n_fft / center / 反射填充，回源标准定义）③ **Mel 滤波器组 + log-mel**（HTK vs Slaney 刻度、Whisper 80 维 log-mel）④ **MFCC**（DCT-II 降相关 + 可选 CMVN）⑤ **整数比重采样 / polyphase**。oracle 用 numpy FFT / 逐帧双循环重算。新 domain tag→族「语音前端」需登记
- [ ] 65. ⭐ 新合集 `autograd`（反向模式 AD 与局部梯度，仅 1 道入门题 `autograd-basics`，反传维度基本空白）——补齐「backward」，与 #46 残项合并。候选（一律用 `torch.autograd.grad` 对拍解析梯度、结构独立重算）：**softmax backward**、**cross_entropy / label_smoothing backward**、**layernorm backward**、**attention（含 causal mask）对 Q/K/V 的梯度**、**embedding 稀疏梯度**、**GRU/LSTM cell backward**。分工：本族只做「给定上游梯度求各输入/参数局部梯度」的反向，前向已在 operators；须确保 float64 CPU 对拍确定性
- [ ] 66. 数据管线续批（归 `data-tokenization` 或新 `data-pipeline`，多为空白）——训练数据构造与调度的确定性算法。候选：① **数据集混采 / interleaving**（权重采样生成 epoch 索引序列、温度混采 DoReMi 式）② **length bucketing / padding-aware batching**（按长度分桶降 padding 浪费，与现有 `sequence-pack-pad` 互补）③ **去重：exact hash + MinHash + LSH 近重复**（shingled n-gram 签名分桶，回源 Broader/Charikar）④ **WordPiece 贪心最长匹配切分**（BERT 式、`##` 续接，异于已做的 BPE/Unigram，确定性强）⑤ **分布式数据分片到 rank·worker**（确定性 shuffle + shard 索引，与 `distributed` 呼应）
- [ ] 67. perf-systems 续批（内存分配 / 复用账本，除 KV paging 外基本空白）——候选：① **PyTorch caching allocator 块管理**（空闲块 best-fit、按 size 分裂 / 合并相邻 free block、段扩展与碎片率账；纯数据结构可对拍，异于 `inf-paged-kv-block-alloc` 的 KV 块表索引）② **激活/梯度显存复用规划**（给定活跃区间做 offset 分配、区间图着色求峰值）③ **offload 逐 step 调度**（HBM↔DRAM↔SSD 预取窗口与带宽重叠账，与 `distributed-zero-offload/infinity` 呼应但做时间线调度）
- [ ] 68. `inference` 合集续批（serving 引擎内核，除现有 7 题外的确定缺口；order 从 **249** 起）：① ⭐ **受限 / grammar 约束解码 token 掩码**（给定 JSON schema / 正则→DFA，按已生成前缀输出下一合法 token 布尔掩码、括号·枚举·类型栈匹配；回源 Outlines / xgrammar / llama.cpp GBNF，oracle 独立自动机走 or 暴力枚举合法字符集——确定性最强）② **抢占决策 swap vs recompute vs abort**（显存不足按剩余 token/优先级/SLA 选牺牲集并算 swap 字节 vs 重算代价，回源 vLLM；与 #62 inference-serving 子族合并推进）③ **Radix-tree 前缀缓存**（token 字典树插入/最长前缀命中/引用计数/LRU 驱逐可复用块数，回源 SGLang RadixAttention，**异于现有 `inf-prefix-cache-block-hit` 的块哈希命中**）④ **disaggregated prefill/decode 放置账**（P/D 分离 KV 跨节点传输量 + 副本放置带宽-显存账，回源 Mooncake/DistServe，环系数口径同 `distributed-sequence-parallel`）⑤ **每步 token 预算背包**（一 forward 名额分给混合 prefill/decode 请求最大化吞吐/公平，回源 Sarathi-Serve，0/1 或分数背包 DP + 暴力子集 oracle）⑥ **请求路由 + 负载均衡**（least-loaded / 一致性哈希打多副本、输出落点分布与倾斜度，并列取小副本号保确定）⑦ **TTFT/TPOT 排队账**（给定到达率/服务率/批上限算首 token 延迟与吞吐，纯公式，与 perf-systems `roofline`/`inference-throughput` 分工=排队侧）
- [ ] 69. ⭐ 新合集 `agentic-systems`（agent infra，全空白）——**红线：把「LLM 输出了什么」当显式入参，只考系统对它的编排/解析/调度/预算/路由决策**（凡需模型生成/判断的一律不出）。候选：① ⭐ **ReAct 循环状态机**（Thought/Action/Action Input/Observation 文本流解析→执行→判终止(无 Action/max_steps/答案收敛)、输出最终观测序列与步数，回源 2210.03629）② ⭐ **函数调用 schema 匹配**（tool_call 参数对 JSON Schema 校验：必填/类型强制/枚举/默认填充，返回规范化参数或拒因，回源 OpenAI function-calling）③ ⭐ **任务 DAG 拓扑执行 + 关键路径**（子任务依赖→拓扑序、并行波次 level、最大并行度、关键路径长，回源 BabyAGI/HuggingGPT，oracle Kahn vs 暴力层序）④ **上下文预算压缩 compaction**（system/history/tools/scratchpad 按优先级+token 数在窗口 W 内截断/占位选保留段，确定性优先级打包）⑤ **generative-agent 记忆召回打分**（score=α·recency 指数衰减+β·importance+γ·relevance 归一 top-k、并列按插入序，回源 2304.03442，**与 `retrieval` 明确分工=打分融合非 ANN**）⑥ **LLM 模型路由 / cascade**（按难度特征分大小模型或先小后大置信度<τ 升级，算成本-准确率-延迟，回源 RouteLLM/FrugalGPT）⑦ **多 agent handoff 图**（转发有向图防环访问集、超步数/预算回退求移交路径，回源 AutoGen/Swarm）⑧ **ToT / best-first 规划搜索**（toy 状态空间 BFS/DFS/A*/beam 输出访问序与解路径，回源 2305.10601，**注意与 speculative 草稿树区分**）。新 domain tag→族「智能体系统」需登记 problemFilter.ts + collections.json + 护栏；建议首批先做 ①②③ 三件套（确定、热、零撞车）

> 暂缓（不适合当前 CPU-only）：真 GPU kernel（归 #7）、需真实大模型前向+训练梯度的整模型反向（除非改走 torch.autograd 对拍原生）。

## 性能与新能力

- [ ] 16. 后端评测结果的用例级历史（最近 N 次提交记录、失败 diff 回看；涉及存储结构，先想清楚再做）
- [ ] 29. 进度统计小视图（可选）：利用 solved_at 时间戳做近 30 天做题热历/每日计数，纯激励向

## 大方向（需决策后再排期）

- [ ] 7. 第二阶段 Triton/CUDA 真实实现（variants 框架与下拉 UI 已就绪，现有题均只有 pytorch 可用 + triton/cuda 置灰）
  - 卡点：官方 Triton 无 macOS wheel，需先二选一
  - 方案 A（推荐）：OrbStack + linux/arm64 容器跑官方 triton CPU backend
  - 方案 B：triton 源码编译到 macOS（有历史失败记录，耗时不可控）
  - 待做：为算子题补 `variants/triton/`、`variants/cuda/` 三件套并把 meta.json 的 available 改 true，评测链路容器化；建议从 relu/sigmoid/softmax 等逐元素题起步再攻 attn/conv
