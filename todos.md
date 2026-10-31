# leet-infra 待办清单

> 按建议实施顺序编号；⭐ = 收益/成本比最高，优先做。
> **本清单只保留待办**：已完成项做完即分批提交，由 `git log` 追溯，不再在此归档；各题的机制细节与剥除口径以该题 `notes.md` 为准，本清单不复述。
> 当前题库（**27 合集 / 624 登记题**；磁盘 625 个题目录，除并行会话在飞的 `fw-autograd-lineage-ledger`（order 631）外全部登记且一一对齐；`collections.json` 的登记项之间无重复、无悬空；参考解 self-AC / integrity / bilingual 复跑结论不再留档，`git log` 可查）**：pytorch 入门 19 + operators 108 + perf-systems 19 + distributed 32 + llm-models 15 + optimization 28 + data-tokenization 22 + quantization 43 + sampling 19 + moe 18 + inference 21 + kvcache 49 + training-objectives 21 + speculative-decoding 16 + peft 36 + diffusion 48 + sparsity 18 + multimodal 20 + audio 5 + retrieval 14 + evaluation 8 + rl-systems 6 + agentic-systems 4 + vision-detection 8 + autograd 16 + numerics 8 + **context-compression 3（2026-10-31 新开，`context` 标签已在 problemFilter.ts / i18n.ts 登记）**。旧空号 405–419 / 431 / 440 / 449 / 466–469 **不回收**，以免与已删题的历史编号混淆；全库 order 唯一有 `test_problem_integrity` 护栏。order 编号池已占用至 **645**，**下一批新题从 648 起**（动笔前重新扫描全库避让并行会话）。
> 判题环境：torch + numpy（llm-models 合集 oracle 用离线 `transformers==5.20.2`），强制 CPU，无 GPU / torchvision；`.venv-eval` 无 diffusers、无 pytest（测试只能用裸 assert，oracle 用纯 torch/numpy 重写）。自测入口：`scripts/add_problem.py --selftest <slug> --mode submit|run`、`backend/tests/test_self_ac.py`。
> **出题通用纪律**（全库口径，下面各条不再重复）：上游依赖 scipy 的分位/聚类/边际码表（`norm.ppf`、`kmeans2`、TurboQuant 的 Beta 码表、字典训练）一律**离线算好作显式入参**，`.venv-eval` 不装 scipy；随机性来源（Bernoulli / Gumbel / 加权 `choice` / `svd_lowrank` / EMA）要么改成确定性 top 预算，要么直接剔除；涉 RoPE / attention 的题把 q、K、cos/sin、注意力矩阵全部作入参，不现跑模型。

## 题库与内容

> 选题标准：面试高频 + 判题确定性好 + 不与现有重复。AI-infra 题均按「算法/数据流/显存账可在 CPU 模拟」为前提设计；真 GPU kernel 归 item 7。

### 题库收尾（优先清空）

> 新合集落地时同步三件套：`frontend/src/utils/problemFilter.ts` 的 TAG_FAMILY + GROUP_ORDER（`familyOf` 要求 tags 含 `operator` 才归族）+ `problems/collections.json` 条目 + `test_problem_integrity` 护栏。

### AI-infra 待选新方向（2026-10 题库盘点，按需择做）

> 现状：训练算子/分布式(含 ZeRO·重算·序列并行)/推理 Serving/KV 压缩/量化/采样/推测解码/对齐(PPO·DPO·RM·GAE·GRPO)/PEFT/扩散/稀疏/多模态/检索/优化器 已密集。**缺口盘点见 #86 与 #102–105（#92/#93/#95/#97/#98 均已于 2026-10-31 收口）**。均满足「CPU 可对拍 + 确定性 + 不与现有重复」，oracle 走原生 torch 对拍 / 独立纯 Python 重算 /（涉模型时）HF transformers。新题共用 order 编号池：实施前须扫描当前全库取未占用号段（**当前从 648 起**），并在验收时复查并行出题冲突；**每开一个新合集需同步** problemFilter.ts 新 domain tag→TAG_FAMILY→GROUP_ORDER + collections.json 条目 + 护栏测试。

- [ ] 86. `multimodal` 合集续批（合集现 **20 题**；次批 5 题 order 533–537 与本批 644–645 均已落地，机制出处 Kirillov(2304.02643)/Yang(2403.06764)/Lin(2403.15388)/HF `FluxPosEmbed`，细节 `git log`）。**原清单三处口径经复核不成立、已按真实源码出题，后续引用勿再照抄旧说法**：SAM 没有 `Conv1d(1, embedding_dim)` 标签嵌入也不存在 `get_predefined_embeddings`（标签走 `nn.Embedding` 的 `point_embeddings`，`-1` 只出现在 `not_a_point_embed` 写回路径）；`utils/evaluation.py::mask_iou` 在上游不存在（掩码质量实际走 `amg.calculate_stability_score` 的 `thr±offset` 计数比与 `select_masks` 的 `iou_preds+(num_points-2.5)*[1000,0,0,0]`）；PruMerge 的出处是 2403.15388 而非 2406.17521；FLUX 侧没有 `FluxParamsRotaryEmbedding`/`_get_q_and_k_rope_freqs` 这对名字。**剩余候选仍需先回源再立项**：① FastV 的**逐层自适应压缩率**（需真实层间重构误差，慎防退化成系数表）；② SAM 的 light-HQP mask decoder 与 hypernet 权重生成（需真实权重，剔除）；③ Qwen2.5-VL / InternVL 的 visual token 打包与 M-RoPE 交错（与 `mm-mrope-position-ids`、`mm-image-token-expand` 查重后再定）；④ FLUX double/single stream 的 `vec` 拼接口径（纯接线，先与 `llm-*` block 题查重）。通道/空间 shuffle 及逆变换已有 `pixel-shuffle-space-depth`，不得再以「PatchMerger 互逆」重复立项；带权重的真实 PatchMerger 不保证可逆。**音频侧**（mel→两次 stride 卷积→audio token 数账）归 `audio` 合集，本合集只收非 DSP 结构题。

- [ ] 102. 新合集 **`linear-attention`**（线性注意力 / SSM 系，2026-10-31 大模型向盘点新增，全库零覆盖；落地按「题库收尾」三件套登记）。**候选（先逐条回源 + 查重，order 从 648 起）**：
  ① **Mamba selective scan**（2312.00752，official `state-spaces/mamba`）——S6 逐元素 Δ/B/C 状态递推账（ZOH 离散化后 Δ·B·C 全显式入参）+ 分块扫描的边界 carry 传递；oracle 纯 Python 逐步递推对拍 torch 复算。
  ② **DeltaNet / Gated DeltaNet**（2406.06484 / 2412.06464，official `fla-org/flash-linear-attention`）——delta rule 逐 token 状态更新 `S += β(k vᵀ − S k kᵀ)`（门控衰减系数显式入参）；WY/UT 分解不出。
  ③ **RWKV v6 channel-wise decay**（official `BlinkDL/RWKV-LM`）——data-dependent 衰减的逐通道状态递推账。
  ④ **Gated Linear Attention**（2312.06635，`fla-org`）——chunk 内并行 + chunk 间状态传递的尺寸/边界账。
  **红线**：真模型权重前向、CUDA kernel 分块细节不出（只考算术账）；与 `cumsum-scan`（前缀和）与 attention 系查重；只考 recurrence 与并行恒等式，不考训练收敛。

- [ ] 103. 新合集 **`interpretability`**（可解释性，2026-10-31 盘点新增，全库零覆盖；落地三件套同上）。**候选**：
  ① **SAE top-k encode/decode**（回源 `EleutherAI/sae` 或 `jbloomAus/SAELens` 二选一）——W_enc/b_enc/W_dec 显式入参，top-k 激活 + 重构误差 + 稀疏度账（JumpReLU 门槛确定性变体可附）。
  ② **activation steering**（steering vector，official `nrimsky/steering-vectors`）——注入后的投影/范数/方向余弦账，隐态全显式入参。
  ③ **tuned lens**（official `AlignmentResearch/tuned-lens`）——affine probe 显式入参的层间 logits 映射 + 交叉熵/KL 账。
  **红线**：需真实大模型激活的题不出，隐态/权重一律显式入参；logit lens 须简化为「最终 norm + unembed 矩阵入参」才可判。

- [ ] 104. 新合集 **`reasoning-systems`**（推理模型语义侧，2026-10-31 盘点新增；rl-systems 只管 infra（async rollout/replay/时间线），本合集管语义与账本；落地三件套同上）。**候选**：
  ① **self-consistency 多数投票聚合**（2203.11171，官方实现并入 lm-eval-harness 待核，先回源）——答案归一化 + 投票记账。
  ② **best-of-N + PRM 步级聚合**（official `openai/PRM800K`）——step 打分表的 last / max / min / product 四种聚合口径账。
  ③ **budget forcing**（s1，official `simplescaling/s1`）——thinking token 预算的截断 / 「Final Answer:」强制续写账。
  ④ **overthinking 指标**——token 效率、正确率-长度曲线的确定性账。
  **红线**：模型生成本身不出，候选/打分表/预算全显式入参；与 `to-gae`/`to-grpo-advantage`（训练侧 advantage）划界。

- [ ] 105. **大模型向现有合集空隙批**（2026-10-31 盘点，逐条 grep 已确认无覆盖；order 从 648 起）：
  `data-tokenization`：① **Gopher rules** 规则过滤（文档长度/符号比例/重复行/省略号比例；无官方码，回源 `huggingface/datatrove` 口径）；② **C4 过滤**（`allenai/c4` 的 badwords + 行去重阈值账）；③ **精确 n-gram 去重**（与 `dt-minhash-lsh-dedup` 近似去重划界：本题 hash 切片精确账）。
  `retrieval`：④ **Matryoshka embedding 截断**（2205.13147，官方仓待核）——不同维度截断的排名保持账；⑤ **ColBERT late interaction**（official `stanford-futuredata/ColBERT`）——MaxSim 逐 token 打分账。
  `inference`：⑥ **trie 前缀 mask 增量缓存**（official `mlc-ai/xgrammar`；与 `inf-grammar-mask-dfa` 划界：增量复用 vs 单步 DFA）；⑦ **正则 → token 级 FSM 编译**（official `dottxt-ai/outlines`）。
  attention 侧：⑧ **NSA 三支混合注意力**（2502.11089，official `deepseek-ai/NSA`）——压缩/选择/滑动三分支的块账；归属先与 kvcache/llm-models 查重再定。
  `evaluation`：⑨ **IF-Eval 可验证指令**（official `google-research/instruction-following-eval`）——逐条规则判别器；⑩ **LLM-as-judge swap 一致性**（lmsys 官方仓）——位置偏差 swap 账。
  **已被占勿再立项（本批盘点 grep 确认）**：MLA（`llm-deepseek-mla`×2）、模型合并 TIES/DARE（`peft-ties-dare-merge`）、DPO/IPO/KTO/ORPO/SimPO（to-* 5 题）、ALiBi、sliding-window、HNSW/IVF-PQ/RRF、unigram/wordpiece 分词、beam search、pass@k、minhash 去重、MoE aux/capacity、调度器系（continuous batching/EDF/priority/chunked prefill）、fp16 loss scaling、采样变体（adaptive-p/epsilon/llama-sampling-pipeline）。

### 各合集续批线索（已收口批次只留未立项候选、红线与待回源清单；落地细节 `git log` 可查）

- `peft` 已收口（2026-10-31 收口 637–639，系列累计 17 题，合集 36 题）。**余族候选（续作先回源 + 查重）**：① PSOFT/OSF（OSF 依赖精确 top-k SVD，oracle 独立性存疑，立项前先解决判题口径；HRA、BOFT、HiRA、SHiRA 已落地）；② WaveFT IDWT 路径（判题环境无 PyWavelets）/DeFT；③ MiSS/Lily/TinyLoRA/AdaMSS/GraLoRA/BeFT（RoAD、DeLoRA 已做）；④ FRoD/PVeRA/xLoRA/Cartridge/PEANUT/ShadowPEFT/Monteclora（VB-LoRA 已落地 531）；⑤ EVA/CoRDa/Astra/LoRA-GA/VeLoRA/BD-LoRA/Arrow/GenKnowSub（KaSA 在 peft 仓无 tuner 目录，已核实）；⑥ P-Tuning/AdaptionPrompt/CPT/LN-Tuning/POLY（TrainableTokens 已落地 639）；⑦ safe_merge/category 叠加/DLN 组合。新题沿用 `['operator', 'peft']`，纯函数/纯张量重建 ΔW 范式。
- `diffusion` 已收口（2026-10-31 收口 642–643，合集 48 题）。**次批候选（先回源 + 查重）**：SASolver、LMSDiscrete、PNDM/IPNDM（2202.09778）、KDPM2 两支、CosineDPMSolverMultistep（2407.14358）；SCM、TCDScheduler（`jabir-zheng/TCD`，与 523 LCM 边界近）、FlowMatchLCM；DPMSolverMultistepInverse（与 520 重叠风险）；`DDPMWuerstchen`（prior/pixel 两级 β）、`CogVideoXDDIM`/`CogVideoXDPMScheduler`（逐 token 噪声级）；hooks 侧 FreeU / mag_cache / taylorseer / sea_cache / faster_cache / first_block_cache / pyramid_attention_broadcast（与 kvcache/inference 缓存族查重后再定）。SDE 侧 v0.42.1 只剩 `scheduling_sde_ve.py`（VE 单型，与 522 重叠度高）与 `DPMSolverSDE` 可调旋钮（`s_churn` 等已不暴露，引用旧条目勿照抄）。**已作废（勿再立项）**：SDEDiscrete/SDE-VP、SLG（=504+617 完全重复）、STG（guiders/ 无独立文件）、CFG++、RePaint decision_threshold/I2LL。**红线**：真实 UNet/DiT 权重前向、训练损失曲线、`AutoPipeline` 接线、diffusers 侧 LoRA（归 peft）、VAE（归 operators）不出；PAG/SLG/STG/APG 类把分支预测量作显式入参只考合成公式；只出 1–2 题代表，勿为 49 个 scheduler 逐个出题。**判题环境**：`.venv-eval` 无 diffusers——ᾱ/σ 序列、t 索引、ε/z、抽样变量一律显式入参，独立纯 torch/numpy oracle 对拍 + 退化断言。
- `quantization` 已收口（2026-10-31 收口 7 题 order 618–622/624/625，合集 43 题；623 被并行会话 `fw-dtype-promotion-ledger` 占用按让号规则顺延）。**B 档逐条回源后不立项**：GPTVQ（Hessian k-means EM）、QTIP（LDLQ+可训练 LUT/CUDA）、AQLM（Adam 反传）、VPTQ（仓内无 quantizer 实现）、FlatQuant/AffineQuant/ResQ/q2n（AdamW 或激活 PCA 旋转）、ADADIM（逐维 MSE 校准）、LLM-FP4（校准区间搜索）、GuidedQuant（梯度引导+训练式旋转）、SignRoundV2/auto-round（signSGD 校准循环）、Caldera（QuIP# LDLQ+外部码本+交替优化）、QLLM（LoRA/离群拆分，与红线重叠）、SliM-LLM（tarball 不完整）、PrefixQuant/IntactKV（训练式 KV）、ZeroQuant-V2（无低秩补偿实现）、Layer-Wise（仅 README）、MX-QLLM/AMXFP4（与 `quant-mx-microscaling` 重叠）。**查重红线**：LoftQ 已由 395 覆盖（量化基座+交替 SVD 初始化不得重做）；`quant-squeezellm-sensitive-kmeans` 占敏感层 k-means 码本；`quant-awq-clip-scale-search` 占裁剪阈值二维网格；`llm-int8-outlier-split` 占离群列混合精度；`gptq-column-quantize` 占逐列 Hessian 误差补偿（RPTQ/GPTVQ 只能做列序/分组）；`quant-double-quant-scale` 占二级 scale；`quant-nf4-quantile-grid` 占 NF4 分位网格；KV 侧 `quant-kivi-kv-asym` 占不对称拆分（PrefixQuant/IntactKV 只能做前缀分段/枢轴保留）。**剔除**：需真实校准前向或 GPU kernel 者（QServe/T-MAC/Atom/Marlin/FP8-LM/COAT/SageAttention/EfficientQAT/QA-LoRA/Q-GaLore/FlexQ/flute/BitMoD/LLMC）与训练收敛类机制。**仓库归属**：OmniQuant=OpenGVLab（`wiki-xx/OmniQuant` 是 LLM-QAT fork，其路径 404 过）、AdaRound 走 `Saad-Fahmy/LLM-QAT`、BiasCompensation=`GongCheng1919/bias-compensation`。
- `sparsity` 已收口（2026-10-31 收口 640–641，合集 18 题）。**余下候选（续作先重抓源码）**：② SliceGPT（2401.15024，`microsoft/TransformerCompression`——eigh 特征向量符号与退化子空间无唯一解，须固定符号约定且测试谱非退化）；④ LoSparse（2306.11222——低秩+稀疏分解，与 `sparsity-sparsegpt-reconstruct`、`quant-gear-lowrank-outlier` 划界）；⑤ ZipLM/Bonsai/LLM-Barber（2302.04089/2402.05406/2408.10631——层/块级误差累积口径，先核对是否只是 Wanda/SparseGPT 调度变体）；另有 SVD-LLM/NoWag/Zero-Sum-SVD 低秩三族（若立项建议新开 `lowrank` 或并入「低秩×稀疏」轴，勿硬塞 quantization）。**暂不立项**：EvoPress（进化搜索随机性不可判）、PBS2P/ARMOR/SuperTuning（无开源实现）。**红线**：头级 `||W_h||_F·||X_h||_F`、元素级 Wanda、2:4 与元数据打包均已占。
- Efficient Prompting + Other 已收口（2026-10-31：新开 `context-compression` 3 题 + zip2zip 归 `data-tokenization`）。**待回源线索**：SVD-LLM（对称拆分 `U√S`/`√S V`）、NoWag（非对称拆分，与 SVD-LLM 二选一或只考「对称 vs 非对称 + 白化口径」）、Zero-Sum-SVD（`r=floor((1-p)mn/max(1,m+n))` clamp 到 `[1,min(m,n)]`，层间分配与 `kv-layer-budget-knapsack` 背包口径近，立项前须证明差异）、AdaSVD（2502.01403，先确认默认分支）、LongLoRA shift 注意力（半头偏移确定性 mask，查重 `sliding-window-attention`）、SageAttention INT8（逐块量化+K 平滑，查重 quant）、BAdam（块坐标优化，查重 opt）、DFloat11（共享指数变长浮点，查重 `quant-uint4-asym-pack`）、Basis Sharing（2410.03765）、D2MoE（2502.17298）、Selective_Context（2310.06201，与 LLMLingua 同族）、QUITO（2408.00274）、CPC（2409.01227）、SWE-Pruner（疑训练式分类器）、Batch Prompting（仅格式打包，价值低）、TriForce（spec 已 16 题）。**不立项**：IO-SVD（仓 size=0）、LORD（仅 HF 模型页）。
  **不立项（逐条判定）**：纯训练/RL/依赖大模型权重——Gisting、AutoCompressors、R2C、LightThinker、DPC、SAC、LatentPress、Density-aware、Highlight-Then-Summarize、AGORA、searchat、Gist-COCO、PromptCompressor、CoT-Influx、context-memory、AdaCoder、Rethinking-instruction、BatchGEMBA、CLLMs、MobileLLM、BiTA、OverFill、EfficientRollout、PromptDistill；工具/benchmark/系统——PCToolkit、SAMMO、GACC-Bench、LLMCBench、PowerInfer、Sequence-Scheduling、Unified-MoE-Compression、Dataset Quantization、Align、SPECTRE、MineDraft、SpecKV、SimSD、Flash-dLLM、ITC-MoE、traj-mc、TIDE、RapidIn；与已有题重复——SnapKV→`kv-snapkv-pooling`、DuoAttention→`kv-duo-attention-heads`、StreamingLLM→`kv-streaming-llm-sink`、Medusa→`spec-medusa-tree-verify`、FlashAttention→`flash-attention-tile`、GaLore→`opt-galore`、CaM→`kv-merging-cos-nearest`/`kv-wkvd-least-squares-merge`、KV_Compression sentinel tokens→kvcache 合集已覆盖。

### `vision-detection` 续批线索（2026-10-31 新增，源单：hoya012/deep_learning_object_detection）

- [ ] 100. `vision-detection` 续批（合集现 **8 题** order 373–380）。按「README 里**带源代码链接**的论文」出题（repo 中 FCOS/DETR/EfficientDet/SNIPER 等无码论文不作唯一出处；已有题的出处口径不动）。**次批候选（先逐条回源 + 查重，order 从 648 起，动笔前重扫全库避让并行会话）**：
  ① **OHEM**（1604.03540，official `abhi2610/Ohem-caffe`）——给定 per-sample loss 与阈值 t、keep 配额，输出难例选择 mask（先按 t 过滤再取 top-N，caffe 系按 batch 配额、论文另有 per-class 口径，立项前回源钉死一处）。与 `focal-loss` 划界：OHEM 是**选择**不是重加权，无调制因子。
  ② **CornerNet corner pooling**（1808.01244，official `princeton-vl/CornerNet`）——top-left / top-right pooling 的 running-max 前向账（右侧/下侧逆序 `cummax` 再相加），可附 heatmap 3×3 max-pool 峰值筛选 + top-k 账；纯前向、无随机数。
  ③ **SPP-Net 金字塔池化**（1406.4729，official caffe）——多级 bin（4×4/2×2/1×1）在任意 H×W 特征图上的 bin 边界整数除法 + 窗口 max + flatten 拼接的尺寸账。与 `roi_pool`（单级固定 bin）划界：增量在「多级金字塔 + 任意输入尺寸的划分账」。
  ④ **R-FCN position-sensitive RoI pooling**（1605.06409，official + unofficial）——k² 通道位置敏感得分图按 (i,j) bin 取对应通道 + bin 内平均投票；纯索引账。
  ⑤ **KL-Loss variance voting**（1809.08545，official caffe2）——NMS 后保留框坐标按邻居 σ² 倒数加权精修。与 `soft-nms`（压分数）划界：本题**改坐标不改分**；损失端与既有 `kl-divergence` 系列查重后只考 voting 侧。
  ⑥ **YOLO v1 网格责任编码**（1506.02640，official darknet）——GT 中心落格的责任分配、xy 网格偏移 + sqrt(w)/sqrt(h) 编码与 decode 互逆账。与 `vision-anchor-match`（anchor 系）划界：无 anchor 网格系。
  **跳过/不立项**：Cascade R-CNN（机制=换 IoU 阈值重训，无确定性判定点）、Relation-Network（relation attention 与 attention 系查重 + 需真实特征）、RefineDet/MS-CNN（尺度分配与 374 FCOS 层分配重复）、Mask R-CNN mask head（RLE/mask IoU 已有 377）、Gaussian YOLOv3（与⑤同类 uncertainty，二选一即可）、DeNet（角点系与②重复）。**红线**：真模型前向/训练收敛不出；判题环境只有 torch+numpy，先回源再立项、实现前逐条复查与 operators 合集 `nms`/`roi_pool`/`soft-nms`/`iou-compute`/`focal-loss` 的边界。

- [ ] 101. `vision-detection` **YOLO 全系列续批**（2026-10-31 新增，与 #100 同池同纪律：order 从 648 起统一取号、动笔前重扫全库避让并行会话、先逐条回源再立项）。v1 编码已由 #100⑥ 占，本条从 v2 起收；YOLOv4 及以后在 hoya012 repo 无码条目，出处一律引官方仓（`AlexeyAB/darknet`、`ultralytics/yolov3|yolov5|ultralytics`、`Megvii-BaseDetection/YOLOX`、`THU-MIG/yolov10`）。**候选（每条须独立回源钉口径 + 查重）**：
  ① **YOLOv2 anchor k-means 聚类**（1612.08242，official darknet）——GT 宽高集上以 `1−IoU` 为距离的 Lloyd 聚类出 anchor 尺寸（含 centroid 初始化与并列打破规则）；判题环境无 scipy，oracle 自实现 k-means、迭代轮次与初值作显式入参保证确定性。与 quant 合集各 k-means 题（码本/权重分配）划界：距离核与对象不同。
  ② **YOLOv2/v3 anchor 系 box 编解码互逆**（1612.08242 / 1704.04896，official darknet + ultralytics yolov3）——`bx=σ(tx)+cx`、`by=σ(ty)+cy`、`bw=pw·e^tw`、`bh=ph·e^th` 的编码/解码双向往返账（ultralytics 侧 `xywh2xyxy` 的 1−exp 防溢出 clamp 口径回源后二选一）。与 #100⑥ 划界：v1 是 sqrt(wh) 无 anchor，v2/v3 是 exp + anchor 先验。
  ③ **YOLOv3 多尺度预测分配 + 逐类 logistic 口径**（1704.04896）——三尺度特征图各自持 anchor 子集的分配账 + 类得分由 softmax 改逐类 sigmoid 的账（多标签口径）。与 `bce-with-logits`(361) 划界：361 只考损失形状，本题考检测头语义（无 cross-class 互斥）；分配部分与 374 FCOS 层分配划界（anchor 子集 vs 无 anchor 中心距）。
  ④ **YOLOv4 Mosaic 增强**（official `AlexeyAB/darknet`，ultralytics 实现同为参考）——四图按随机拼接中心 (x_s,y_s) 各裁互补象限再拼盘的坐标账；随机性全部作显式入参（拼接点、四图缩放系数），只考坐标/有效区域不读图像。全库无 augmentation 题（已 grep 确认），注意归集：若并入 data 合集需按其标签惯例，或留 `vision-detection`。
  ⑤ **YOLOv4 DIoU-NMS**（official AlexeyAB）——抑制判据由 IoU 换 `IoU−ρ²/c²` 的 NMS 变体。与 `nms`(66)/`soft-nms`(308)/`vision-rotated-nms`(379) 划界：本题换**判据**，soft-nms 换分数、379 换框形；惩罚项公式 `iou-variant-losses` 已覆盖，本题只复用口径不重推，判题点在「排序序变化导致的保留集差异」。
  ⑥ **YOLOX SimOTA 动态标签分配**（2107.08430，official `Megvii-BaseDetection/YOLOX`）——给定 cost 矩阵（cls loss + λ·reg loss，IoU 门控取 10）按动态 top-k（`dynamic_k = clamp(k_iou_min …)` 求和取整）选正样本 + 权重归一。与 `vision-anchor-match` 静态阈值划界；动态 k 的取整细节回源钉死（YOLOX 用 round，TOOD/PP-YOLOE 略异，只做 YOLOX 一家）。
  ⑦ **YOLOv5 build_targets 网格/anchor 双扩展分配**（official `ultralytics/yolov5`）——wh 比例 ≤ `anchor_t` 的静态 anchor 分配 + 对格点 `(gij)` 扩展 ±0.5 邻居偏移（off 集合 5 点）的候选扩展账，输出 tbox/tcls/tindex。纯索引账，与 ③⑥ 划界：这是 v5 特有的双扩展静态分配。
  ⑧ **YOLOv8 TAL 任务对齐分配器**（official `ultralytics/ultralytics`，思想源 TOOD 2108.07755）——alignment metric `t = s^α · u^β`、top-k 候选、mask 和为权重归一、跨 batch 唯一性仲裁（`select_highest_overlaps` 的重复 GT 处理）。分数与 IoU 矩阵全显式入参，确定性。与 ⑥ 划界：TAL 是 metric 排序，SimOTA 是 cost+动态 k。
  ⑨ **DFL 分布化 box 回归**（Generalized Focal Loss 2006.04388，YOLOv8 head 采用，official ultralytics + `implen/GFocal`）——l/t/r/b 逐边离散化为 `reg_max+1` bin 的期望 decode + `DFL = −((y_r−y)·log y_l + (y−y_l)·log y_r)` 的边权记账；与 `focal-loss` 划界：DFL 是两端点插值的分布回归，不是分类调制。
  **慎做/不立项**：YOLOv10 一对一/一对多双分配（一致性匹配训练式成分重，除非只考「一对一 head 上直接当最终输出」的平凡口径，价值低）；YOLOv6/v7（aux head、模型缩放，判定点弱）；YOLOv9 PGI/GELAN、YOLOv11/12（纯架构，CPU 无真实权重不可判）。**红线**：真模型前向、权重、训练收敛不出；`mish` 已有算子题不重出；CIoU/DIoU/GIoU 损失已被 `iou-variant-losses` 占、不得以「YOLOv4 loss」名义重做；判题环境无 scipy/torchvision，k-means 与 NMS 全部纯 torch/numpy 自实现。

### `pytorch` 合集扩批（2026-10-31 改名与重排新增）

- [x] 99. ⭐ `pytorch` 合集扩题（已改名 **PyTorch 入门** / *Intro to PyTorch*；合集现 **19 题**，合集内顺序 = `collections.json` 数组顺序，调学习顺序只改数组、不动 `meta.order`）。**批次一（并行会话）**：order 623 `fw-dtype-promotion-ledger`（Medium，类型提升三段式 + 写回门槛三禁令）、626 `fw-stride-layout-ledger`（Hard，shape/stride/storage_offset/连续性在 11 个视图序列上的逐步账 + view-or-copy 判定）。**批次二已收口（2026-10-31，2 题，order 646–647，合集 17→19）**：646 `fw-grad-graph-mechanics`（Hard，`['framework','autograd']`，插在 autograd-basics 之后）——12 步建图的机制账：`clone` vs `detach` 用 `data_ptr` 探针（`w_copies_input`）分界、`(w*2+1).sum()` 的链是 `SumBackward0→AddBackward0→MulBackward0→AccumulateGrad` 四项、no_grad 产物可再开梯度而 inference tensor 抛 `RuntimeError`、`retain_graph` 只在第一次传、梯度 `[2,2]→[4,4]` 累加证据；变异 7 有效全抓 + 1 等价（`t_rg_error` 恒 True 硬编码观测等价）。647 `fw-module-registration`（Medium，`['framework','nn-module']`，插在 nn-module-block 之后）——注册账：遍历按「本层 `_parameters`→`_buffers`→`_modules`」分桶（先赋值的子模块参数排在后赋值的直接 Parameter 之后）、`named_parameters` 去重留先注册名而 **`state_dict` 不去重**（共享参数双键）、`persistent=False` 不进 sd 但进 named_buffers、普通 list 是注册盲区、`register_parameter(None)` 双双隐身、buffer 重赋普通 Tensor 注册仍在、`.to()` 跟随；变异 9/9 全抓。均 selftest submit/run 全绿、torch 2.14.1 实测后出题。
  **剩余候选（未立项，续作时先实测 + 查重；order 从 648 起）**：③ DataLoader 批处理账——`batch_sampler`/`drop_last`/采样序列展开（与 253 的 DistributedSampler 分工：本题只管单进程装配）；④ `nn.init` 的 fan_in/fan_out 与 `calculate_gain` 口径账；⑤ 序列化侧 `map_location` / `weights_only` 与键级加载账（与 `fw-state-dict-remap` 的键重映射划清）；⑥ RNG 可复现账——`manual_seed` / 显式 generator / `initial_seed`，随机源全部作显式入参。**查重红线（勿再立项）**：batchnorm 前向（16）、归约轴（56）、where/select（42）、自定义 collate（119）、序列打包 padding（120）、分布式 clip-grad-norm（97）、`opt-clip-accum`（197）、`opt-ema`（198）、`fw-distributed-sampler`（253）、`bce-with-logits`（361）、`ag-cross-entropy-backward`（382）。**红线**：CUDA/device 不出、torchvision/PIL 不出、真实大模型前向不出；沿用「纯函数账 + torch 原生对拍」范式，测试只用裸 `assert`。

> 1. **不适合当前 CPU-only**：真 GPU kernel（归 #7）；需真实大模型前向 + 训练梯度的整模型反向（除非改走 torch.autograd 对拍原生）。
> 2. **撞车/无新机制，勿再立项**：label_smoothing 前向（=`to-label-smoothing-ce`）· 自定义 collate_fn（=`data-collator`/`sequence-pack-pad`）· SwiGLU/GEGLU（=`llm-swiglu-mlp`/`glu`，GEGLU 只作 variant）· masked/温度 softmax（=`causal-mask`/`attention-softmax-temp`）· Elder·inverse-RoPE（=`kv-key-rerotation-rope`）· NoPE（=position_ids 置零的一行退化）· QK-Norm 单列（=order37；relu² 整前向已落地 `llm-squared-relu-attention`）· EAGLE-1 特征级自回归草拟（需真实隐态）· batch 内 padding 浪费/ragged batching（=`sequence-pack-pad`；length bucketing 已判无新机制）· 模型并行 layer-wise 跨设备传输重叠（=`distributed-pipeline-*`，且时延假设无权威口径）· KV 量化与驱逐组合 pipeline（归 `quant-kivi-kv-asym`）· DDPM ancestral 整段循环（=递推叠 `diff-ddpm-step`，判题信号弱）· Tversky/IoU 损失族（=`dice-loss`/`iou-compute`/`focal-loss` 的系数泛化）· consistency/progressive 蒸馏 1-step 前向（依赖真实网络参数）· value-head token 级 reward 展开与 KL-in-reward（=`to-gae`/`kl-divergence`/`to-kl-controller`）· SWA 含 BN 重置（BN rebatch 需真实前向，去掉后与 `opt-ema` 重叠）· KFAC/真 Hessian Sophia-G（二阶信息不可确定性模拟）· Adam 的 eps 位置变体与无 bias-correction 子类（已被 adamw/adafactor/adabelief 参数空间覆盖）· HyDE/查询改写、graphRAG 社区摘要、LLM reranker（需模型生成，触 agentic 红线）· PIL/torchvision 真 resize、直调 faiss 逐位对拍（判题环境仅 torch+numpy）· 向量库分片路由/副本分布（=`inf-cache-aware-routing`）· **kvcache 侧经回源判定不立项**：CaM、EasyKV、VATP、NACL、Keyformer、KVQuant、SKVQ、KVTuner、CSKV、ParisKV、MaskKV、dKV-Cache、ArkVale（核心 topk 在 CUDA、CPU 侧记账与 `kv-quest-page-bound` 重复）、SparK（上游文件残缺、与 `kv-think-channel-mask` 同赛道）、KVSharer（仓库无 .py）、WindowKV（与窗口/预算/合并三题重叠）、`LogQuant`/`OTT` 的量化核（与 `quant-kivi-kv-asym` 逐行同，故只出索引调度已落 579/580）、`PyramidKV`/`H2OKV`/`ALLKV` 的 SCOPE 调度复制（同一段调度已落 585）、Page-EntroKV 的 `largest_remainder_quotas`（与 `kv-palu-rank-budget` 重复）、MiniCache（官方仓只有静态项目页无实现）、Eigen Attention（能确定性裁出的只有 SVD+累计能量定秩，与 `kv-xkv-layer-svd-share` 重复）、CacheBlend（主仓仅 README，选择性重算打分需真实前向，触校准红线）。

## 性能与新能力

- [ ] 16. 后端评测结果的用例级历史（最近 N 次提交记录、失败 diff 回看；涉及存储结构，先想清楚再做）
- [ ] 29. 进度统计小视图（可选）：利用 solved_at 时间戳做近 30 天做题热历/每日计数，纯激励向

## 大方向（需决策后再排期）

- [ ] 7. 第二阶段 Triton/CUDA 真实实现（variants 框架与下拉 UI 已就绪，现有题均只有 pytorch 可用 + triton/cuda 置灰）
  - 卡点：官方 Triton 无 macOS wheel，需先二选一
  - 方案 A（推荐）：OrbStack + linux/arm64 容器跑官方 triton CPU backend
  - 方案 B：triton 源码编译到 macOS（有历史失败记录，耗时不可控）
  - 待做：为算子题补 `variants/triton/`、`variants/cuda/` 三件套并把 meta.json 的 available 改 true，评测链路容器化；建议从 relu/sigmoid/softmax 等逐元素题起步再攻 attn/conv
