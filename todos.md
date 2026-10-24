# leet-infra 待办清单

> 按建议实施顺序编号；⭐ = 收益/成本比最高，优先做。
> **本清单只保留待办**：已完成项做完即分批提交，由 `git log` 追溯，不再在此归档。
> 当前题库状态：operators 80 + 框架 6 + perf-systems 9 + distributed 17 + llm-models 6 + data-tokenization 4 + quantization 7 + sampling 12 ＝ **141 题**，全部含 notes.md 详解、示例值实跑 dump 对拍；判题环境 torch+numpy（+ llm-models 合集 oracle 用离线 `transformers==5.20.2`）、强制 CPU、无 GPU/torchvision。全量后端 `-n auto` 449 passed（含题库完整性护栏 test_problem_integrity）+ 前端 33 vitest + build 全绿。

## 题库与内容

> 选题标准：面试高频 + 判题确定性好 + 不与现有重复。AI-infra 题均按「算法/数据流/显存账可在 CPU 模拟」为前提设计；真 GPU kernel 归 item 7。

### operators 零散承接（线代/损失/归一化缺口）

- [ ] 38. ⭐ 补 einsum / matmul·bmm / vector_norm·matrix_norm / logsumexp / bce_with_logits / kl_div / diff / repeat_interleave——均可直接对拍原生 torch，确定性最强，先做这批垫底座
- [ ] 46. operators 缺口第二批：label_smoothing_loss / focal / dice 等损失、group_norm·instance_norm 前向、softmax·cross_entropy 的 **backward**（对拍 `torch.autograd.grad`）——补齐 loss 家族与「反传」这一当前空白维度

### AI-infra 新合集候选（每个 = collections.json 新增一个 collection）

- [ ] 47. ⭐ 新合集 `moe`（专家并行/路由）：top-k gating（softmax→top-k→renorm）+ 负载均衡 aux loss（switch / coefficient of variation）+ capacity factor 丢 token + 专家分发（可与 `distributed-all-to-all` 交叉引用）——面试热词、CPU 确定性佳
- [ ] 41. ⭐ 新合集 `inference`（LLM 推理/Serving）：continuous batching 调度器 + speculative decoding 接受/拒绝准则（rejection sampling 精确对拍）+ beam search + KV cache 分页/LRU 驱逐
- [ ] 44. 新合集 `training-objectives`（对齐/训练目标）：DPO loss + PPO/GRPO 目标片段 + reward modeling + label smoothing（与 kl_div 呼应）
- [ ] 45. 新合集 `numerics`（数值稳定性显式成题）：softmax/logsumexp 减最大值、Welford/两-pass 方差、NaN/Inf 传播与归约
- [ ] 51. ⭐ 新合集 `llm-models`（大模型结构，像 transformers 逐组件实现→拼成整模型）：**Batch 1 Qwen2 骨干 6 题已落地**（order108–113：RMSNorm/RoPE/SwiGLU/GQA 注意力/decoder 层/整模型），均经真实 judge 对拍 HF eager AC。续做：Llama·Mistral（去 q/k/v bias、head_dim 由 hidden/nh 推）、DeepSeek-V3（MLA / noaux_tc MoE / QK-Norm）、Gemma 等其余家族复用同一 HF-oracle 范式

## 质量与工程优化

> 本批随合集扩张（前端新增 `systems`/`distributed` 两个族、collections.json 反复增删）暴露出的护栏缺口。

- [ ] 49. 加题脚手架 `scripts/add_problem.py` 增强：生成后顺带跑一次示例值 dump 自测，并提示「该题 domain tag 对应族是否已在 problemFilter 登记」，把当前靠人工记忆的步骤固化
- [ ] 50. 题量增长到 112 题后的前端复核：合集页分组/滚动与筛选性能、搜索可命中标题与标签、family 目录顺序随新合集增长的合理性

## 性能与新能力

- [ ] 16. 后端评测结果的用例级历史（最近 N 次提交记录、失败 diff 回看；涉及存储结构，先想清楚再做）
- [ ] 29. 进度统计小视图（可选）：利用 solved_at 时间戳做近 30 天做题热历/每日计数，纯激励向

## 大方向（需决策后再排期）

- [ ] 7. 第二阶段 Triton/CUDA 真实实现（variants 框架与下拉 UI 已就绪，现有题均只有 pytorch 可用 + triton/cuda 置灰）
  - 卡点：官方 Triton 无 macOS wheel，需先二选一
  - 方案 A（推荐）：OrbStack + linux/arm64 容器跑官方 triton CPU backend
  - 方案 B：triton 源码编译到 macOS（有历史失败记录，耗时不可控）
  - 待做：为算子题补 `variants/triton/`、`variants/cuda/` 三件套并把 meta.json 的 available 改 true，评测链路容器化；建议从 relu/sigmoid/softmax 等逐元素题起步再攻 attn/conv
