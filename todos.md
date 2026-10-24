# leet-infra 待办清单

> 按建议实施顺序编号；完成一项勾选一项。⭐ = 收益/成本比最高，优先做。
> 未完成事项在前，已完成事项统一归档在末尾「已完成」区。
> 近期批次：算子补全 24 题 + 扩展组 E/H/I 8 题 + 缺口批次 7 题（lrn/conv1d/conv3d/mqa/gru_cell/roi_pool/roi_align，其中 roi_* 用纯 torch + 独立 numpy 校验）已完成，operators 合集 44→68→76→83 题、题库总计 89 题；全量 self_ac 已并行化（`-n auto` 约 14s，串行约 57s）。
> 笔记覆盖面：operators 83 + 框架 6 + perf-systems 9 + distributed 14 共 112 题全部含 notes.md 详解；题面示例值均实跑 dump 对拍（perf-systems 为纯计算、distributed 为多 rank 张量模拟，均用独立重算交叉核对）。
> perf-systems 已扩到 9 题：激活显存/梯度检查点、FLOPs/MFU、算子融合、参数量估算、训练态显存四件套+ZeRO、KV cache 显存（含 MQA/GQA）、Roofline 算术强度、环形 All-Reduce 通信量、LLM 推理 prefill/decode 吞吐账。
> distributed 首批 4 题（order94–97）：All-Gather / Reduce-Scatter / All-Reduce 集合通信原语（多 rank 张量模拟逐 rank 终值）+ 分布式梯度裁剪（全局范数不能各 rank 独立裁）；前端新增 `distributed`→「分布式」族。
> distributed 第二批 4 题（order98–101）：Broadcast / Scatter 补全原语 + ZeRO-1/2/3 每 GPU 通信量计算（纯计算，接 training-state-memory）+ Ring All-Reduce 逐步通信轮次模拟（返回 2(W-1) 个中间态快照，按题面精确调度对拍）。
> distributed 第三批 3 题（order102–104）：All-to-All（网格转置、MoE 专家并行核心）+ 流水线并行 GPipe 气泡率（(p-1)/(m+p-1) 纯计算）+ ZeRO-Offload 显存-PCIe 带宽取舍（Hard 计算）；合集从集合通信原语扩展到 DP/TP/PP/EP 多维并行的显存-带宽-气泡账本。
> distributed 第四批 3 题（order105–107）：Gather（单目标拼接、补全 scatter/collect 对称）+ 1F1B 峰值激活显存（min(m,p-i)，接 pipeline-bubble）+ 张量并行每层通信（Megatron 每层 4 次 all-reduce 的激活量）；至此集合通信原语矩阵与 DP/TP/PP 账本基本齐备。
> 下一批建议顺序：AI-infra 方向扩展（见「题库与内容」38–45，43 perf-systems 已落地两批；均可 CPU 确定性判题）；Triton/CUDA 真核仍卡在 item 7（需 GPU 容器化）。

## 题库与内容

> 选题标准：面试高频 + 判题确定性好 + 不与现有重复。判题环境仅 torch+numpy、强制 CPU、无 GPU/torchvision，故以下 AI-infra 题均按「算法/数据流/显存账可在 CPU 模拟」为前提设计；真 GPU kernel 归 item 7。

### operators 零散承接（线代/损失缺口）

- [ ] 38. 补 einsum / matmul·bmm / vector_norm·matrix_norm / logsumexp / bce_with_logits / kl_div / diff / repeat_interleave——均可直接对拍原生 torch，确定性最强，先做这批垫底座

### AI-infra 新合集候选（每个 = collections.json 新增一个 collection）

- [ ] 39. ⭐ 新合集 `distributed` 分布式训练原语（已落地 14 题 order94–107，见归档）：基本齐备；可选续加 Reduce（单目标求和）/ ring vs recursive-doubling 延迟 / ZeRO-Infinity NVMe / MoE 专家负载路由
- [ ] 40. ⭐ 新合集 `data-tokenization`：BPE 分词器（merge 训练 + 编码从零实现）+ data collator（padding/attention mask 生成）+ 序列 pack/pad——几乎无浮点、确定性最佳
- [ ] 41. 新合集 `inference`（LLM 推理/Serving）：continuous batching 调度器 + speculative decoding 接受/拒绝准则（rejection sampling 精确对拍）+ beam search + typical/min-p 采样 + KV cache 分页/LRU 驱逐
- [ ] 42. 新合集 `quantization`（现有对称/非对称 2 题独立扩展）：per-channel / group-wise fake-quant + weight-only 反量化 matmul 模拟 + autocast + loss scaling（fp16 溢出/下溢确定性模拟）
- [ ] 44. 新合集 `training-objectives`（对齐/训练目标）：DPO loss + PPO/GRPO 目标片段 + reward modeling + label smoothing（与 kl_div 呼应）
- [ ] 45. 新合集 `numerics`（数值稳定性显式成题）：softmax/logsumexp 减最大值、Welford/两-pass 方差、NaN/Inf 传播与归约

## 性能与新能力

- [ ] 29. 进度统计小视图（可选）：利用 solved_at 时间戳做近 30 天做题热历/每日计数，纯激励向

## 工程与健康度

- [ ] 16. 后端评测结果的用例级历史（最近 N 次提交记录、失败 diff 回看；涉及存储结构，先想清楚再做）

## 大方向（需决策后再排期）

- [ ] 7. 第二阶段 Triton/CUDA 真实实现（variants 框架与下拉 UI 已就绪，68 题均只有 pytorch 可用 + triton/cuda 置灰）
  - 卡点：官方 Triton 无 macOS wheel，需先二选一
  - 方案 A（推荐）：OrbStack + linux/arm64 容器跑官方 triton CPU backend
  - 方案 B：triton 源码编译到 macOS（有历史失败记录，耗时不可控）
  - 待做：为算子题补 `variants/triton/`、`variants/cuda/` 三件套并把 meta.json 的 available 改 true，评测链路容器化；建议从 relu/sigmoid/softmax 等逐元素题起步再攻 attn/conv

## 已完成（归档）

- [x] 43. ⭐ 新合集 `perf-systems`（AI 系统性能，3 题 order85–87）：`transformer-mem`（逐层激活显存账 + 梯度检查点「存输入/重算一层」，返回整数字节）、`transformer-mfu`（每层 matmul 参数量 + attention 非参数 4sh FLOPs、step=3×前向≈6ND、MFU=achieved/peak）、`operator-fusion`（逐元素链 HBM 流量、融合收益、memory-bound 下 speedup=算子数）。均为**纯计算题**（题面自定简化教学公式，无外部 oracle），test_cases 用独立 Python 重算核对；示例值草稿脚本跑实。前端 problemFilter 新增 `systems`→「显存 · 性能」族、collections.json 加 `perf-systems` 合集。全量 self_ac 并行 122 passed、前端 33 vitest + build 全绿

- [x] 43（第二批）. `perf-systems` 续加 3 题 order88–90：`transformer-params`（按结构清点 4h²+2hf+4h/层 + 嵌入 tied/untied + final LN，GPT-2-small 风格≈123.6M）、`training-state-memory`（混合精度+Adam 16 bytes/参数四件套 + ZeRO stage1/2/3 每卡分区比）、`kv-cache-memory`（`2·L·kv_heads·d_head·seq·batch·dtype` 字节，MQA/GQA 共享 KV 头缩小比例）。仍为纯计算题，test_cases 用逐分量/逐层独立重算交叉核对、示例值草稿脚本跑实。perf-systems 3→6 题、题库总计 92→95。全量后端 `-n auto` **125 passed**、前端 33 vitest + build 全绿

- [x] 43（第三批）. `perf-systems` 续加 3 题 order91–93：`roofline-arithmetic`（算术强度 AI=flops/bytes、屋脊点 peak/bw、可达=min(peak,AI·bw) 与 memory/compute-bound 判定，AI==ridge 归 compute）、`parallel-allreduce`（环形 2·(W-1)/W·S 每卡通信量 + 逐跳延迟耗时，W≤1 零通信）、`inference-throughput`（每 token 2N、prefill=batch·prompt·2N 定 TTFT、decode=batch·gen·2N，同一 effective_flops 折时）。仍为纯计算题，test_cases 用不同推导路径的独立重算 + 不变量（体积≤2S、total≥ttft、gen_len=0）交叉核对。perf-systems 6→9 题、题库总计 95→98。全量后端 `-n auto` **128 passed**、前端 33 vitest 全绿

- [x] 39（首批）. ⭐ 新合集 `distributed` 分布式训练原语首批 4 题 order94–97（多 rank 张量模拟、非纯公式）：`distributed-all-gather`（Easy，各 rank 拿按 rank 顺序拼接的完整结果）、`distributed-reduce-scatter`（Medium，逐元素求和后 rank i 取第 i 段，各 rank 不同）、`distributed-all-reduce`（Medium，求和广播到所有 rank，含 gather∘scatter 组合恒等）、`distributed-clip-grad-norm`（Hard，全局 p-范数统一裁剪 scale，不能各 rank 独立裁，返回 (clipped, total_norm)，含不等长分片）。集合通信建模为纯函数（输入=通信前各 rank 数据列表、输出=通信后各 rank 数据列表）。前端 problemFilter 新增 `distributed`→「分布式」族、collections.json 加 `distributed` 合集。distributed 0→4 题、题库总计 98→102。全量后端 `-n auto` **132 passed**、前端 33 vitest + build 全绿

- [x] 39（第二批）. `distributed` 续加 4 题 order98–101：`distributed-broadcast`（Easy，src 张量复制给所有 rank，考 src≠0 取对应索引）、`distributed-scatter`（Medium，All-Gather 逆操作，src 完整数据均分 W 段，含 cat(scratch)==full 互逆不变量）、`distributed-zero-communication`（Medium 纯计算，DDP/ZeRO-1-2/ZeRO-3 每 GPU 每迭代通信字节，环系数 c=(W-1)/W，ZeRO-1/2≡DDP、ZeRO-3 多 2cP，接 training-state-memory）、`distributed-ring-allreduce-steps`（Hard，按题面精确调度模拟 Reduce-Scatter + All-Gather，返回 2(W-1) 个 (W,W) 中间态快照；同步轮语义+下标公式易错，题面给 W=3 完整轨迹）。示例值均用临时脚本实跑 dump（含 W=2..8 随机普适验证、ownership=q 诊断）后删脚本。collections.json `distributed` 4→8 slug、题库总计 102→106（前端 `分布式` 族已存在无需改）。全量后端 `-n auto` **136 passed**、前端 33 vitest + build 全绿

- [x] 39（第三批）. `distributed` 续加 3 题 order102–104（合集从集合通信原语扩展到多维并行账本）：`distributed-all-to-all`（Medium，多 rank 张量模拟，out[j]=各 rank 发给 j 的分片按源序拼接，即 (src,dst) 网格转置；参考解用 reshape-transpose、测试用循环拼接两路独立对拍，含退化为 All-Gather 恒等与元素守恒）、`distributed-pipeline-bubble`（Medium 纯计算，GPipe 气泡率 (p-1)/(m+p-1)、利用率 m/(m+p-1)、makespan m+p-1；测试用对角填充数忙碌格的独立模拟做 oracle，校 bubble+util==1、p=1/m=1 边界、m 增气泡单调降）、`distributed-zero-offload`（Hard 纯计算，GPU 只留 fp16 参数=2N、CPU 扛梯度+优化器=14N、每迭代 PCIe=4N、t_pcie>BW 比较定瓶颈（严格大于），接 training-state-memory/zero-communication 凑齐 ZeRO 显存-带宽三角）。示例值用临时脚本实跑 dump（all-to-all 两路一致、pipeline/offload 多组）后删脚本。collections.json `distributed` 8→11 slug、题库总计 106→109（前端 `分布式` 族复用）。全量后端 `-n auto` **139 passed**、前端 33 vitest + build 全绿

- [x] 39（第四批）. `distributed` 续加 3 题 order105–107：`distributed-gather`（Medium 多 rank 模拟，单目标拼接：out[dst]=各分片 cat、其余 rank 原样保留自己那片（区别于 all-gather），与 scatter 互逆）、`distributed-pipeline-1f1b-memory`（Hard 纯计算，GPipe 每段存 m 份 vs 1F1B 第 i 段存 min(m,p-i) 份，返回 (gpipe_peak, offfb_worst, offfb_best)；m≤p 最重段不省、m>p 比值 p/m；测试用逐段 profile 取 max/min 做独立 oracle）、`distributed-tensor-parallel-comm`（Medium 纯计算，Megatron TP 每层 4 次激活 all-reduce，act=batch·seq·hidden·dtype、per_layer=8·(W-1)/W·act、total×n_layers；W=1 零通信、act 与 world 无关）。示例值用临时脚本实跑 dump 后删。collections.json `distributed` 11→14 slug、题库总计 109→112（前端 `分布式` 族复用）。全量后端 `-n auto` **142 passed**、前端 33 vitest + build 全绿

- [x] 37. 孤儿 localStorage 一次性清理：新增 utils/storageCleanup.ts（`removeKeysWithPrefixes` + `cleanupOrphanStorage`，哨兵标记键 `leet-infra:storage-cleanup-v1` 保证只跑一次、后续启动 O(1) 跳过、隐私模式/配额异常静默不崩溃），main.ts 启动时调用，清除早期可编辑笔记遗留的 `leet-infra:notes:*` 键；前缀列表可续用于后续遗留键。6 个 vitest 用例 + `npm run build`(vue-tsc) 通过

- [x] 35. 笔记覆盖面补齐（提交 5675789）：为 5 道基础算子（relu/sigmoid/leaky-relu/hardsigmoid/one-hot-encoding）+ 6 道框架题（tensor-shape-indexing/broadcast-ops/autograd-basics/nn-module-block/optimizer-training-loop/dataset-dataloader）补 notes.md，题库 82 题全含详解；同步修正 test_api 的「notes 缺失返回空」断言改为 tmp 目录单元隔离（不再依赖 relu 恰好无笔记）
- [x] 33. 候选扩展组 E/H/I 新题 8 道（提交 fe98e35/580a2c3/b761ee8）：E 损失家族 mse/cross-entropy/bce/huber（对拍 F.mse_loss/F.cross_entropy/F.binary_cross_entropy/F.huber_loss）、H 优化器数学 sgd-momentum-step/adam-step（复刻 torch.optim 单步、float64 对拍 <1e-12）、I 量化 symmetric/asymmetric fake-quant（对拍 torch.fake_quantize、zero_point 越界 clamp）；operators 68→76，全量 self_ac 全绿；负例均用 try/except ValueError（判题 worker 无 pytest）
- [x] 32. ProblemTable 按 family 折叠分组（提交 928f75a）：68 题合集可导航，curated tag→族映射 + 固定族顺序、组头可折叠、分组开关随 persistKey 持久化、首页与合集页启用 groupable（68 行无需虚拟滚动）；vue-tsc + build 通过
- [x] 34. test_self_ac 提速（提交 0447112/aa97ac2）：新增 backend/tests/conftest.py（sys.path 注入 + `--smoke` 档），test_self_ac 按 --smoke 切 /run(样例) 与 /submit(全量)；setup_eval_env.sh 后端依赖加 pytest-xdist；README 补 `-n auto`/`--smoke` 用法。实测 82 题：串行~57s → `-n auto` 并行~14s；并修复并行暴露的基准 daemon 竞态（首 AC 后台线程赶在响应前落盘，禁用该次 spawn 消除竞态），连续多次并行全绿
- [x] 36+15. 前端 vitest 单测（提交 bee7ea2）：装 vitest@2 + @vue/test-utils + jsdom，vite.config 加 test 段、package.json 加 test/test:watch；抽纯逻辑到 utils/{markdownMath,problemFilter,draft,uiPrefs}.ts（组件改为 import 消费、行为不变）；27 用例覆盖 extractMath 块/行内/代码保护/转义 + MarkdownView 挂载回归（`_ & \` 不生成 `<em>`）+ 过滤三重条件 + familyOf 归族优先级 + buildSections 排序 + draftKey 隔离 + 首页标签记忆 + pinia store；`npm test` 全绿、`npm run build` 通过
- [x] ⭐ 算子合集「遗漏算子补全」24 题（B1–B8，提交 45fcfaa/37a5858/73ca0d1/9542f9a/5d82c3e/25e4463/aa1361f/2e36e04）：B1 Attention(GQA/Flash分块/滑动窗口/Soft-Capping)、B2 卷积(分组/转置/深度可分离)、B3 池化插值(最近邻上采样/自适应最大池化/grid_sample)、B4 归约扫描(按轴求和/累积和/Top-K/排序argsort)、B5 张量线代(线性层/余弦相似度/log-softmax)、B6 采样(Gumbel-Softmax/Top-p核采样)、B7 检测序列(IoU/NMS/LSTM单元)、B8 激活(ReLU6/SELU)；operators 44→68、全量 self_ac 74 passed；每题 notes.md + 题面 dump 对拍；采样类噪声作显式输入保证确定性
- [x] ⭐ 新批次前端 E2E 抽查（原 31）：浏览器验证 operators 合集 44 行渲染 + 实现方式下拉、rope/alibi 等题面 KaTeX、relu 走完整「提交→AC→参考答案解锁→重新做→再 AC」双轨与彩花/用时展示、笔记只读；**抽查抓到并修复一处渲染 bug**——rope 题面块级 `$$\begin{pmatrix}` 公式因 marked 误处理 `\\ & _` 导致 KaTeX 匹配不到定界符而裸奔，改 MarkdownView 先保护代码块再抽出数学占位、marked 处理后 HTML 转义回填根治（relu 代码块/softmax 混排/笔记均复验无回归、控制台 0 error）
- [x] 笔记标签改为只读渲染 + 移除「编辑/预览」切换与顶部 A−/A+ 编辑器字号按钮（提交 003325f）
- [x] 题目算法详解体系：为算子合集 39 题撰写 notes.md（推导/数值稳定/真实语义坑/跨题关联/对拍口径），后端 `Problem.notes()`+`get_problem.notes` 字段、前端笔记标签默认只读渲染并在无 notes.md 时置灰不可选（提交 95e3583）
- [x] 实现方式选择器由按钮组改为下拉 `<select>`（工作页 + 合集页两处，不可用项 disabled+「即将支持」后缀，提交 b09f33a）
- [x] 算子合集扩充批次（A+B+C+F+G，39 题）：A 激活 9、B 归一化 6、C 池化卷积 8、F Attention 9、G 张量操作 7；全部题面示例值实跑 dump、逐组 importlib 自测 + collections.json 挂载 + pytest 全绿（79 passed）、分 5 次提交
- [x] test_api 硬编码题数修复：EXPECTED_TOTAL 由文件系统动态推导，collections 断言改读 collections.json，加题不再打破测试
- [x] 题库重构批次：题目目录升级为 `variants/<implementation>/` 三件套；算子合集重建 relu/sigmoid/swish/softmax/gelu 5 题；PyTorch 合集重设计为框架使用 6 题；旧 5 题删除。后端 `resolve_variant` 回退 + API `?variant=` + Submission.variant + benchmarks 加 variant 维度；前端 variant 控件 + 合集偏好 + 草稿按 slug×variant 隔离 + 导航透传；测试 test_variants/test_self_ac；脚手架支持新结构
- [x] 1. 代码草稿持久化（localStorage 每题一份，含浏览器验证）
- [x] 2. git 初始化 + 首次提交（main）
- [x] 3. 题库热加载（_INDEX 按 mtime 指纹失效，增删题目零重启）
- [x] 4. 运行/提交分离（run 只跑样例不记录，submit 全量 + AC 彩花特效）
- [x] 5. ⭐ 做题页上一题/下一题导航（subheader 导航组 + AC 后「下一道未解」胶囊）
- [x] 6. 耗时对比展示（首次 AC 后台基准测试存 benchmarks.json，通过用例显示「参考 x.xms · N.N×」）
- [x] 8. PyTorch 题库扩充：框架使用 6 题（tensor-shape-indexing / broadcast-ops / autograd-basics / nn-module-block / optimizer-training-loop / dataset-dataloader）
- [x] 9. 参考答案标签页升级为 Monaco 只读模式（只读高亮 + ⧉ 复制）
- [x] 10. 提交快捷键 ⌘⇧↵
- [x] 11. 合集详情页搜索/难度/状态筛选（抽离共享 ProblemTable 组件）
- [x] 12. 「重置进度」入口（二次确认，重置后参考答案重新锁定）
- [x] 13. 失败时结果面板自动滚动到首个失败用例、默认展开失败项
- [x] 14. scripts/start.sh 一键生产模式（构建+启动，README 已更新）
- [x] 17. ⭐ 筛选状态保持（ProblemTable persistKey 独立记忆）
- [x] 18. 题目标注所属合集（summary 附 collections，行内徽章跳转）
- [x] 19. 站内「? 帮助」弹窗（快捷键/双轨规则/评测环境/题库扩充）
- [x] 20. 运行/提交可取消（/api/judge/cancel + 运行中「✕ 取消」）
- [x] 21. 圆环进度抽成 ProgressRing 小组件
- [x] 22. git 提交检查点（分批功能提交，工作区干净）
- [x] 23. ⭐⭐ 评测提速（常驻 warm worker，崩溃自动回退 legacy）
- [x] 24. ⭐ 加题脚手架 scripts/add_problem.py（交互式五件套 + 合集询问 + 热加载验证）
- [x] 25. 题目笔记（做题页第三标签，localStorage markdown）— **后续演进见上：已升级为 notes.md 只读详解 + 置灰，编辑/预览切换已移除**
- [x] 26. 单题重刷（unmark API +「⟳ 重新做」，笔记保留）
- [x] 27. 进度导出/导入（/api/progress/export + import 过滤未知 slug）
- [x] 28. 编辑器字号调整（全局 editorFontSize）— **后续演进：A−/A+ 按钮已按需求移除，仅保留编辑器默认字号**
- [x] 额外完成：左侧题面 Tabs、首页合集 Tabs + collections.json、标签页记忆、顶栏 logo/favicon、合集卡片圆环进度图标、quick win 批次 E2E 验证
