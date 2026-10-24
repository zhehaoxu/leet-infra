# leet-infra 待办清单

> 按建议实施顺序编号；完成一项勾选一项。⭐ = 收益/成本比最高，优先做。
> 未完成事项在前，已完成事项统一归档在末尾「已完成」区。
> 近期批次：算子合集「遗漏算子补全」24 题（Attention 4 / 卷积 3 / 池化插值 3 / 归约扫描 4 / 张量线代 3 / 采样 2 / 检测序列 3 / 激活 2）已完成提交，operators 合集 44→68 题、全量 self_ac 74 passed；24 题均含 notes.md 详解、题面示例值实跑 dump 对拍。
> 下一批建议顺序：**35（框架题/置灰题笔记覆盖面决策）→ 33（损失家族 E / 优化器数学 H / 量化 I 补题）→ 32/34（合集规模化与全量自测提速）→ 36/15（MarkdownView 回归单测）→ 7（Triton/CUDA 大方向）**。

## 题库与内容

- [ ] 35. ⭐ 笔记内容覆盖面决策：operators 68 题中 63 题已有 notes.md 详解，仅 5 道早期/置灰算子题（relu/sigmoid/leaky-relu/hardsigmoid/one-hot-encoding）与 6 道框架题（tensor-shape-indexing/broadcast-ops/autograd-basics/nn-module-block/optimizer-training-loop/dataset-dataloader）无笔记。裁决「是否给框架题也写详解 / 这 5 道基础算子确属无需详解」
- [ ] 32. 合集详情页大数据量评估：operators 已达 68 题，确认 ProblemTable 是否需要按 family 折叠分组（激活/归一化/卷积/池化插值/Attention/张量/采样/检测）或虚拟滚动
- [ ] 33. （可选，用户裁决）候选扩展组 E 损失家族、H 优化器数学、I 量化/低精度：候选题表已给过未入选（D 归约/扫描已在补全批次完成）；adam 类对拍建议 float64 或放宽 atol

## 性能与新能力

- [ ] 29. 进度统计小视图（可选）：利用 solved_at 时间戳做近 30 天做题热历/每日计数，纯激励向
- [ ] 34. test_self_ac 全量耗时增长：74 题参考解全量 AC 约 36s（单线程 warm worker），继续加题会线性变慢；评估 pytest-xdist 并行或拆「冒烟样例 / 全量」两档

## 工程与健康度

- [ ] 36. MarkdownView 数学保护逻辑的回归单测：`extractMath`/回填目前仅靠人工 E2E 保障（本轮就靠它抓到 rope 块级 pmatrix 裸奔 bug），建议加 vitest 用例锁住「$$块级 + \\ & _ 混合、代码块不被误吞、行内 $」几类场景防回归（与 15 合并做）
- [ ] 37. 孤儿 localStorage 清理：笔记改只读后历史键 `leet-infra:notes:<slug>` 已无引用，可加一次性清理或惰性忽略
- [ ] 15. 前端逻辑单测（vitest：store 过滤、草稿读写、tabs 记忆、MarkdownView 数学保护）
- [ ] 16. 后端评测结果的用例级历史（最近 N 次提交记录、失败 diff 回看；涉及存储结构，先想清楚再做）

## 大方向（需决策后再排期）

- [ ] 7. 第二阶段 Triton/CUDA 真实实现（variants 框架与下拉 UI 已就绪，68 题均只有 pytorch 可用 + triton/cuda 置灰）
  - 卡点：官方 Triton 无 macOS wheel，需先二选一
  - 方案 A（推荐）：OrbStack + linux/arm64 容器跑官方 triton CPU backend
  - 方案 B：triton 源码编译到 macOS（有历史失败记录，耗时不可控）
  - 待做：为算子题补 `variants/triton/`、`variants/cuda/` 三件套并把 meta.json 的 available 改 true，评测链路容器化；建议从 relu/sigmoid/softmax 等逐元素题起步再攻 attn/conv

## 已完成（归档）

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
