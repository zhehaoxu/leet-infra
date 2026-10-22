# leet-infra 待办清单

> 按建议实施顺序编号；完成一项勾选一项。⭐ = 收益/成本比最高，优先做。
> 未完成事项在前，已完成事项统一归档在末尾「已完成」区。
> 上一批（算子合集扩充 A+B+C+F+G 共 39 题）已全部完成：operators 合集 44 题，pytest 79 绿，分组提交。
> 下一批建议顺序：**31（新题批次 E2E 抽查）→ 30（遗留算子题挂载决策）→ 7（Triton/CUDA 真实实现）**。

## 题库与内容

- [ ] 30. 遗留算子题归入合集：`attention-softmax` / `cross-entropy-loss` / `multi-head-self-attention` 三道早期算子题目前不在任何合集里，建议评估质量后挂载进 operators（或单独成「进阶」合集），避免成为孤儿题
- [ ] 31. ⭐ 新批次 39 题前端 E2E 抽查：批量加题目前只验证到 pytest 层，需浏览器验证合集详情页 44 题渲染、题面 KaTeX（attention/rope 公式密集）、抽 1–2 题走「运行/提交」双轨（attention 合集无旧题混排、dropout 随机题重复提交确定性）
- [ ] 32. 合集详情页大数据量评估：operators 已达 44 题，确认 ProblemTable 筛选/分页体验是否需要虚拟滚动或分组小节（按 family 标签折叠：激活/归一化/卷积/Attention/张量）
- [ ] 33. （可选，用户裁决）候选扩展组 D 归约/扫描、E 损失家族、H 优化器数学、I 量化/低精度：候选题表已给过，未入选；adam 类对拍建议 float64 或放宽 atol

## 性能与新能力

- [ ] 29. 进度统计小视图（可选）：利用 solved_at 时间戳做近 30 天做题热历/每日计数，纯激励向
- [ ] 34. test_self_ac 全量耗时增长：49 题参考解全量 AC 已 ~37s，继续加题后会线性变慢；评估并行化（pytest-xdist）或拆分为「冒烟样例/全量」两档

## 工程与健康度

- [ ] 15. 前端逻辑单测（vitest：store 过滤、草稿读写、tabs 记忆；纯本地工具优先级低，可选）
- [ ] 16. 后端评测结果的用例级历史（最近 N 次提交记录，失败 diff 可回看；涉及存储结构，先想清楚再做）

## 大方向（需决策后再排期）

- [ ] 7. 第二阶段 Triton/CUDA 真实实现（variants 框架已就绪，44 题均只有 pytorch 可用 + triton/cuda 置灰）
  - 卡点：官方 Triton 无 macOS wheel，需先二选一
  - 方案 A（推荐）：OrbStack + linux/arm64 容器跑官方 triton CPU backend
  - 方案 B：triton 源码编译到 macOS（有历史失败记录，耗时不可控）
  - 待做：为算子题补 `variants/triton/`、`variants/cuda/` 三件套并把 meta.json 的 available 改 true，评测链路容器化；建议从 relu/sigmoid/softmax 等逐元素题起步再攻 mattn/conv

## 已完成（归档）

- [x] 算子合集扩充批次（A+B+C+F+G，39 题）：A 激活 9（tanh/leaky/elu/prelu/hardswish/hardsigmoid/mish/softplus/glu）、B 归一化 6（batchnorm/layernorm/rmsnorm/groupnorm/instancenorm/weightnorm）、C 池化卷积 8（maxpool/avgpool/adaptive/globalpool/conv2d/im2col/bilinear/bmm）、F Attention 9（sdpa/mha/causal-mask/kv-cache/rope/alibi/temp-top-p/qk-norm/dropout）、G 张量操作 7（embedding/one-hot/gather/where/pad/unfold-dilated/concat）；全部题面示例值实跑 dump、逐组 importlib 自测 + collections.json 挂载 + pytest 全绿（79 passed）、分 5 次提交
- [x] test_api 硬编码题数修复：EXPECTED_TOTAL 由文件系统动态推导，collections 断言改读 collections.json，加题不再打破测试
- [x] 题库重构批次：题目目录升级为 `variants/<implementation>/` 三件套结构；算子合集重建为 relu/sigmoid/swish/softmax/gelu 5 题；PyTorch 合集重设计为框架使用 6 题；旧 5 题删除
  - 后端：`Problem.resolve_variant` 不可用回退默认、API `?variant=` 与 `Submission.variant`、benchmarks 存储加 variant 维度；进度按 slug 记录
  - 前端：合集详情页「实现方式」分段选择器（localStorage 偏好）、做题页 variant 控件（不可用置灰 + 回退提示）、草稿按 slug×variant 隔离、导航链接透传 `?variant`
  - 测试：test_api/test_judge 适配新题库 + 新增 test_variants、test_self_ac；脚手架 add_problem.py 生成新结构
- [x] 1. 代码草稿持久化（localStorage 每题一份，含浏览器验证）
- [x] 2. git 初始化 + 首次提交（main，56 文件，环境/产物已忽略）
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
- [x] 25. 题目笔记（做题页第三标签，markdown 编辑/预览，localStorage）
- [x] 26. 单题重刷（unmark API +「⟳ 重新做」，笔记保留）
- [x] 27. 进度导出/导入（/api/progress/export + import 过滤未知 slug）
- [x] 28. 编辑器字号调整（全局 editorFontSize，10–24 限幅实时跟随）
- [x] 额外完成：左侧题面 Tabs、首页合集 Tabs + collections.json、标签页记忆、顶栏 logo/favicon、合集卡片圆环进度图标、quick win 批次 E2E 验证
