# leet-infra 待办清单

> 按建议实施顺序编号；完成一项勾选一项。⭐ = 收益/成本比最高，优先做。
> 本批（题库重构：算子多实现变体 + PyTorch 框架合集重建）已全部完成并通过 E2E。
> 下一批建议顺序：**7（Triton/CUDA 真实实现）**；可选小活 29、15/16。

## 已完成

- [x] 题库重构批次（本轮）：题目目录升级为 `variants/<implementation>/` 三件套结构；算子合集重建为 relu/sigmoid/swish/softmax/gelu 5 题（每题声明 pytorch 可用 + triton/cuda 置灰）；PyTorch 合集重设计为框架使用 6 题（张量形状索引/广播/autograd/MLP block/训练循环/Dataset）；旧 5 题删除。
  - 后端：`Problem.resolve_variant` 不可用回退默认、API `?variant=` 与 `Submission.variant`、benchmarks 存储加 variant 维度；进度按 slug 记录
  - 前端：合集详情页「实现方式」分段选择器（localStorage 偏好）、做题页 variant 控件（不可用置灰 + 回退提示）、草稿按 slug×variant 隔离、导航链接透传 `?variant`
  - 测试：test_api/test_judge 适配新题库 + 新增 test_variants、test_self_ac（11 题参考解真实评测全 AC），40 passed；浏览器 E2E 7/7 PASS
  - 脚手架 add_problem.py 生成新结构，可选多实现算子题
- [x] 1. 代码草稿持久化（localStorage 每题一份，含浏览器验证）
- [x] 2. git 初始化 + 首次提交（main，56 文件，环境/产物已忽略）
- [x] 3. 题库热加载（_INDEX 按 mtime 指纹失效，增删题目零重启）
- [x] 4. 运行/提交分离（run 只跑样例不记录，submit 全量 + AC 彩花特效）
- [x] 额外完成：左侧题面 Tabs（题目说明/参考答案）、首页合集 Tabs + collections.json + 合集详情页、标签页记忆、顶栏 logo 与 favicon、合集卡片圆环进度图标、上一题/下一题导航与参考答案只读化（即原 5/9 项）
- [x] quick win 批次已全部完成（10/12/13/14/17/18/19/21/22，含浏览器 E2E 9/9 验证）

## 性能与新能力（本批新增，建议优先）

- [x] 23. ⭐⭐ 评测提速（常驻 worker）（已完成：预热进程流水线 WORKER_SRC，spawn 与空闲期重叠，崩溃自动回退 legacy 路径；测试验证结果语义不变）
- [x] 24. ⭐ 加题脚手架 scripts/add_problem.py（已完成：交互式五件套 + 合集询问 + 热加载验证通过）
- [x] 25. 题目笔记（已完成：做题页第三标签，markdown 编辑/预览切换，localStorage 每题一份，复用草稿防抖）
- [x] 26. 单题重刷（已完成：POST /api/problems/{slug}/unmark + 已解题「⟳ 重新做」按钮，清 solved+草稿、笔记保留）
- [x] 27. 进度导出/导入（已完成：/api/progress/export 下载 + import 覆盖恢复并过滤未知 slug，首页入口）
- [x] 28. 编辑器字号调整（已完成：全局 editorFontSize + localStorage，工作bench A−/A+ 控件，10–24 限幅，所有 Monaco 实例实时跟随）
- [ ] 29. 进度统计小视图（可选）：利用 solved_at 时间戳做近 30 天做题热历/每日计数，纯激励向

## 做题体验优化

- [x] 5. ⭐ 做题页上一题/下一题导航（已完成：subheader 导航组 + AC 后「下一道未解」胶囊，浏览器验证通过）
- [x] 9. ⭐ 参考答案标签页升级为 Monaco 只读模式（已完成：只读高亮 + ⧉ 复制代码按钮，浏览器验证通过）
- [x] 10. 提交快捷键 ⌘⇧↵（已完成：Monaco 注册 Ctrl+Shift+Enter 提交，空状态文案说明两个快捷键）
- [x] 11. 合集详情页增加搜索/难度/状态筛选（已完成：抽离共享 ProblemTable 组件，首页与详情页复用，浏览器验证通过）
- [x] 12. 「重置进度」入口（已完成：POST /api/progress/reset + 首页二次确认按钮，重置后参考答案重新锁定）
- [x] 13. 运行/提交失败时结果面板自动滚动到首个失败用例，用例面板默认展开失败项、折叠通过项（已完成）
- [x] 6. 耗时对比展示（已完成：首次 AC 后台用参考解测基准存 data/benchmarks.json，run/submit 结果附带，通过用例显示「参考 x.xms · N.N×」，仅展示不设门槛）
- [x] 17. ⭐ 筛选状态保持（已完成：ProblemTable persistKey 存 localStorage，首页/各合集独立记忆）
- [x] 18. 题目标注所属合集（已完成：后端 summary 附 collections，行内徽章点击跳合集详情页）
- [x] 19. 站内「? 帮助」弹窗（已完成：顶栏 ? 按钮，快捷键/双轨规则/评测环境/题库扩充四节，Esc 关闭）
- [x] 20. 运行/提交可取消（已完成：随 23 一并实现，POST /api/judge/cancel + 运行中「✕ 取消」按钮）

## 工程与健康度

- [x] 14. scripts/start.sh 一键生产模式（已完成：构建+启动，可选端口参数，README 已更新）
- [ ] 15. 前端逻辑单测（vitest：store 过滤、草稿读写、tabs 记忆；纯本地工具优先级低，可选）
- [ ] 16. 后端评测结果的用例级历史（最近 N 次提交记录，失败 diff 可回看；涉及存储结构，先想清楚再做）
- [x] 21. 圆环进度抽成 ProgressRing 小组件（已完成：首页合集卡片 + 详情页 hero 复用）
- [x] 22. git 提交检查点（已完成：4 个分批功能提交 + quick win 批次提交，工作区干净）

## 大方向（需决策后再排期）

- [ ] 7. 第二阶段 Triton/CUDA 真实实现（variants 框架已就绪，当前仅置灰占位）
  - 卡点：官方 Triton 无 macOS wheel，需先二选一
  - 方案 A（推荐）：OrbStack + linux/arm64 容器跑官方 triton CPU backend
  - 方案 B：triton 源码编译到 macOS（有历史失败记录，耗时不可控）
  - 待做：为 5 道算子题补 `variants/triton/`、`variants/cuda/` 三件套并把 meta.json 的 available 改 true，评测链路容器化
- [x] 8. PyTorch 题库扩充（已完成：随本轮题库重构重建为框架使用 6 题——tensor-shape-indexing / broadcast-ops / autograd-basics / nn-module-block / optimizer-training-loop / dataset-dataloader）
