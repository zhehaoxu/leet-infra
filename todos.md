# leet-infra 待办事项

## 小改进（quick wins）

- [x] 1. 代码草稿持久化：每题编辑器内容存 localStorage，刷新/切页自动恢复；「重置代码」清除草稿（已完成，浏览器端验证通过）
- [ ] 2. git 初始化 + 首次提交（.gitignore 已就绪）
- [ ] 3. 题库热加载：problems.py 的 _INDEX 缓存按目录 mtime 失效，新增题目免重启后端

## 产品向增强

- [x] 4. 运行 / 提交分离：「运行」只跑样例用例，「提交」全量评测并标记 AC / 解锁参考答案（已完成，含提交 AC 彩花特效，浏览器端验证通过）
- [ ] 5. 做题页上一题 / 下一题导航，AC 后可自动跳转
- [ ] 6. 耗时对比展示：你的耗时 vs 参考解基准（后端缓存基准值），仅展示不设门槛

## 大方向（需决策）

- [ ] 7. 第二阶段 Triton 题库
  - 卡点：官方 Triton 无 macOS wheel，需先二选一
  - 方案 A（推荐）：OrbStack + linux/arm64 容器跑官方 triton CPU backend
  - 方案 B：triton 源码编译到 macOS（有历史失败记录，耗时不可控）
- [ ] 8. PyTorch 题库扩充：autograd / LLM 组件（KV-cache、top-p 采样）/ 内存布局陷阱，方向待确认
