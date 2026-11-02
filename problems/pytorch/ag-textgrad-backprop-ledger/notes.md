# notes（ag-textgrad-backprop-ledger）

## 回源

- 论文：TextGrad（arXiv 2406.07496）。§2 给出框架（variables / operations / textual
  gradients / "sum" 表示多条反馈的合并），Fig 2 画出 `backward()` 的流程图。论文对**遍历
  顺序、归约分组、多条反馈如何相加**都只给概念，没有给出可执行口径。
- 代码：`zou-group/textgrad` main（HEAD 即论文官方实现）。
  - `textgrad/variable.py`：11-80 构造（`predecessors = set(...)`、requires_grad 校验、
    值类型断言）、151-190 `backward()`（后序拓扑 + 逆序处理 + 只清根）、282-322
    `_check_and_reduce_gradients`、325-367 `_backward_idempotent`（`__add__` 的后向）。
  - `textgrad/autograd/algebra.py`：34-101 `Sum`、104-157 `Aggregate`。
  - `textgrad/autograd/llm_ops.py`：36-98 `LLMCall.forward` / `backward` 的 base-vs-chain
    分派、100-160 `_backward_through_llm_chain`。
  - `textgrad/autograd/reduce_prompts.py`：1-17 `REDUCE_MEAN_SYSTEM_PROMPT` 与
    `construct_reduce_prompt`（`<FEEDBACK>` 包一条、换行连接）。
- 本题只搬图结构与账：LLM 生成 → `grads` 表，总结生成 → `reduces` 表，其余全是遍历与累积。

## 论文口径

1. **反馈即梯度**：每条梯度是一段自然语言，不是向量；因此「求和」是拼接、「均值」是让
   另一个 LLM 总结。
2. **链式法则靠模板**：上游节点的梯度文本会被塞进一个固定模板再交给 LLM 评估下游变量。
   本题把这个模板的**字面**钉住（`Here is the combined feedback we got for this specific
   ... and other variables: ... .`），因为它决定了所有中间节点拿到的梯度文本。
3. **变量与操作分家**：`Variable` 只存值、角色、前驱、梯度；操作在调用时建节点并挂
   `grad_fn`。所以「图」是前向调用顺序的副产物，没有显式的图对象。

## 官方实现里的坑

1. **只清根节点的梯度**：`backward()` 里只有 `self.gradients = set()`，其余节点保留上一轮
   残留。连续两次 `backward()` 会让中间节点的梯度条数翻倍，归约也会因此再次触发。
2. **逆后序**：`build_topo` 先递归前驱再 append，处理时 `reversed(topo)`。兄弟支路因此是
   **先声明、后处理**，同一个共享输入收到梯度的先后也反过来。
3. **幂等广播双推**：`Sum.backward` / `_backward_idempotent` 对每个前驱 `add` 了两次内容
   完全相同的梯度（第二次没有 `_reduce_meta`）。于是：
   - 一个前驱的 `grad_count` 是 2，`get_gradient_text()` 是同一段文本出现两次；
   - 两条空串拼起来是 `"\n"` 而不是 `""`，**下游因此误判「自己有梯度」走 chain 路**——
     这是本题最隐蔽的连锁 bug，普通用例根本看不出来。
4. **aggregate 只推给最后一个前驱**：`Aggregate.backward` 的 `for` 循环里只赋值不推送，
   循环结束后才用最后的 `variable` 推一条。也就是说 `aggregate([a, b])` 永远只把梯度给 `b`。
5. **aggregate 把标记挂在自己身上**：`forward` 里 `aggregated_variable._reduce_meta = [...]`，
   于是这个节点一旦收到 ≥2 条梯度，会先被**自己**的标记触发归约；而那些梯度通常没带标记，
   分组为空 → 不调总结、直接清空，往下推的是空串。
6. **归约的三道门**：没有标记 / 梯度文本为空 / 只有一条梯度，三者任一成立就原样返回。只有
   过了三道门才分组，且**没标记的梯度在这一步被丢掉**（`new_gradients` 只装每组的结果）。
7. **`id(variables)` 当分组 id**：官方用列表对象的 `id` 给归约分组打标，所以同一个列表对象
   被两次 `aggregate` 复用时，两组会并成一组。本题用 `group` 字段显式表达这件事。
8. **`llm` 的 base/chain 分派看的是「响应自己有没有梯度」**：没有就走 base（直接让 LLM 评估
   该变量），有就把响应梯度当链式反馈带上。两条路都只推**一条**梯度，并且都跳过
   `requires_grad=False` 的前驱（系统提示词通常就是这种）。
9. **构造期就报错**：`requires_grad=False` 的变量不允许有需要梯度的前驱，官方在
   `__init__` 里直接抛异常，不等反传。

## 本题的规范化

官方唯一的不可判点来自三处 `set`（`predecessors`、`gradients`、角色集合）。本题统一
规范成「按加入顺序去重」，并明确它影响的可观测点：

- `order`：`predecessors` 的迭代序决定 DFS 与逆后序；
- `roles`：`sum` / `aggregate` 的角色文本里多个前驱角色的先后；
- `reduce_prompts`：同一组内多条梯度拼接的先后。

除这三处之外没有偏离官方。`group` 字段是 `id(list)` 的显式化：默认等于节点 id，两个
`aggregate` 写同一个 `group` 就复现了「共用同一个列表对象」的分组合并。

## 变异

43 条变异：**41 条实质变异全部 CAUGHT**，2 条判为等价：

- `m-grad-role`（梯度条目的 `role` 从 `feedback to X` 改成 `X`）与
  `m-reduce-role-first`（归约后角色取组内第一条而不是去重连接）：两者的产物都只存在于
  「梯度条目的 role」这一字段，而账本只暴露**值**、条数、提示词与节点角色；`role` 既不参与
  `grad_text`，也不被任何后向读取（后向读的是**节点**的 role）。因此对外行为完全一致。
  600 组随机图（549 组成功建图并反传）逐一对比账本，两者 0 差异。
- `m-reduce-gate-text`（去掉「梯度文本为空就不归约」这道门）首轮 SURVIVED：`len > 1` 时
  `"\n".join` 至少产出一个换行，这道门只对「带标记但**零条**梯度」的节点有效——补
  `check_bare_aggregate_root`（aggregate 当根、自己没梯度）后才把它抓住。

## 查重

- 全库无 TextGrad / 文本梯度 / 自动微分相关题目；`agent-react-loop`、`agent-context-compaction`
  是单角色 ReAct 与上下文压缩，`ag-metagpt-msgpool-ledger` 是跨角色投递，均无「计算图 +
  梯度累积/归约」轴。
- 与 `rs-*`（推理侧打分、投票、PRM 归并）划界：那些题的分数是标量，本题的“梯度”是文本，
  核心是图的遍历与累积语义。
