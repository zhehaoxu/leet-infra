# 详解：autograd 血缘账——叶、requires_grad 与反向图

## 出处

教程侧是官方 *Autograd*（`docs/source/beginner/basics/autogradqs_tutorial.py`）与 *Autograd mechanics*（`docs/source/notes/autograd.rst`）：前者讲「叶、`requires_grad`、`.grad` 累加、`no_grad`」，后者补「`inference_mode` 与 `no_grad` 的区别、叶子/非叶子、就地操作的报错」。两边都只给结论清单，逐语句的元数据变化（借标志的视图、`detach_` 对 inference 张量失效、被释放的图哪些还能再反）得看实现或动手测。

实现侧按 torch 2.14.1 实测口径复现（与 `solution.py` 顶部列的是同一批位置）：

- `torch/csrc/autograd/variable.cpp` 的 `Variable::set_history`：只有 grad mode 且操作数里存在「非 inference 的 requires_grad 张量」时才挂 `grad_fn`，这是规则 1/2 建图条件的来源。
- `aten/src/ATen/native/TensorViews.cpp`：视图共享 `_base` 与基张量的 `requires_grad`（`grad mode` 不参与决策），`detach` 新建张量并切断 `_base`。
- `torch/csrc/autograd/engine.cpp` 的 `run_backward`：隐式梯度只接受标量输出，按 `retain_graph` 决定是否释放走过的节点，`accumulate_grad.cpp` 的 `AccumulateGrad` 负责分配/累加 `.grad`。
- `VariableType` 生成的就地包装里的 `check_inplace`：两条报错 —— `a leaf Variable that requires grad is being used in an in-place operation` 与 `a view of a leaf Variable that requires grad...`；`Tensor::detach_` 对视图抛 `Can't detach views in-place`。
- `Tensor::requires_grad_` 对 inference 张量置真是先于「同值空操作」的硬性拒绝（`Setting requires_grad=True on inference tensor outside InferenceMode is not allowed`）。

## 核心与易错点

1. **inference 张量不传播 `requires_grad`**：`inference 张量 + 普通叶` 的结果由图外的普通叶驱动建图，`inference + inference` 结果干脆不进图；而「inference 且要梯度」的张量当**操作数**会触发 `Inference tensors cannot be saved for backward`，属于本题保证里排除的组合。
2. **视图分两种，这是全题最容易翻车的一处**：在求导态、根要梯度时切出来的视图会拿到 `TBackward0`/`SelectBackward0`/`UnsqueezeBackward0`，它**自带元数据**，之后根 `detach_` 关标志（链 4 第 2 行）它读到的仍是 `rg=1`；反之在 `no_grad`/`inference_mode` 里、或根当时不要梯度时切出来的视图**没有元数据**，`is_leaf` 为真、标志是**实时**读根的（根点亮它也点亮、根关掉它跟着关）。`check_borrowed_view_flag` 与 `check_view_flag_and_detach` 各钉住其中一半。
3. **借标志的视图能建图却不能累加**：`mul(borrowed, borrowed)` 在求导态下 `rg` 为真、照挂 `MulBackward0`，但两条入边都是空的——它没有 `AccumulateGrad` 宿主。于是这样的图反向**成功却不给任何张量分配缓冲**；拿它自己当反向起点（标量视图）则直接失败，因为引擎既找不到节点也没有地方累加。
4. **`detach_` 对 inference 张量是空操作**：它不是新建张量、而是复位 autograd meta；inference 张量根本没有可复位的 meta，标志保持原值。想关掉 inference 张量的标志只能靠 `requires_grad_(False)`。
5. **`requires_grad_` 的判定顺序**：先查 inference 置真（一律拒绝，哪怕当前已是真）→ 再查「值相同」（静默空操作，非叶、视图都合法）→ 最后查「不同值只在叶上」（非叶失败）。链 6 第 8 行正是「非叶 + 同值」的成功空操作。
6. **释放不是「图作废」而是「缓冲作废」**：一次不带 `retain_graph` 的反向会释放走过的节点，但只有反向时要回读保存张量的节点（mul、sigmoid、relu）才会在第二次反向时抛 `Trying to backward through the graph a second time`；add/neg/sum/mean/clone 与三种视图的 backward 不读回数据，释放对它们毫无影响。`check_non_saving_nodes_replay` 用「叶→sum」连反三次（含不带 retain 的第二次）钉住这一点，`check_backward_graph_consumption` 用共享 `MulBackward0` 钉住另一半。
7. **梯度缓冲的身份**：`grad_id` 记「第一次分配它的那条 `backward` 语句」；同一块缓冲被反复累加时 `grad_id` 不变；`grad_none` 之后的下一次反向会**重新分配**（id 换成新语句下标）；`grad_zero_` 只清零，身份不变。链 2 把这四个阶段串成一条链。
8. **`retain_grad` 只影响「中间张量也留缓冲」**：叶本来就会留；对当前不要梯度的张量调用直接失败。注意「留了标记」不等于「一定拿得到」：标记之后的那次反向若因图被释放而失败，这个中间张量的 `.grad` 永远是空（`check_diamond_allocates_one_buffer` 第 9 行）。
9. **`add_` 的入边包含主体此前的节点**：所以就地写之前通过旧节点连上的叶，在新的图上依然会被反向到（链 10 第 6 行第二个叶拿到梯度）；同时主体的 `grad_fn` 被替换，`probe` 读到的是 `AddBackward0`。
10. **标量叶可以直接反向**：没有 `grad_fn` 但自己持有元数据的标量（普通叶、或 `requires_grad_` 点亮的标量缓冲宿主）反向成功并给自己分配缓冲；借标志的标量视图则失败。两者在观测行里只差 `ok`，容易以为「没 `grad_fn` 就一律不能反」。

## 定义域外的真实行为（刻意排除）

- **在借标志的视图上改 `requires_grad_`**：torch 会在这时给它临时造一份元数据，视图与根从此分道扬镳。行为依赖内部时机，本题直接用程序保证排除（`requires_grad_` 的主体一律不是视图）。
- **「inference 且要梯度」的张量当操作数**：会触发 `Inference tensors cannot be saved for backward`，与「建图/不建图」的二分法冲突，排除。
- **非标量 `backward()`**：不传 `gradient` 的隐式梯度只对标量合法，非标量会报 `grad can be implicitly created only for scalar outputs`。本题只按 `rank > 0` 判失败，不考显式 `gradient` 入参。
- **半途而废的反向**：真实引擎在多分支图上可能先给某个叶分配了缓冲、再撞上被释放的节点而失败。这会让「失败 ⇒ 状态不变」不再成立，本题用 oracle 的 `partial backward` 守卫把它挡在生成器之外，并在程序保证里写明。
- **更高阶**：`create_graph=True`、对 `grad` 再求导、`register_hook`、`torch.autograd.grad`、`functional_call` 等都不在范围内；`torch.compile` 与任何 CUDA/device 分支按全集红线不出。
- 操作集里没有 `exp`/`mm`/`sub`/`div`：只挑了能覆盖「建图 / 不建图 / 要回读保存张量 / 不要回读」四类的代表，避免把题目变成记住一堆类名。

## 对拍口径

solution 是纯 Python 的元数据推演，不 import torch。`test_cases.py` 拿 torch 当 oracle：

- `Runner` 逐语句执行真实 torch，`ts[i]` 保存第 `i` 条语句创建的张量；观测行的 `view_of` 用 `id(t._base)` 反查语句下标，`grad_id` 用 `id(t.grad)` 反查分配它的 `backward` 语句下标；`keep` 列表兜住引用，防止 `id` 复用把身份判据带歪。
- 观测只读五个公开接口（`requires_grad`/`is_leaf`/`torch.is_inference`/`dim`/`grad_fn`/`_base`/`grad`），不碰任何数值，因此 torch 版本间只要语义不变结论就稳定。
- 生成器先造候选语句再交给真实 torch 试跑：抛出的错若不在白名单（`EXPECTED_ERRORS`），说明撞上了本题排除的组合，整条语句作废（`OutOfDomain`）；白名单内的错再检查「是否半途分配过缓冲」，有则同样作废。
- 随机对拍：8 个固定种子 × 40 条程序、每条 6–26 条语句，共 5182 条语句（其中 754 条是被拒绝的失败行），solution 与 torch 逐行全等，0 处分歧；随机源全部是 `random.Random(seed)`，没有全局状态污染。
- 题面 6 条示例链由参考实现与 torch 双向实测得出；另有 13 组专项检查（图消费、可重放、菱形缓冲、就地改写、三区域与血缘、叶标志与缓冲、借标志视图、视图与 `detach_`、`requires_grad_`/`retain_grad`/标量叶、inference 与 `detach_`、非法输入）逐条回灌 torch 复核。

## 复杂度

单条语句的建图/反向遍历是 O(图规模)，整段程序 O(语句数 × 图规模)；本题语句数 ≤ 40，全量对拍（近 5200 条语句）在本机 1 秒内完成。
