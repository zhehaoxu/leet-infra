# PEFT：GenKnowSub 通用知识减除

## 题目描述

**GenKnowSub**（arXiv:2505.10939，ACL 2025，官方 `saharsamr/Modular-LLM`；PEFT 侧与 Arrow 同文件，`use_gks`/`gks_` 前缀）假设任务 LoRA 里**纠缠着通用知识**，损害零样本泛化：构建一个（或多个）**通用域 LoRA**，把它的 \(A/B\) 因子**取平均后从每个任务适配器中减去**（"forgetting-via-negation" 任务算术，arXiv:2212.04089），得到更纯粹的残差任务模块；再交由 Arrow 路由。减法作用在**因子**上，因此减除后的任务增量 \((B-\bar B)(A-\bar A)\) 带交叉项，**不等于** \(B@A-\bar B@\bar A\)。

本题复刻官方 PEFT `gen_know_sub` + `create_arrow_model` 门控的确定性账：

- **通用平均**：`avg_A = stack(A_gks).mean(0)`、`avg_B = stack(B_gks).mean(0)`（逐元素均值，多个语言 GK LoRA 参与平均）。
- **因子减除**：官方就地 `lora_A[name].weight.data.sub_(avg_A)`（B 同理）；判题域为纯函数——clone 后减。首轮（`gks_done=False`）净化**全部**任务适配器；增量轮只净化新加入的（`gks_added_adapter_names`）。
- **门控状态机**：`use_gks=False` → skip；`use_gks ∧ gks_done ∧ 无新增` → skip；`use_gks ∧ ¬gks_done` → all；`use_gks ∧ gks_done ∧ 有新增` → added。
- **路径校验**：`use_gks=True` 但通用 LoRA 路径为空 → ValueError（"You should provide general LoRA paths if you want to use GenKnowSub."）。

## 任务

- `mode="gks_average"`：g_a、g_b（GK 因子列表）入参，返回 (avg_A (r,in), avg_B (out,r))。
- `mode="gks_subtract"`：a_stack、b_stack、avg_a、avg_b、only（可选索引子集）入参，返回净化后的 (a_new, b_new)；原栈不变。
- `mode="gks_gate"`：use_gks、gks_done、n_added 入参，返回 "skip"/"all"/"added"。
- `mode="require_gks_paths"`：use_gks、n_paths 入参，返回 n_paths；非法组合抛 ValueError。
- `mode="task_delta"`：a_sub、b_sub 入参，返回逐专家 \((out,in)\) 增量 stack。

## 示例

seed 0/1/2，float64：

```
gks_average(g_a [3,2,8]×3, g_b [3,6,2]×3) -> avg_a (2,8), avg_a[0,0] = -0.7495449019423001
gks_subtract(a (4,2,8), b, avg_a, avg_b) -> a - avg_a 逐专家；a_new[0,0,0] = 0.438255347752268
gks_subtract(..., only=[1,3]) -> 仅专家 1/3 被净化，其余逐位不变
gks_gate(True, False, 0) == "all"；gks_gate(True, True, 0) == "skip"；gks_gate(True, True, 2) == "added"
require_gks_paths(True, 0) -> ValueError
task_delta(a_new, b_new)[0,0,0] = 2.1708577088384393
  且 d == B@A - B@avgA - avgB@A + avgB@avgA（交叉项），≠ B@A - avgB@avgA
```

## 约束

- 平均与减除都在**因子**（A/B）上，不在 ΔW 上——交叉项不可省略。
- `gks_subtract` 是纯函数：输入栈必须逐位不变（官方 in-place `data.sub_` 的判题域等效）。
- 门控优先级：`use_gks=False` 恒 skip（覆盖 gks_done/新增）。
- `only` 子集净化时未选中专家**逐位相等**（`torch.equal`），不是近似。
- 任务增量的交叉项展开：\(B\bar A + \bar B A - \bar B\bar A\) 三项全部非零。
