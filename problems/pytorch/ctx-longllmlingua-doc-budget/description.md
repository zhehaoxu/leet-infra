# 上下文压缩：LongLLMLingua 文档预算、动态压缩率与窗口阈值计划

## 题目描述

LongLLMLingua（2310.06839）在 LLMLingua 之前加了三层结构：先按「问题感知 PPL」给文档排序、在 token 预算里挑选文档（document-level），再给每条文档分配不同的保留率（重相关的文档多留、弱相关的少留），最后在 200 token 的滑动窗里按逐 token 的 NLL 阈值删词（token-level）。真实实现里三层的接口各藏着一个容易丢的细节：排序方向由条件类型决定、预算表达式是字符串四则、窗口边界按 `len+2` 记账而不是真实 token 数。本题把文档 PPL 与逐 token NLL 都作显式入参，复现这三层的确定性账目。

## 任务

实现 `solve(doc_token_ids, doc_ppl, token_loss, target_token, context_budget="+100", force_context_ids=None, reorder_context="original", dynamic_context_compression_ratio=0.0, condition_in_question="none", iterative_size=200)`：

- `doc_token_ids`：$B\ge 1$ 条文档，每条是非空 `list[int]`；
- `doc_ppl`：长度 $B$ 的 `list[float]`，每条文档的**问题感知 PPL 分数**（上游由 `get_condition_ppl` 现算，本题显式给出；上游对长度项乘了 `2/250*0`，即长度不参与排序分）；
- `token_loss`：所有文档 token 依序拼接后逐 token 的 NLL（上游 `loss` 按下一 token 预测对齐，本题对齐到 token 本身）；`10000.0` 是哨兵值，只退出阈值统计、不影响保留判定；
- `target_token`：压缩预算（实数）；先做 `target_token < 0 → 100`，再套 `context_budget`；
- `context_budget`：形如 `"+100"`、`"*0.5"` 的四则运算串（`+ - * /` + 无符号数），对预算做一次二元运算；
- `force_context_ids`：无条件保留的文档下标（同样占预算）；
- `reorder_context`：`"original"`（按文档下标升序还原）或 `"two_stage"`（选中序的偶数位在前、奇数位反转接后）；
- `dynamic_context_compression_ratio`：$\lambda \ge 0$（上游取 `abs`）；
- `condition_in_question`：`"none"` 时按 PPL **降序**（无问题时信息量大的文档先进预算），`"question"` 时**升序**（条件 PPL 越低越相关）；排序稳定；
- `iterative_size`：窗口宽 $W \ge 1$。

返回 `dict`，键为 `rank_order`、`selected`、`dynamic_ratios`、`tau`、`window_plan`、`thresholds`、`keep_masks`。

## 精确规则

**① 排序。** `rank_order` = 按 `direction * ppl` 的稳定排序，`direction = -1`（`none`）或 `+1`（`question`）。

**② 预算与选文档。** 先做 `target<0 → 100` 与四则变换得到**预算值**（下称 `budget`；窗口阶段用的就是它，选文档的扣减只决定停止条件）。从 `rank_order` 依次扣减**文档原始 token 数**：`budget -= len(doc)`，把下标并入选集（去重），`budget < 0` 即停。`force_context_ids` 预置入选集。随后按 `reorder_context` 得到输出序 `selected`。

**③ 动态压缩率。** 记选集（还原前、按排序序）长度 $N$：比率表为

$$\Big[i \cdot \tfrac{\lambda}{N-1}\Big]_{i=-(N-1),\,-(N-3),\,\dots,\,N-1}\ \text{再整体反转，}$$

$N=1$ 时为 0；按文档下标映射后**跟随文档**重排到 `selected` 序（不是按位置）。$\lambda = 0$ 时全 0。

**④ 窗口计划。** 输出序下每条文档长度取 $L_i = \text{len} + 2$（上游给 CLS/SEP 留位的口径；预算选文档用的是不带 +2 的原始长度——两处不一样是本题考点）。$\tau = \dfrac{budget}{\sum_i L_i + 1}$，文档保留率 $r_i = \mathrm{clamp}(\tau + \text{dynamic\_ratios}_i,\ 0,\ 1)$。窗口从虚拟 BOS（`last = 1`）起顺序装文档：`last + L >= W` 时以 `delta_end = W - last` 收窗，余量 `last = last + L - W`；余量超过 $W$ 时整除出 $k = \lfloor last/W \rfloor$ 个 `(W, r)` 整窗，剩余为下一窗首段（`last` 归零则不留）；否则 `last += L`。收尾余段照常收窗。

**⑤ 阈值与保留。** 展平后第 $g$ 个 pair（上游 `end` 每 pair 推进 $W$）覆盖块 $[gW,\ gW + \text{delta\_end})$。其阈值：`ratio == 1.0` 直接 `-inf`；否则把 `token_loss` 中 `!= 10000.0` 的值**降序**排序取第 `max(0, min(len-1, int(len*ratio) - 1))` 位。保留掩码：块内 `loss > threshold`（位置 0 恒保留），块外恒保留。注意 `ratio` 是**保留**率——保高 NLL（信息量大）的 token，`r=0` 时块内全删。

## 示例

```text
示例一（预算 + two_stage + 动态比率 + 窗口计划）
doc_token_ids = [[10,11,12], [20,21], [30,31,32,33]]   # 原始长度 3, 2, 4
doc_ppl = [2.0, 8.0, 5.0]，condition = "none" → rank_order = [1, 2, 0]
target = 10，budget "+5" → 15
  依次扣减：doc1 15-2=13 → doc2 13-4=9 → doc0 9-3=6（未耗尽，全选中）
  used（排序序）= [1,2,0]；two_stage：偶数位 [1,0] + 奇数位反转 [2] → selected = [1,0,2]
λ = 0.6，N=3：表 [0.6, 0.0, -0.6]，按下标映射 → dynamic_ratios = [0.6, -0.6, 0.0]
窗口（iterative_size=8）：L = [4, 5, 6]，budget=15 → tau = 15/(15+1) = 0.9375
  r = [1.0, 0.3375, 0.9375]
  last=1：doc1(4) 装入；doc0(5)：5+5>=8 → 收 (3, 0.3375)，余 2；doc2(6)：2+6>=8 → 收 (6, 0.9375)，余 0
  window_plan = [[[4,1.0],[3,0.3375]], [[2,0.3375],[6,0.9375]]]

示例二（阈值与保留，token_loss 无哨兵、共 9 个）
loss = [3.0, 1.0, 4.0, 1.5, 3.5, 0.5, 2.5, 2.0, 2.8]
pair (3, 0.3375)（第 1 块，块 [8,11) 截到 [8,9)）：
  int(9*0.3375)=3 → 取降序第 2 位 = 3.0；位置 8 的 loss=2.8 不大于 3.0 → 删
  keep_masks[1] = [T,T,T,T,T,T,T,T,F]
pair (4, 1.0)：ratio=1.0 → -inf 全留
```

## 约束

- `doc_token_ids` 非空且每条非空；`doc_ppl` 与文档数等长、均为有限实数。
- `token_loss` 长度等于全部 token 数之和；`10000.0` 为哨兵。
- `context_budget` 必须匹配 `^[+\-*/]\d+(\.\d+)?$`；`reorder_context`、`condition_in_question` 只接受列出的取值。
- `force_context_ids` 的下标必须在 `[0, B)` 内；`iterative_size` 为 `>= 1` 的整数。
- 不修改任何输入；所有非法输入抛 `ValueError`。

## 与相邻题目的边界

- `ctx-llmlingua2-word-compress` 是 LLMLingua-2 的**词级分类后处理**（BERT 打分、无文档结构）；本题是 LongLLMLingua 的**文档级预算 + 窗口级阈值计划**（LLaMA 打分、文档排序入参）。
- `distributed-cp-zigzag-partition` / `distributed-cp-varlen-zigzag-plan` 是 CP 的**静态布局**（因果负载均衡）；本题是压缩预算的**动态分配**（按 PPL 与窗口 NLL）。
