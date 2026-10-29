# 参考做法：分组选择、偏置隔离与符号更新

## 实现

1. `scores = sigmoid(x @ gate_w.T)`，`choice = scores + bias`。
2. 把 choice 按连续等长组 reshape，每组取最大的两个值相加作为组得分。
3. 组得分降序取 `top_groups` 个组，稳定性排序保证并列取小组号。
4. 用组掩码把未选组的 choice 置为 `-inf`，再降序取 k 个专家。
5. 权重只取**原始 scores**，在选中项内归一化；bias 不进入权重。
6. `bincount` 统计整个 batch 的负载；若更新，则按 `sign(平均负载 - 实际负载)` 调整 bias。

偏置更新是整 batch 一次，不是逐 token；平均负载为 $Tk/E$。`update_bias=False` 时用 `clone` 返回独立张量，避免调用方修改输入。

## 正确性检验

14 个 `check_*` 覆盖：手算示例与偏置更新、偏置只影响选择、组选择用 top-2 和、全组选择退化为普通 top-k、batch 全局负载更新、不更新时返回副本、gamma=0、组与专家并列顺序、k 边界、饱和输入、非连续输入、36 组随机枚举、结构错误、最大形状。

oracle 用纯 Python 逐 token 计算 sigmoid、分组、掩码、选择、归一化和计数，独立于 torch 的 `topk`/`argsort`/`scatter`。手算示例中过载专家偏置下降、欠载专家上升，能捕捉符号写反；偏置示例能捕捉把 bias 混进权重的错误。

## 范围与来源

机制对应 DeepSeek-V3 技术报告中描述的 auxiliary-loss-free 负载均衡：sigmoid 门控、逐专家偏置参与选择、分组 top-k，以及按负载用固定步长调整偏置。本题把路由前向与一次 batch 级更新写成显式、可复现的规则。

题目不声称与任何训练框架的调度完全一致：真实训练中偏置在优化步边界更新、负载统计口径和分布式归约方式可能不同；本题不涉及梯度、优化器、容量约束或 all-to-all 通信。并列顺序、组大小和更新符号均由题面固定，todos 中提到的其他机制（死专家账本、soft-clamp、expert dropout）未在本批实现，也未在此题中混入。
