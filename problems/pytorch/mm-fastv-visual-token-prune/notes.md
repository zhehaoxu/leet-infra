# 解析：FastV 第 K 层视觉 token 剪枝

## 实现思路

先做完整校验，再取 `attention[:, :, -1, image_start:image_start + image_len]` 沿 head 维求均值，得到每行视觉段的分数账本。保留个数用 Python `round(image_len * (1 - r))` 计算，而不是 `int()` 截断，也不是 numpy/torch 的舍入，随后夹进 `[0, image_len]`。

选择用「稳定降序排序」实现：先对分数做一次 `descending=True` 且 `stable=True` 的排序，并列分数因此保持原始升序下标，再取前 `k` 个，最后对选中的下标重新升序排序并加上 `image_start`。前缀与尾部分别用 `torch.arange` 生成并 `expand` 到 batch，三段拼接成 `keep_indexs`。

剪枝后的特征用 `gather` 沿 token 维收集，`position_ids` 直接是 `keep_indexs` 的副本：本题的空洞是原样保留的位置编号，而不是压缩后的连续编号。全程不调用 `topk`，也不修改任何输入。

## 与官方代码的刻意差异

官方实现只有一行剪枝逻辑，但带有四处工程不确定性。本题把它们固定成可判定的规则，而不是静默复现：

- **顺序**。官方 `torch.topk` 之后，幸存视觉 token 按分数**降序**排在 `keep_indexs` 里。本题把它们重新按**下标升序**发出，保证 token 与位置编号仍是因果顺序；对应地，`hidden_states` 的收集顺序也跟随升序下标。
- **并列**。`torch.topk` 在相等分数上的取舍由后端决定，未定义。本题显式规定「分数降序、下标升序」，并列时保留更靠前的 token。
- **批量**。官方代码写了 `[0]` 下标，只对单条序列生效。本题把同一条规则逐行应用到 batch 上，每行独立选择自己的视觉 token，但共享同一个 `k`，所以各行输出长度一致；`keep_indexs` 和 `position_ids` 都带 batch 维。
- **打分口径**。论文正文描述为「所有层、所有 token 的平均注意力」，官方代码只用第 K 层注意力矩阵的**末行**并对 head 取均值。本题实现**代码**的口径，并要求非末行不影响结果；层 K 的位置由调用方通过传入哪一层的注意力来体现。

另外两点属于本题新增的可观察量，而不是上游行为：`score` 张量作为逐 token 分数账本返回，便于判定；`keep_indexs` 与 `position_ids` 同时返回，用于检查实现是否偷偷重编号。

## 独立验证

oracle 不复用参考解的张量排序与 `mean`。它把两个输入 `tolist()` 后，用标量 for 循环逐 head 累加末行列元素、再除以 head 数得到分数；选择用 `sorted(range, key=lambda i: (-score[i], i))` 手工完成，输出下标另用一次 `sorted` 恢复升序，最后用 Python 列表推导按行挑出保留的 embedding 并转成张量。随机用例与 oracle 逐元素对拍，整数输出要求 `torch.equal` 精确一致。

16 个 `check_*` 覆盖两个实跑样例、half-to-even 边界表（`round(1.5)=2`、`round(2.5)=2`、`round(0.5)=0`）、`r=0` 恒等、`k=0` 全剪、前缀尾部在高 `r` 下存活、并列钉死下标、跳号未被重新编号、只有末行与 head 均值参与、保留行是原始行的逐字节拷贝、升序重发不等于分数顺序、float32/float64、非连续视图、输入不变性、十条随机形状扫描以及全部非法输入（`r` 越界或非实数、bool 冒充 int、形状不方、head 为 0、dtype 不一致、非有限值等）。

## 复杂度

记 `L' = L - (image_len - k)`。打分为 `O(B H image_len)`，排序为 `O(B image_len log image_len)`，收集 embedding 为 `O(B L' D)`，合计时间 `O(B (H image_len + image_len log image_len + L' D))`；除输出外只需 `O(B image_len)` 的分数与 `O(B L')` 的下标中间量。相比在第 K 层之后继续跑满 `L` 个 token 的注意力，本题省下的是 `(image_len - k)` 个 token 的后续计算，这正是 FastV 的加速来源。

## 来源与范围

依据已回源的 FastV 官方仓库 `pkunlp-icler/FastV`，文件 `src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py` 中 `FASTV_prune and layer_idx == FASTV_k` 分支：head 均值、末行取值、`round(image_token_length * (1 - r))`、只裁视觉段、`keep_indexs` 拼接以及 `position_ids = keep_indexs.unsqueeze(0)` 的跳号语义均来自该处。README 默认 `K=3, r=0.75`，论文主打配置是 `K=2, R=50%`，本题不绑定任何一组默认值，`r` 与视觉段位置全部由参数给出。

本题不含 LM head、KV cache、模型前向、HF `generate` 的 mask 调整，也不判断哪一层是 K；注意力矩阵必须是已 softmax 的结果，由调用方保证。它是一个剪枝索引步骤的确定性复刻，不是完整的 FastV 推理系统。

```text
.venv-backend/bin/python scripts/add_problem.py --selftest mm-fastv-visual-token-prune --mode submit
```
