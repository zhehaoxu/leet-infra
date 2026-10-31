# 多模态：LLaVA-PruMerge 的 IQR 自适应 token 数

## 任务

LLaVA-PruMerge（Shang et al., arXiv:2403.15388）在 CLIP 视觉塔内部做 token 缩减。姊妹题（order 536）把「保留多少 token」钉成显式的 `reduction_ratio` 入参；本题补上另一半——**官方的自适应比率**：用 Tukey 1.5 倍 IQR 离群规则在注意力值上数离群点，得到保留比率，再据此选出关键 patch。

本题只考「自适应比率 + 选择」这两步：patch 特征、真实前向与合并都不在此处；`cls_attn` 已由调用方在 head 维平均完毕。参数、形状与异常约定统一见 starter 的双语契约，以下只定义计算规则。

## 规则

### 步骤一：自适应比率（上游 `outlier_dectection` 的逐行转写）

1. 把**整个 `[B,N]`** 摊平成一条序列——比率是跨 batch 共享的，不是逐行算的；并先把序列 **cast 成 float32**（float64 输入也在这一步被降精度，这是上游行为点，测试有专门的压线用例）。
2. `Q1` = 第 25 百分位、`Q3` = 第 75 百分位，都用 numpy 默认的 **linear 插值**口径（虚位 `q/100*(n-1)` 线性内插）。
3. `IQR = Q3 - Q1`，上界 `upper = Q3 + 1.5 * IQR`（Tukey 1.5 倍，只看上尾；上游被注释掉的下界不出题）。
4. `ratio = 严格大于 upper 的元素个数 / 序列总长`，一个 Python float。

### 步骤二：选择（与 order 536 同口径）

- 保留个数 `k = int(N * ratio)`，**向下截断**。
- **与 536 的关键分工**：`k == 0` 在这里是**合法结果**——上游自适应路径允许一个 token 都不保留（`torch.topk(..., 0)` 返回空集），此时 `kept` 是 `[B,0]` 空张量、`pruned` 是完整的 `0..N-1`；536 在同样情形抛 `ValueError`。
- `kept`：每行 `cls_attn` 最大的 `k` 个下标，并列按「值降序、下标升序」判定，最终**按升序下标**发出；`pruned` = 其余下标，同样升序。

### 数学保证（测试断言）

任何有限序列中至少有一个元素落在 `Q1` 附近、不严格大于 `upper`，因此 `ratio < 1` 恒成立、`k ≤ N-1`、`pruned` 永不为空。

## 样例 1：全相同值 → 比率为零

`B=2, N=5`，全部元素 `0.1`（float32）：

```text
ratio = 0.0
kept_indexs   = shape [2, 0]（空）
pruned_indexs = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
```

全部相等时 `Q1 = Q3`、`IQR = 0`、`upper = Q3`，而「严格大于」容不下任何等于上界的元素。

## 样例 2：单个离群点

`B=1, N=8`（float64），`cls_attn = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.50]`：

```text
ratio = 0.125            # 1 / 8
kept_indexs   = [[7]]
pruned_indexs = [[0, 1, 2, 3, 4, 5, 6]]
```

`Q1 ≈ 0.0275`、`Q3 ≈ 0.0625`、`upper ≈ 0.115`，只有 `0.50` 越界；`k = int(8 * 0.125) = 1`。

## 样例 3：比率摊平在 batch 上

`B=2, N=4`（float32），行 1 全 `0.1`、行 2 是 `[0.1, 0.1, 0.1, 0.9]`：

```text
ratio = 0.125            # 8 个值里恰有 1 个越界，两行共享同一个比率
kept_indexs   = shape [2, 0]   # k = int(4 * 0.125) = 0，两行都不保留
pruned_indexs = [[0, 1, 2, 3], [0, 1, 2, 3]]
```

若把比率逐行算，行 2 会得到 `0.25`、`k = 1`——这是逐行变异被测试抓住的对照样例。

## 判题

10 个 `check_*`：手算锚点样例（全零比率、单离群、摊平共享、并列选择）、`k=0` 空选、float32/float64 输入的一致性（cast 在先）、float32 cast 的压线用例（`nextafter` 步长在 float64 下是离群、cast 后归并）、非连续视图与输入不变性、混合分布随机用例与独立 oracle 逐元素对拍、全部非法输入。oracle 不复用任何 numpy/torch 的分位与选择：percentile 用纯 Python 的 float32 逐步算术 + numpy 双路 lerp 手写（已对 `np.percentile` 做 4 万组位级对拍零差异），选择用标量循环。

```text
.venv-eval/bin/python scripts/add_problem.py --selftest mm-prumerge-iqr-adaptive --mode submit
```
