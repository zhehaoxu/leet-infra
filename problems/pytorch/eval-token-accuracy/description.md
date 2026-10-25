# 评测：忽略位的 token 准确率

## 题目描述

指令微调 / 对话模型的报告里最常用的一个数字就是 token 级准确率：模型预测的 token 有多少比例落在真正要监督的位置上。**关键在于分母**——prompt、system 模板和右侧 pad 上都写着 `ignore_index`，这些位置一个都不许进分母；而"每条样本先算自己的准确率、再对样本取平均"（macro）与"把全部有效 token 倒进一个池子算"（micro）在样本长短不一时**是两个不同的数**。同一份预测，报哪个口径必须能自己算出来。

## 规则

- `pred_ids`、`labels`：int64，形状都是 \((B, L)\)，**已经逐位对齐**（`pred_ids[b, i]` 就是用来判 `labels[b, i]` 的那个预测；本题不再做右移）。
- `labels[b, i] == ignore_index`（默认 \(-100\)）的位置是 prompt / pad，既不进分子也不进分母；这些位置上 `pred_ids` 写什么都无所谓，即使恰好也等于 \(-100\)。
- 命中定义：\(\text{pred\_ids}[b,i] = \text{labels}[b,i]\) 且该位有效。
- 返回二元组 `(micro, macro)`（都是 Python float）：
  - \(\text{micro} = \dfrac{\sum_{b,i}\text{hit}}{\sum_{b,i}\text{valid}}\)——先跨 batch 求和再相除。
  - \(\text{macro} = \dfrac{1}{|\mathcal B|}\sum_{b\in\mathcal B}\dfrac{\text{hit}_b}{\text{valid}_b}\)，其中 \(\mathcal B\) 是**有效位数大于 0** 的样本集合；纯 prompt / 纯 pad 的样本连样本数都不进。
- 整批有效位数为 \(0\) 时按约定返回 \((0.0, 0.0)\)（与 HuggingFace `evaluate` 的空集口径一致）。
- 禁止调用现成评测库。

## 示例

```
pred_ids = [[7, 7, 1, 2, 9],      labels = [[-100, -100, 1, 2, 3],
            [4, 6, 1, 1, 1]]                [4, 5, -100, -100, -100]]

行0：有效位 3 个（id 1,2,3）；预测 1,2,9 -> 命中 2 个 -> 行内 acc = 2/3
行1：有效位 2 个（id 4,5）；  预测 4,6   -> 命中 1 个 -> 行内 acc = 1/2

micro = (2 + 1) / (3 + 2)              = 0.6
macro = (2/3 + 1/2) / 2 = 0.5833333333
```

两个数不相等：长句（行0）在 micro 里权重更大，而 macro 给长短句各一半权重。若每个样本的有效位数相同，两者必然重合。

## 约束

- CPU int64 输入；\(B, L \ge 1\)。
- 两个返回值都落在 \([0, 1]\)。
- **退化性质**：全 `ignore_index` \(\Rightarrow (0.0, 0.0)\)；`pred_ids` 恰为 `labels` \(\Rightarrow (1.0, 1.0)\)；各样本有效位数相同 \(\Rightarrow \text{micro} = \text{macro}\)；只有一条样本含有效位 \(\Rightarrow\) 两者也相等。
- 分母是**有效位数**而不是 \(B\cdot L\)；把忽略位当成「预测错」会系统性压低指标，当成「预测对」（两边都写着 -100，比较恰好相等）则会虚高。
