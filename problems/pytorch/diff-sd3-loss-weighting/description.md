# 训练损失：SD3 的 σ 损失加权与置信度感知交叉熵

## 题目描述

Flow Matching 训练的一步要算两笔与「损失」有关的账：一是按噪声水平 σ 给每个样本的误差配上不同权重（SD3 论文的 σ-sqrt / cosmap 名册）；二是给 token 级分类头额外加一项「只在已经预测正确的 token 上压低熵」的正则项。两者都出自 diffusers 的 `training_utils.py`。

## 规则

第一部分，σ 加权：输入浮点张量 `sigmas`（元素为有限正数）与字符串 `weighting_scheme`。

- `"sigma_sqrt"`：权重为 \(\sigma^{-2}\)，并且**一律转成 float32**（上游在这里有一句 `.float()`），所以传 float64 的 σ 拿回来的是 float32。
- `"cosmap"`：权重为 \(2/(\pi(1-2\sigma+2\sigma^2))\)，**dtype 跟着 `sigmas` 走**。
- 其它任何字符串（包括 `"none"`、空串、大小写不同或拼错的 `"cos_map"`）：返回全 1 的 `ones_like`，保持 dtype，**不报错**。

第二部分，置信度感知损失：`logits` 形状 \((...,V)\)，`labels` 形状 \((...)\) 的整数张量。

- `valid` \(=(\text{labels}\ne\text{ignore\_index})\)；被忽略的位置既不进分子也不进分母。
- `weights` 取 `per_token_weights` 或全 1，统一转成 `logits.dtype`。
- `loss_sft` \(=\sum \text{nll}\cdot\text{weights}\cdot\text{valid}\big/\operatorname{clamp\_min}(\sum\text{weights}\cdot\text{valid},1)\)，其中 nll 是逐 token 的交叉熵。
- `lambda_conf == 0` 时直接返回 `(loss_sft, loss_sft, 0)`，不算熵。
- 否则取 `pred = argmax(logits)`、`correct = valid ∧ (pred == labels)`，
  \(p=\operatorname{softmax}(\text{logits.float}()/\text{temperature})\)（**温度只进熵，不进 CE**），
  \(H=-\sum p\log\operatorname{clamp\_min}(p,\text{tiny})\)，
  `loss_conf` \(=\sum H\cdot\text{weights}\cdot\text{correct}\big/\operatorname{clamp\_min}(\sum\text{weights}\cdot\text{correct},1)\)，
  `loss` \(=\) `loss_sft` \(+\) `lambda_conf` \(\cdot\) `loss_conf`。
- 两个分母都过 `clamp_min(1)`：权重整体小于 1 时结果不是加权平均，而是被 1 截断后的求和。

返回 `(weighting, loss, loss_sft, loss_conf)`：`weighting` 形状同 `sigmas`；后三个是 0 维张量、dtype 与 `logits` 相同。

## 示例

`sigmas = [0.1, 0.5, 1.0]`（float32）：

```
sigma_sqrt -> [99.999992, 4.000000, 1.000000]   # 0.1 的 float32 倒数平方不是整 100
cosmap     -> [0.776366, 1.273239, 0.636620]    # 1.273239 = 4/π，在 σ=0.5 取最大
none       -> [1.000000, 1.000000, 1.000000]
```

`logits` 三行 `[[1,3,0], [0,1,2], [2,0,1]]`、`labels = [1, -100, 1]`、`lambda_conf=0.5`、`temperature=2`：

```
loss = 1.741706, loss_sft = 1.288726, loss_conf = 0.905959
lambda_conf=0 -> loss = loss_sft = 1.288726, loss_conf = 0.000000
```

第二个位置被 `ignore_index` 排除；第三个位置的标签是 1 而 argmax 是 0，只进 CE、不进置信度项——后者只看命中的第一个位置。

## 约束

- `sigmas` 必须是浮点张量，非空、元素全部有限且为正；σ=0 会让 `sigma_sqrt` 爆成 inf，按非法输入处理。
- `logits` 至少 2 维、浮点、最后一维非空；`labels` 必须是整数张量且形状等于 `logits.shape[:-1]`。
- `labels` 中不等于 `ignore_index` 的取值必须落在 \([0,V)\)，\(V\) 是 `logits` 的最后一维。
- `temperature > 0`；`lambda_conf` 必须是有限数字；`per_token_weights` 必须是形状同 `labels` 的浮点张量。
- 上述条件不满足一律抛 `ValueError`。
