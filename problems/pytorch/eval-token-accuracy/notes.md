# 详解：忽略位的 token 准确率

## 出处与工程背景
- token-level accuracy 是 `F.cross_entropy` 每位置 argmax 命中率的直接可读版，也是 `trax.metrics`, `flax` / HF `evaluate` 里 `accuracy` 的默认写法；对话模型（ChatML / Llama-style SFT）的评测脚本几乎都会顺手打印它。
- 有效位的来源是数据整理与掩码题：`data-collator` 把 pad 位的 labels 写成 `ignore_index`，`sft-assistant-mask` / `sft-turn-loss-weight` 把 prompt 段与多轮模板写成 `ignore_index`。本题是这些掩码的**消费端**——掩码错了，指标看起来仍然"正常"，所以必须自己数分母。
- 与 `eval-perplexity-logits` 的分工：那题从概率出发（连续、可比较、对置信度敏感），这题从 argmax 出发（离散、只看对错）。同一份模型，PPL 变好而 token acc 不动是很常见的（概率质量 redistribution）。

## 记号与推导
设 \(\mathcal V=\{(b,i): y_{b,i}\ne t\}\) 为有效位置集合（\(t=\) `ignore_index`），\(h_{b,i}=\mathbb 1[\hat y_{b,i}=y_{b,i}]\)。
\[ \text{micro}=\frac{\sum_{(b,i)\in\mathcal V} h_{b,i}}{|\mathcal V|}, \qquad
\text{macro}=\frac{1}{|\mathcal B|}\sum_{b\in\mathcal B}\frac{\sum_{i:(b,i)\in\mathcal V} h_{b,i}}{|\mathcal V_b|}, \quad \mathcal B=\{b:|\mathcal V_b|>0\} \]
两者关系：\(\text{micro}\) 是\(|\mathcal V_b|\) 为权的加权平均，
\[ \text{micro}=\sum_{b\in\mathcal B}\frac{|\mathcal V_b|}{|\mathcal V|}\,\text{acc}_b, \qquad \sum_b \frac{|\mathcal V_b|}{|\mathcal V|}=1 . \]
所以只有当所有 \(\text{acc}_b\) 相等（或所有 \(|\mathcal V_b|\) 相等）时 micro 与 macro 才重合；一般地
\(|\text{micro}-\text{macro}|\) 大致量级是「行内 acc 与有效位数的协方差」：长且准的样本把 micro 抬高，长且差的样本把 micro 拉低。

## 核心与易错点
1. **不要移位**：本题的 `pred_ids` 已与 `labels` 逐位对齐（移位是上游的事——对 LM logits 取 argmax 时就已经配过对了）。再移一次会把整行错位、指标看着像 0。
2. **mask 必须是 labels 侧的**：常见错法是用 `pred_ids != -100` 当 mask——`pred_ids` 从不会是 \(-100\)（它是词表 id），于是分母变成 \(B\cdot L\)，pad 位因"pred != label"被判为错，指标被稀释。
3. **\(-100\) 与 \(-100\) 相等是真陷阱**：mask 必须**同时**作用在分子与分母上。若写成 `hit = pred == labels` 而漏掉 `& valid`，那么当 `pred_ids` 在忽略位上也填了 \(-100\)（有些实现会直接复制 labels）时，`-100 == -100` 会白送命中；若漏掉的是分母，pad 位又被当成预测错误，两头都不对。
4. **macro 的分母是"有有效位的样本数"**：全 prompt 的样本必须连样本数都不进，否则它的 0/0 被当 0 计入，把 macro 拉低。这与 `to-grpo-advantage` 里"空组不产生 advantage"是同一种处理。
5. **micro 不是 macro 的别名**：短回答被长回答稀释，是 batch 内样本数少时的常见误读；报数时要写清口径，跨实验比较必须固定同一口径。

## 哨兵约定
空集返回 \((0.0, 0.0)\)，与 HuggingFace `evaluate` 的 `accuracy` 对空输入返回 0.0 的口径一致；好处是永远落在值域 \([0,1]\) 内、不会让下游绘图崩掉，代价是与"全错"重合，因此**判题用例同时覆盖 `all_wrong` 与 `no_valid`**，实现时不能靠返回值区分，只能靠 valid 计数区分（`total_valid == 0` 分支）。

## 对拍口径
参考解用张量归约：`valid = labels != ignore_index` → `hit = (pred == labels) & valid` → 全局 `sum` 相除得 micro；行内 `sum(dim=1)` 后用 `per_valid > 0` 布尔索引再 `.mean()`（走 double 避免 float32 平均误差）得 macro。判题 `_oracle` 是纯 Python 双循环逐元素比较与累加、并按行收集 acc（结构完全不同、数值等价），逐用例用 `allclose(atol=1e-5, rtol=1e-5)`（在 float64 张量上）比较。另有 5 枚退化/定向断言：空集、完美预测、等有效位数时 micro==macro、不等有效位数时两者分离（0.4 vs 0.625 手算值）、忽略位上写什么都不得改变结果。

## 复杂度
\(O(BL)\) 时间、\(O(B)\) 额外空间（行内计数）。

## 实践提醒
对话数据里"有效位"往往只占整条样本的一小段（prompt 可占 80% 以上），此时 macro 与 micro 的差异可达数个百分；若要按轮次加权，请改用 `sft-turn-loss-weight` 那类权重再算加权 acc，而不是自己改分母。
