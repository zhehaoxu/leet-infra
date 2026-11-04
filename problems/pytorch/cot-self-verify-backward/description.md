# Self-Verification 后向验证

## 题意

Self-Consistency 投票只数答案出现次数，Self-Verification（arXiv:2212.09561）更进一步：把每个候选答案**回填**进问题做后向验证——把问题里的数字逐个挖空成 `'X'`，让 verifier 判断「当前候选答案是否还能与上下文自洽」，与挖空前原数字一致者加分，总分最高的候选胜出。

给你（GPT 采样、answer_cleansing 与 verifier 模型都在池外）：`question`（问题原文）、`sampled_answers`（k 份采样的清洗后答案串）、`verifier_outputs`（扁平判定串池）、`ground_truth`、可选缓存 `declarative`。

## 你要实现的账本

### 1. 候选收集

逐个采样串尝试 `str(float(x))` 规范化（数值串统一 `"5"→"5.0"`，非数值原样），非空且**未见过的**才收（首现序去重）。

### 2. declarative 改写句

- 缓存非空且含 `answers[0]`：把**最后一次**出现的 answers[0] 替换为当前候选（官方反向 find 定位拼接 ≡ `rfind` 替换）；
- 缓存为空或不含 answers[0]：fallback 模板 `"{question_tail} The answer is {answer}."`（question_tail 是最后一个句号之后的文本；GPT 重生成不入判题域）。

### 3. 数字挖空子问题

```
text = question.replace(",", ".")                  # 官方把逗号当句点用
ts = text[:len(text) - text[::-1].find(".")]       # 最后一个句号处截断（含句号）
numbers = findall(r"-?\d+\.?\d*", ts)              # 含负号与句点尾（"6." 会被抽出）
对每个数字：
    遍历 ts.split(".") 的句段，每个含该数字的句段都把
    该数字的字符区间替换为 "'X'"（字符列表切片赋值，首现位置；
    官方 number_find 置位缺失 → 多句段全换是忠实行为）
    未替换句段原样补 "."，再 replace("..", ".")   # 尾空段双句点修正
    去首空格、尾补一空格、replace(" .", ".")
    子问题 = '"' + new_ts + declarative + '"'
最多取前 3 个数字（官方 sentences[:3] / ans[:3]）
    陷阱：跨句点与句点尾的数字匹配不到句段 → 挖空不发生
```

**陷阱**：跨句点的数字（如 `-2.5`）在任何单句段里都找不到完整匹配 → 挖空不发生、句子原样；带句点尾的数字（`"6."`）同理——`"6."` 不在不含句点的句段里。

### 4. verifier 评分

每个子问题的判定串逐个 try float 转换；期望值 try `float(挖空数字)` 转换（失败用原串、再失败 0 分）；`score_d = sum(v == expected)`，候选总分 = Σ_d score_d。判定串池按 `a_idx*3 + d` 扁平索引（每候选最多 3 个子问题），缺失按空池（0 分）。

### 5. 选择与判定

- 候选唯一：`scores = {0: 1}`，不构造子问题不评分；
- 否则 argmax(scores)，**平局取最前**（np.argmax 语义）；
- `is_true` = 各候选（规范化后）是否等于规范化 GT；`chosen_correct` = 胜出候选是否命中。

返回 `dict(answers, sentences, sub_numbers, scores, chosen_idx, chosen_answer, is_true, chosen_correct)`。

## 输入约束

- `question`/`ground_truth`/`declarative` 是字符串（question 非空）；
- `sampled_answers` 非空 list[str]；`verifier_outputs` 是 list[list[str]]（判定串全字符串）。

## 提示

- `"-2.5"` 被句点切成 `-2` 与 `5` 两段——挖空替换以**句段内子串匹配**为准；
- 空串 declarative 落入「不含 answers[0]」分支同走 fallback（`""` 不含任何非空串）；
- 评分的 float 规范化是双向的：判定串与期望值各自独立 try float。
