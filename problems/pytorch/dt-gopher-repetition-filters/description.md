# Gopher 重复文本过滤：行/段/n-gram 十三条阈值账

## 题意

Gopher（Rae et al., 2112.11446）附录 A.1.1 的 MassiveWeb **Repetition Removal**：对文档计算 13 个重复度量（Table A1），任一度量**严格大于**阈值（"surpasses"）即整文档拒绝。本题逐条转写这套「文档内重复账」——与 `dt-minhash-lsh-dedup` 划界：那是**文档间** MinHash 13-gram Jaccard 近似去重，本题是**文档内**精确 n-gram 切片计数。

## 求解

13 个度量与计算口径（题面钉死，全部精确匹配、区分大小写与标点）：

1. **行**：`text.splitlines()` 每行原样（不 strip，**空行也参与重复**）。
   - `duplicate_line_fraction` = 重复行实例数 / 总行数（某行出现 k≥2 次，贡献 k 个实例）；
   - `duplicate_line_character_fraction` = 重复行实例字符和 / 全部行字符和。
2. **段落**：按空行（strip 后为空的行）分隔的极大连续非空行块；段字符 = 块内各行字符和（不含分隔空行与段内换行）。
   - `duplicate_paragraph_fraction` / `duplicate_paragraph_character_fraction` 同上口径。
3. **词**：由空白分隔的最大非空白连续段（记录字符区间 [s, e)）；word n-gram = 连续 n 个词，出现实例覆盖区间 = [首词 s, 末词 e)。
   - `top_n_gram_character_fraction`（n=2,3,4）：出现次数最多的 n-gram（并列取**首次出现位置最早**者），其全部出现实例的覆盖区间并集 / len(text)；
   - `duplicate_n_gram_character_fraction`（n=5..10）：全部出现次数 ≥2 的 n-gram 的**全部**出现实例覆盖区间并集 / len(text)——重叠区间合并，字符不重复计（论文 "taking care not to count characters that occur in overlapping n-grams more than once"）。

分母为 0、len(text)==0 或词数不足 n 时该度量记 0.0；所有覆盖 fraction 分母 = len(text)（含空白与换行），覆盖只计词字符。

**拒绝规则**：按 Table A1 行序（line → paragraph → line char → para char → top2 → top3 → top4 → dup5..dup10）找到第一个 `metric > threshold` 的度量作为 `reject_reason`；`thresholds` 入参可覆盖任意子集（键须是 13 个合法度量名、值 ∈ [0,1]）。

## 示例

```
输入：text = "x\n\nx\n\na\nb\n\nc\nd\n\ne\nf\n\ng\nh"
输出：reject_reason = "duplicate_line_fraction"（5 个空行 + "x"×2 = 7 个重复行实例 / 15 行）
```

## 约束

- text 为 str；thresholds 为 None 或合法 dict，否则 `ValueError`。
- 返回 `{"accepted", "reject_reason", "metrics"}`，metrics 含全部 13 个度量名 → float。
