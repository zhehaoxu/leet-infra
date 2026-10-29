# 解题思路与来源边界

## 1. 验证注册表，再构建 trie

先验证全部 content 非空且互不重复，包括空原文或将被移除的 special 项。遍历原列表注册 pattern，终止节点记录原列表下标；开启 `split_special_tokens` 时，不将 special pattern 放进 trie。不要原地删除列表项或重新编号。

这一步只是本题的候选过滤策略，不意味着 special token 有比普通 token 更高的匹配优先级。

## 2. 最长选择必须先于 single_word 检查

在当前位置沿 trie 尽可能走远，记录最后遇到的终止节点：

- 没有命中则原文游标前进一个字符。
- 有命中就确定当前起点的最长原始候选。
- 之后才检查 `single_word` 的左右邻居是否属于 ASCII 字母、数字或 `_`。
- 无论边界检查成功还是失败，下次扫描都从候选末尾开始。

因此，原文 `zabc!` 中若注册普通 `ab`、要求单词边界的 `abc` 和普通 `b`，实际输出为：

```python
[('zabc!', 0, 5, -1)]
```

不能在边界失败时回到 `ab`；也不能在 `abc` 内部继续找 `b`。连“起点在内部、终点在外部”的重叠候选也已被原始非重叠选择排除。

## 3. 普通片段延迟输出

另存 `ordinary_start`。只有遇到成功匹配时，才输出从 `ordinary_start` 到该匹配起点的普通片段，再输出该 token 并更新 `ordinary_start`。边界失败不更新这个变量，所以失败候选与前后未匹配文本自然合并。最后补上尾部普通文本。

不能把相邻的两个成功 token 合并，即使其 ID 一样。空白不被 added token 吸收，片段始终是原文切片。

设原文长 $n$，pattern 最大长度 $L$，注册字符总数 $S$。简单 trie 实现的时间为 $O(S+nL)$，含输出空间为 $O(S+n)$；本题 $L\leq30$，无需实现 Aho–Corasick 自动机。

## 4. 真正执行的来源核验

实际通过 `urllib.request`、`io.BytesIO`、`tarfile` 在内存读取 [tokenizers 0.23.3 的 PyPI 源码包](https://files.pythonhosted.org/packages/e0/7c/2cabb2174e772636683008f2c5621949b645da7d303c596589e84516a184/tokenizers-0.23.3.tar.gz)，未下载到文件或解压落盘。对应文件是 `tokenizers/src/tokenizer/added_vocabulary.rs`：

- `99–102`：左右边界与空白正则。其单词字符规则在本题 ASCII 域内为字母、数字、下划线。
- `379–423`：分别构建 normalized 和 non-normalized 集合，采用 `LeftmostLongest`。
- `444–462`：先迭代非重叠候选，再跳过 special 和检查单词边界；失败不会回退短匹配。
- `464–480`：完整实现还支持空白吸收，本题关闭。
- `530–552`：完整实现有规范化前后两阶段，本题固定为同一阶段。

实际探针使用本地 `.venv-eval` 的 Python **3.12.12**、`tokenizers` **0.23.3**：

- 创建 `Tokenizer`，模型为 `WordLevel({'<unk>': 0}, unk_token='<unk>')`。
- 不设置 normalizer 或 pre-tokenizer。
- 所有 `AddedToken` 显式设为 `normalized=False`、`lstrip=False`、`rstrip=False`，并设置相应 `single_word`、`special`。
- 设置 `encode_special_tokens`，执行 `encode(text, add_special_tokens=False)`。
- 通过 encoding 的 offsets 还原原文片段，将库 ID 映射回注册下标；普通区间合并。

冻结了 **10 条原生结果**，覆盖左最早、同起点最长、边界失败阻断、跨区间重叠、下划线和数字边界、标点内容、special 无优先权、跳过孤立 special 和空原文。

## 5. 明确偏离：special 的过滤时机

**本题不是 tokenizers 0.23.3 的完整等价实现。** 源码 `444–453` 显示，原生实现先从包含 special 的 trie 产生候选，再在迭代体中跳过 special。本题明确要求在候选选择前移除 special，所以存在可观测差异。

已实际执行的辨别输入：

```python
text = 'abc!'
added_tokens = [
    {'content': 'ab', 'single_word': False, 'special': False},
    {'content': 'abc', 'single_word': True, 'special': True},
]
split_special_tokens = True
```

- 原生库完整注册并开启 `encode_special_tokens`：`[('abc!', 0, 4, -1)]`。
- 本题契约及参考解：`[('ab', 0, 2, 0), ('c!', 2, 4, -1)]`。

第二个探针是 `abcd!`，注册 special `abc` 和普通 `bcd`，开关为真：原生库保留整段普通文本；本题输出：

```python
[('a', 0, 1, -1), ('bcd', 1, 4, 1), ('!', 4, 5, -1)]
```

这 **2 条预过滤 fixture** 是先移除 special 再创建真实 tokenizer 得到的，保留原始注册 ID；它们与 10 条原生 fixture 分开标注，不冒充原生开关行为。

其他范围约束同样明确：只处理 ASCII、identity normalization、`lstrip=rstrip=False`、单阶段、唯一非空 content、不做 BPE；返回的 ID 是注册下标而非完整模型词表 ID。不得扩展成“特殊 token 自动有优先权”或宣称实现了完整 AddedVocabulary。

## 6. 独立 oracle 与自测

测试 oracle 不用 trie：暴力枚举所有出现，按起点升序、长度降序排序，先固定非重叠原始区间，再独立做边界过滤，最后按成功区间拼出完整输出。它不会先过滤不合法边界后再选最长。

**16 个 checks** 包含 12 条库探针 fixture、大小写、ID 保留、相邻匹配、无空白吸收、完整偏移覆盖、非法注册、输入不可变性、300 字符及 30 个/最长 30 字符 pattern 边界。固定种子 `4592026` 生成 180 组一般随机数据与 60 组高重叠前缀数据，各测两个开关值，共 **480 次随机 oracle 对拍**。交付的实现和测试仅依赖 Python 标准库。

先运行 `.venv-backend/bin/python scripts/add_problem.py --selftest dt-added-token-match --mode submit` 得到 **16/16，ACCEPTED**，之后用 `.venv-eval/bin/python` 执行所有文档样例，最后撰写中英文文档。
