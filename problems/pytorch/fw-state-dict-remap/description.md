# Checkpoint 键重映射与 strict=False 账目

## 题目描述

模型改版后旧 checkpoint 的键名对不上新模块：前缀改名、`w/b/ln` 换成 `weight/bias/layernorm`、Conv1D 风格的 `embed.w` 要转置成 Linear 的 `embed.weight`、旧语言模型头被删除、新模型加了旧 checkpoint 没有的 `gate` 层。这是 HF `convert_*_checkpoint` 脚本的日常。本题要求完成一次完整的「重映射 → `strict=False` 加载 → 记全四类账」的流程。

## 规则

实现 `solve()`，返回一个 `dict`，包含四类账目：`mapping`（重命名对照）、`missing_keys`、`unexpected_keys`、`loaded`（加载后的参数值）。

新模型结构、旧 checkpoint 的键/形状/确定性取值、重映射规则、返回值格式，全部在 starter 的 docstring 中给出（唯一完整信息源）。要点：

1. 旧 checkpoint 的值是连续递增整数段（\(\texttt{arange} + \texttt{cursor}\)），不依赖随机数。
2. `transformer.h.{i}.*` 重命名为 `blocks.{i}.*`；`embed.w` 重命名为 `embed.weight` **并把值转置**（旧 (3,2) → 新 (2,3)，行主序展平）。
3. 用 `model.load_state_dict(remapped, strict=False)` 加载；`missing_keys`/`unexpected_keys` 返回前**必须自行排序**（框架不保证顺序）。
4. `loaded` 取加载后的 `model.state_dict()` 逐参数行主序展平成 `list[float]`；未被加载的参数保持零初始化。

## 示例

```
输入：无
输出：
{
 "mapping": {
  "embed.w": "embed.weight",
  "transformer.h.0.b": "blocks.0.bias",
  "transformer.h.0.ln": "blocks.0.layernorm",
  "transformer.h.0.w": "blocks.0.weight",
  "transformer.h.1.b": "blocks.1.bias",
  "transformer.h.1.ln": "blocks.1.layernorm",
  "transformer.h.1.w": "blocks.1.weight"
 },
 "missing_keys": ["blocks.0.gate", "blocks.1.gate"],
 "unexpected_keys": ["lm_head.weight"],
 "loaded": {
  "embed.weight":     [25.0, 27.0, 29.0, 26.0, 28.0, 30.0],
  "blocks.0.weight":  [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
  "blocks.0.bias":    [6.0, 7.0],
  "blocks.0.layernorm": [8.0, 9.0, 10.0],
  "blocks.0.gate":    [0.0, 0.0],
  "blocks.1.weight":  [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
  "blocks.1.bias":    [19.0, 20.0, 21.0, 22.0],
  "blocks.1.layernorm": [23.0, 24.0],
  "blocks.1.gate":    [0.0, 0.0, 0.0, 0.0]
 }
}
```

注意 `embed.weight` 的转置效应：旧值 \(\begin{pmatrix}25&26\\27&28\\29&30\end{pmatrix}\) 转置后行主序展平是 \(25,27,29,26,28,30\)，不是 \(25,26,\dots,30\)。

## 约束

- 返回 `dict`，键为 `mapping` / `missing_keys` / `unexpected_keys` / `loaded` 四项。
- 所有值精确对拍（构造均为小整数，`float32` 无损）；两个键列表必须是排序后的 `list[str]`。
- 流程完全确定，不涉及随机数，不要动全局 seed。
