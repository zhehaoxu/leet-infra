# SFT 数据：助手回复损失掩码

## 题目描述

指令微调（SFT）的一条黄金准则：**只让模型在助手（assistant）回复的 token 上计算损失**，system / user / 角色分隔符 / padding 都不该参与监督——否则模型会去「背诵」用户说的话。这就是 TRL 的 prompt-loss masking、各家的 assistant-only loss mask。本题从「每个 token 的角色标注」还原出这份 0/1 监督掩码。

## 规则

实现 `solve(token_ids, role_ids, assistant_id, eos_id, supervise_eos=True)`，返回 float32、形状 \((B, L)\) 的掩码，取值 \(\{0.0, 1.0\}\)。

- `token_ids`、`role_ids`：int64 \((B, L)\)，同一 batch 已右补齐到等长 \(L\)（pad 位置的 role 不等于 `assistant_id`）。`role_ids[b, i]` 是第 \(b\) 条第 \(i\) 个 token 所属角色编号。
- 把 role 等于 `assistant_id` 的**连续段**（一段极大化 assistant run）标为监督位，即掩码取 1。
- `supervise_eos=False` 时，对每个 assistant 段**只**丢弃其**最后一个** token（若该位置 `token_ids[b, ·] == eos_id`）；段中间即使出现 `eos_id` 也保留。`supervise_eos=True`（默认）则整段都监督。
- 其余位置（非 assistant、被丢弃的段尾 EOS、pad）掩码取 0。

## 示例

`assistant_id=2, eos_id=99, supervise_eos=False`：

```
token_ids = [[5,6,7,8,99,0],      role_ids = [[0,1,2,2,2,3],
             [1,2,99,3,4,0]]                  [1,2,2,2,3,3]]
mask      = [[0,0,1,1,0,0],       # 行0 assistant 段 [2,3,4]，段尾 tok=99=eos 被丢弃
             [0,1,1,1,0,0]]       # 行1 assistant 段 [1,2,3]，段尾 tok=3≠99，整段保留（99 在段中间不丢）
```

同一输入 `supervise_eos=True` 时行0掩码变为 `[0,0,1,1,1,0]`。

## 约束

- CPU int64 输入、float32 输出，形状严格 \((B, L)\)。
- assistant run 是「role 等于 `assistant_id`」的极大连续段；跨段互不影响。
- `supervise_eos` 只影响每段**末尾一位**、且仅当该位 token 等于 `eos_id`。
- 输出取值仅 0.0 / 1.0。
