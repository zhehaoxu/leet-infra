# 解题说明 · StreamingLLM

## 为什么这样"打分"

kvpress 里所有 `ScorerPress` 都靠 `scores.topk(n_kept, -1).indices` 挑保留位。StreamingLLM 不产生连续重要性，只需要"中段先被踢"——用 0/1 二值打分就够，`topk` 会优先挑 1；剩下从 0 里挑的顺序无所谓（反正都会被剪）。

## n_pruned 的取整

`n_pruned = k_len - int(k_len * (1 - compression_ratio))`。用 **int**（截断）而不是 round，与 kvpress `compute_n_kept` 保持一致：`n_kept = int(k_len * (1 - cr))`，`n_pruned = k_len - n_kept`。

例：`k_len=10, cr=0.4` → `n_kept=int(6.0)=6` → `n_pruned=4`；`k_len=8, cr=0.25` → `n_kept=int(6.0)=6` → `n_pruned=2`。

## Sink 的直觉

Softmax 要求"分母不为 0"；因果掩码下第一个 token 只能看到自己，模型学到把"多余的注意力"甩到最早位置。若 sink 被剪，注意力会失控。因此 sink 数一般取 4；题目里作为参数显式给出便于对拍。

## 完整流水线里的角色

`StreamingLLMPress` 是纯位置规则、不看内容——最便宜、也是所有 KV 压缩 baseline 的第一号对照组。
