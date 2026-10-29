# PEFT：MultiTask-PromptTuning 任务共享池拼接

## 题目描述

**MultiTask Prompt Tuning**（arXiv:2202.13914 / 2211.03831）把软提示拆成两部分：

1. **共享提示** `shared_prompt`：所有任务共用，长度为 \(s\)。
2. **任务专用提示** `task_prompts`：每个任务一段长度为 \(t\) 的提示嵌入。

对于一个 batch，根据 `task_ids` 取出每样本对应的任务专用提示，再把共享提示与任务提示按顺序拼接，得到每样本的前缀嵌入：

\[P_b = \bigl[\text{shared_prompt};\ \text{task_prompts}[\text{task_ids}[b]]\bigr],\qquad b=1,\dots,B.\]

输出形状为 \((B, s+t, H)\)，其中 \(H\) 为隐藏维度。

## 任务

给定：

- `shared_prompt`：\((s, H)\) 的 float32 张量；
- `task_prompts`：\((T, t, H)\) 的 float32 张量，\(T\) 为任务数；
- `task_ids`：\((B,)\) 的 int64 张量，取值范围 \([0,T)\)。

返回拼接后的前缀嵌入 `P`：\((B, s+t, H)\)。

- 共享提示在 batch 内复制 \(B\) 份。
- 拼接顺序为「共享段在前，任务段在后」。
- 输出 float32。

## 提示

- 用 `task_prompts[task_ids]` 做任务级索引，得到 \((B,t,H)\)。
- 用 `shared_prompt.unsqueeze(0).expand(B, -1, -1)` 把共享提示扩展到 batch 维度。
- 沿长度维（dim=1）拼接即可。
