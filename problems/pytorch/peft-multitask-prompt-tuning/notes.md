# PEFT：MultiTask-PromptTuning 任务共享池拼接——算法详解

## 1. 公式与出处

MultiTask Prompt Tuning 源自 arXiv:2202.13914（MPT）与 arXiv:2211.03831（MPT 在任务路由上的扩展）。它在 Prefix-Tuning/Prompt-Tuning 的基础上把软提示拆成：

- 所有任务共享的 `shared_prompt`；
- 每个任务独有的 `task_prompts[task_id]`。

对于一个 batch，每样本的前缀嵌入为：

\[P_b = \bigl[\text{shared_prompt};\ \text{task_prompts}[\text{task_ids}[b]]\bigr].\]

## 2. 实现细节

- `task_prompts[task_ids]` 利用高级索引一次性取出 \((B,t,H)\)。
- `shared_prompt.unsqueeze(0).expand(B, -1, -1)` 把共享段复制到 batch 维度。
- 沿 dim=1（序列长度维）拼接，得到 \((B, s+t, H)\)。

## 3. 退化与判题考点

- batch 大小 \(B=1\)：仍要正确复制共享段并拼接。
- 所有样本同一 `task_id`：共享段仍按 batch 复制，任务段相同。
- 拼接顺序必须是「共享在前、任务在后」，反例断言验证。
- 完全确定性，无随机性。

## 4. 与现有题的分工

- `peft-prompt-tuning`：单任务可学习连续提示。
- `peft-prefix-tuning`：Prefix-Tuning 的前缀生成。
- 本题：多任务场景下共享池 + 任务专用段的拼接。
