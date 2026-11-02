"""
Search-R1 检索掩码账。

Search-R1（arXiv 2503.09516）的多轮 rollout 把「生成 → `<search>` →
`<information>` 观测」交错拼成一条序列，训练时用 `info_mask` 把观测段
剔出 loss。回源 PeterGriffinJin/Search-R1 main：
search_r1/llm_agent/generation.py 的 `_postprocess_responses`、
`postprocess_predictions`、`execute_predictions`、`run_llm_loop`、
`_update_right_side`、`_update_rolling_state`、`_compose_final_output`，
以及 search_r1/llm_agent/tensor_helper.py。

solve(rows, vocab, search_results=None, wraps=None, max_turns=3,
      max_start_length=512, max_prompt_length=64, max_obs_length=32,
      drop_toks=("<|im_end|>", "<|endoftext|>"))

rows：[{"prompt": [id, ...], "gens": [[id, ...], ...]}]，`gens` 按顺序
供每轮生成消费（主循环每轮一次，必要时再加一次 final rollout）。
vocab：{id: token 文本}（显式 tokenizer；四个标签各占一个 token）。
search_results：{query 文本: [结果 token id]}（已 strip）。
wraps：{"obs_prefix": [...], "obs_suffix": [...], "retry": [...]}，
观测段的包装 token、以及非法动作的重试提示 token。

返回 {"rows": [...], "active_num_list": [...]}；每行账含 actions /
contents / obs_kinds / obs_truncated / turns / valid_actions /
valid_searches / finished / right_ids / segments / dropped /
final_ids / final_info_mask / rolling_ids。

域约束（ValueError）：某个 id 不在 vocab；某轮缺生成；search 的 query
在 search_results 里没有结果；任一边界为负。

账目只记真实 token：左端 prompt pad 与右端补偿 pad 在 attention_mask
与 info_mask 中恒 0，因此不入账（见 notes）。
"""


def solve(rows, vocab, search_results=None, wraps=None, max_turns=3,
          max_start_length=512, max_prompt_length=64, max_obs_length=32,
          drop_toks=("<|im_end|>", "<|endoftext|>")):
    raise NotImplementedError
