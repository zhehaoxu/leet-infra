# GenKnowSub 实现笔记

## 出处与结构

- 论文 arXiv:2505.10939（ACL 2025 短文，德黑兰大学），官方仓 `saharsamr/Modular-LLM`；PEFT 侧实现在 `src/peft/tuners/lora/arrow.py` 的 `gen_know_sub`（与 Arrow 同 PR #2644 落地，`use_gks`/`gks_adapter_names`/`gks_done`/`gks_added_adapter_names`）。
- `gen_know_sub`：`avg_A = stack([lora_A[n].weight for n in gks_adapter_names]).mean(0)`；`lora_A[name].weight.data.sub_(avg_A)`（B 同理）**就地**改写任务因子。`gks_done=False` 时全量净化；之后仅 `gks_added_adapter_names`。
- 门控：`if not self.use_gks or self.gks_done and not self.gks_added_adapter_names: return`——Python 运算优先级：`not use_gks or (gks_done and not added)`。
- `create_arrow_model`：`use_gks=True` 且 `general_adapter_paths` 空抛 ValueError；`gks_` 前缀适配器与 task 适配器一起过 `check_loaded_lora_compatibility_arrow`（共享 rank/形状/模块集）。
- 理论："forgetting-via-negation" 任务算术（arXiv:2212.04089）。

## 判题口径与怪癖

- 减法在**因子**上：\((B-\bar B)(A-\bar A) = B@A - B@\bar A - \bar B@A + \bar B@\bar A\)——交叉项恒等式是本题最强信号（m-sub-* / m-td-* 变异暴露）。
- 官方 in-place `data.sub_` 在判题域等效为 clone+sub（纯函数）；输入栈逐位不变的断言防 in-place 污染。
- 增量轮的 `only` 子集：花式索引赋值 `a_new[idx] = a_new[idx].sub_(avg_a)` 两侧同序，因此 `sorted(only)` 与 `list(only)` **真等价**（不可作变异）；未选中专家用 `torch.equal` 断言。
- 门控是三态状态机不是布尔：`or`→`and`、`and`→`or`、`"all"`→`"added"`、`"added"`→`"skip"` 四个变异各自暴露一个分支。
- 多语言 GK LoRA（英/法/德）参与平均是论文卖点——平均分支 `mean(0)`/`dim=0`/`mean(1)` 变异覆盖。
- 纯均值减法账小，靠交叉项恒等式与门控状态机补足变异面。
