# 详解：SFT 助手回复损失掩码

## 出处
工程约定，非单一论文：ChatML / Llama 对话模板把每条 turn 打上角色；TRL `DataCollatorForCompletionOnlyLM`（`response_template="[/INST]"` 等）与各家 SFT 脚本都通过「屏蔽 prompt、只保留 completion」实现 assistant-only 监督。是否把段尾 EOS 纳入监督取决于训练配方（要教会模型「何时停止」通常监督 EOS，反之可关），本题把它做成 `supervise_eos` 开关。

## 核心与易错点
1. **run 检测**：assistant 段是 role 连续等于 `assistant_id` 的**极大**段，段间以 role 变化分隔；每段单独处理段尾。
2. **EOS 只丢段尾**：`supervise_eos=False` 只对段的**最后一个**位置生效，且必须 `token_ids[b,·]==eos_id`；段中间出现的 `eos_id` 属于回复内容，保留。这是最容易写错的地方——不要「扫到 eos 就截断」。
3. **pad 天然排除**：pad 位 role 不等于 assistant，自动为 0；无需特判 pad_token_id。
4. **输出取值**：只能是 0.0 或 1.0，且 shape 严格等于输入 \((B, L)\)。

## 对拍口径
参考解「逐行找 assistant 极大段 → 决定是否丢段尾 → 段内置 1」。判题 `_oracle` 用**完全独立**的逐位置规则：`mask[i]=1` 当且仅当 `role[i]==assistant_id` 且**不**满足「i 是 assistant run 的最后一个且 token[i]==eos_id 且 not supervise_eos」——run-last 用「下一位 role≠assistant」判断，与参考解的「找段再回退一位」结构不同、数值等价。另设「两 supervise_eos 对比」「段中含 eos 不丢」「无 assistant 全 0」「多段」断言。

## 复杂度
\(O(B L)\)。
