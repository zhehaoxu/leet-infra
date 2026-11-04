# 解题笔记 — Moonshine context biaser

## 官方账

- moonshine-ai/moonshine `core/context-biaser.cpp` + `.h` + `context-biaser-test.cpp`（官方 doctest 全行为钉值）；`scripts/eval-keyterm-biasing.py` 是评测编排不入题。仓已从 usefulsensors 迁移到 moonshine-ai。
- 判题域：Trie 注册（共享前缀复用/重复计数/空序列）、变体规则、对数爬坡、apply 提案去重 max、advance root 常驻走账、解码循环模拟。分词（bin-tokenizer）与流式投机验证账不入题（另立）。

## 踩坑

- **官方测试值是断言真值**：depth1=2.0、depth2=2(1+ln2)、depth3=2(1+ln3)、共享前缀单次 3.0、双路径取 max 3(1+ln2)——直接照抄官方 doctest 数值，不需要 RNG 锚点（本题无 torch）。
- **apply 守卫是真等价**：`sequence_count==0` 的 no-op 守卫剔除后行为不变（空 trie 无候选可提，apply 仍返回 {}）——变异测试不可判，记入真等价剔除清单。
- **bias_walk 的时序**：bonus 落在 token 选择**之前**、advance 用**已选 token**；写反（先 advance 后 apply）首 token 就拿不到入口 bonus——m-walk-order 变异 CAUGHT-ERR 兜住。
- **活跃节点在「错 token 那一步」仍然生效**：`[10,42,10]` 的第 2 步（42 自己）时 node1 还活着（11 的深 bonus 仍被提案），42 走完才回落 root——断言写 `{10:2.0}` 会漏掉 `{11: 2(1+ln2)}`，期望值要按官方语义逐步推。
- **advance 无去重但天然无重复**：root 先 push、各活跃节点的孩子节点索引全局唯一——`next_active` 不会重复；不必再去重（加了也不算错，但非官方行为）。
- **重复注册序列**：`sequence_count` 仍 +1（计数语义是「注册次数」不是「不同序列数」），但节点不增——两个分支都要断。
- advance 的 `sequence_count==0` 提前返回同样是防御性（active 本来就是 [0]），与 apply 守卫同属观察等价区，变异只挑一个代表即可。

## mut 15 变异清单

bonus：去 (1+·) 平滑；apply：深度用 parent 而非 +1、去重 max→sum（叠）、max→min、越界检查移除；variants：锚判断移除、trim 移除、空串守卫移除；build：child 深度不 +1、共享前缀不复用（强制建新）、空序列也计数；advance：root 不 push、匹配后 push 自身（走账搁浅）；bonus_for_token：max→min；bias_walk：apply/advance 次序颠倒（CAUGHT-ERR）。15/15 全 CAUGHT + 1 真等价剔除（apply 空守卫）。
