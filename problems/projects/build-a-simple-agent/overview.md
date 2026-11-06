# 项目总览：把六块拼成一条完整的数据流

> 这一页在第 05 步（终局验收）通过后解锁，是你带走的那张「认知资产」图：做题时每一步只看局部，这里把全程数据流拼完整。

## 全程数据流拼图

```text
[ user turn ]
      |
      v
[ fixture LLM replies ]  -- explicit args, no model in sandbox
      |
      v
[ 01 validate(spec, args) ]  -> ok / missing / unexpected / type_errors / normalized
      |
      v
[ 02 run_loop(script, tools, max_steps) ] -> trace / steps_used / terminated_by / final
      |
      +---------------------------+---------------------------+
      v                                                       v
[ 03 append / compact ]                         [ 04 plan(outcomes, policy) ]
   ledger, compactions                            delays / breaker / stopped_by
      |                                                       |
      +---------------------------+---------------------------+
                                  v
[ 05 run_dialog(turns, tools, policy) ] -> per-turn calls, breakpoint, final_answer
                                  v
[ 06 audit_stage_order(order) ] -> accepted / reason
```

## 你实际写过的六块

| Step | 你写的那块 | 前序官方脚手架 | 拿它做什么 |
| --- | --- | --- | --- |
| 01 | schema 校验 | — | 给循环一个可断言的边界 |
| 02 | 循环状态机 | tool_schema | 把 fixture 回复变成轨迹 |
| 03 | 记忆账本 | tool_schema, react_loop | 把轨迹变成可回放、可压缩的账 |
| 04 | 重试熔断 | tool_schema, react_loop | 把失败路径变成确定性计划 |
| 05 | 终局拼装 | 前四块全部 | 断言多轮工具调用序列与答案 |
| 06 | 设计复盘 | 前五块全部 | 判断架构决策顺序是否合理 |

## 三条带走的结论

- **状态传递走脚手架链，不走判题机**：判题机一旦回放用户历史代码，状态空间会爆炸，防作弊也会崩坏。把前序参考实现冻成官方模块，每一步就仍然是「一次执行、一个输入」。
- **Agent 项目里真正可判的是纯逻辑**：模型输出不可复现，但 schema 校验、循环终止条件、账本压缩、退避预算、熔断窗口、调用序列都完全确定；把生成部分全部外置成 fixture，题就稳了。
- **训练类步骤不能判「收敛」**：若换成「从零训练一个小语言模型」，L3 也要固定种子 + 固定 batch 跑 N 步，只判 loss 落带与参数快照容差，绝不判收敛。

## 下一步

回到项目页重做任意一步，或看看 `scaffold/` 里的官方模块与你的实现差在哪；项目进度单独记录在 `data/project_progress.json`，不会污染普通题库的 `progress.json`。
