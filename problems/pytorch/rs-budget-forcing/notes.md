# notes — rs-budget-forcing（order 662）

## 回源

- 出处：s1: Simple test-time scaling（2501.19393）的 budget forcing 技巧：`<think>` 段达到 token 预算即注入 `</think>Final Answer:`（length forcing）；未到预算但想延长思考则注入 `Wait`（wait forcing，多轮续跑）。官方实现绑模型推理流，本题把可观察部分抽成**逐 token 记账的状态机**：步序列、预算、Wait 成本、回答截断。
- 口径钉死的三个易错点：
  - 被截断的思考步记**前缀长度**进 `step_emitted`（不是 0 也不是整步）；
  - Wait 步先付 `wait_cost` 再发射，Wait token 计入总量但**不计入 `think_emitted`**；
  - `wait_cost` 付不起即停止尝试（不是跳过继续试后续 Wait）。

## 判题面

- oracle：`test_cases._py` 独立重写（while 循环逐 token 扣账），200 组随机对拍（预算、步长、wait 组合随机）。
- 锚点：整步边界 `emitted + c == budget` 恰好放行；截断步记前缀（40 不是 55：wait 步记 `wait_emitted` 不计 `think_emitted`）；付不起 wait_cost 停止且 `wait_log=False`；wait 后再截断停止后续 Wait；回答按 remaining 截断 + `answer_truncated`；无 Wait 时退化 length forcing。
- 非法输入：三个序列非列表、条目 bool/负/非 int、budget 非正 int、wait_cost bool/负 → ValueError。

## 变异记录（9 条全抓）

budget-off-by-one（`<=` → `<`）、forced-not-set、wait-cost-free（不扣 wait_cost）、wait-after-forced（截断仍续 Wait）、final-no-truncate、wait-not-counted（wait 扣账丢失）、step-prefix-dropped（截断步记 0）、think-total-wait（think_emitted 混入 wait）、first-wait-free（首个 Wait 免单）。

## 红线

不出模型调用与真实解码流；与 `rs-overthinking-metrics`（效率统计账）划界：本题是**预算控制状态机**，不做跨样本统计。
