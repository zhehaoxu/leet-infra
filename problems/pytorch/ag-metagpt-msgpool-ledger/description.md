# MetaGPT 消息池路由账（ag-metagpt-msgpool-ledger）

**难度**：Hard ｜ **合集**：agentic-systems ｜ **order**：731

把 MetaGPT 的 publish-subscribe 消息池无状态化：给定角色表（订阅、地址、记忆开关、恢复锚点）与消息流，复算**投递路由**（谁收到、为什么）、**文本改写**（`[Message] from X to Y:` 前缀与角色归一）、**history 去重**，以及每轮 `_observe` 的**过滤 + 记忆 + 空转判定**。回源 [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) main：`metagpt/environment/base_env.py::Environment.publish_message`、`metagpt/environment/mgx/mgx_env.py::publish_message` / `_publish_message` / `move_message_info_to_content`、`metagpt/roles/role.py::_watch` / `_observe` / `put_message`，`metagpt/memory/memory.py::add` / `get` / `find_news`，`metagpt/utils/common.py::is_send_to`，`metagpt/schema.py::Message` / `MessageQueue.pop_all`。

## 你要实现的

`solve(roles, messages, tl_name="Mike", is_public_chat=True, rounds=1, all_tag="<all>", no_one_tag="no one")`：返回整轮跑批的账。

- `roles`：`[{"name", "profile", "cls", "watch", "addresses", "enable_memory", "recovered", "latest_observed", "observe_all"}]`。`addresses` 为 `None` 时默认 `{cls, name}`（`cls` 是 `any_to_str(role)` 的类路径）。
- `messages`：按发布顺序，每条 `{"id", "content", "sent_from", "send_to", "cause_by", "role", "metadata", "publicer", "user_defined_recipient"}`。
- `rounds`：`env.run()` 的次数；每轮按角色表顺序让**非空转**角色跑一次 `_observe`。

返回 `{"publishes", "roles", "history_ids", "direct_chat_roles"}`。`publishes` 每条记 `branch`（`direct_in`/`direct_out`/`tl`/`skipped`/`regular`）、`send_to`、`delivered`、`direct_chat_added`、`history_added`、`content_out`、`role_out`；`roles` 每个角色记逐轮 `news`（被 `env.run` 跳过的那轮记 `None`）、`memory`、残留 `buffer`、最终 `idle`。

## 账点（源码逐行核对）

1. **投递只看 `addresses`**：`is_send_to` 是 `MESSAGE_ROUTE_TO_ALL in send_to` 或 `addresses ∩ send_to ≠ ∅`。默认地址是 `{any_to_str(role), name}`，所以**点名类路径也能送达**；自定义 `addresses` 之后连团队负责人的名字都可能收不到。
2. **`<all>` 在环境层被加上**：`_publish_message` 里 `if self.is_public_chat: message.send_to.add(MESSAGE_ROUTE_TO_ALL)`，所以公开模式下人人收件——但 `send_to` 集合本身也涨了这一项（账里看得到）。
3. **普通消息必经团队负责人**：`message.send_to.add(tl.name)`——加的是 **name** 不是 profile；即使收件人里已经有 TL 也照加（集合去重）。私聊模式（`is_public_chat=False`）下如果 TL 的地址里没有这个名字，TL 就真收不到。
4. **转交判定用 profile**：`elif publicer == tl.profile`（`publicer` 是「谁代发的」，UI 传的是 TL 的 profile）。名字等于 profile 是巧合，判据是 profile。
5. **只有 `send_to == {"no one"}` 才丢**：TL 的哑消息走 `if message.send_to == {"no one"}: return True`——**整集合精确相等**，多带一个收件人就不丢；而且这条早退发生在 `attach_images` 之后、`history.add` 之前，所以**连 history 都不记**（`history_added == 0`）。
6. **直聊簿记**：`user_defined_recipient` 分支里，只有**当前空转**的收件人进 `direct_chat_roles`（忙的话是「用户在帮它干活」，不进簿）；回包由 `sent_from in direct_chat_roles` 命中并**立刻消费**（`remove`），随后私聊模式下**不发布**——但 `history.add` 照样执行（`history_added == 1`、`content_out is None`）。
7. **文本改写**：`move_message_info_to_content` 只留 `system/user/assistant` 三种 role（其余压成 `assistant`）；`sent_from` 优先取 `metadata["agent"]`；`send_to == {<all>}` 时目标显示成团队负责人名字，否则显示「收件人减 `<all>`」按字典序拼接；发送者为空则显示 `User`。
8. **history 按整条消息去重**：`Memory.add` 是 `if message in self.storage: return`（pydantic 全字段等值）。每次成功发布会进两条（改写过的那份 + 原文），重复发同一条则**不再涨**——但投递照旧（buffer 里会有多份）。
9. **`_observe` 的两道兴趣门**：`cause_by in watch` **或** 名字在 `send_to` 里；再过第三道「不在已有记忆里」。同一批 buffer 里的重复消息**全部**能过（第三道门只跟记忆比，不跟批内比）。
10. **`observe_all_msg_from_buffer`**：buffer 里的**全部**消息进记忆（哪怕没被 watch 到、这一轮没反应），只有被 watch 到的进 `news`。`enable_memory=False` 时第三道门失效（`old_messages = []`），已观测过的消息会再次进 `news`。
11. **恢复口径**：`recovered and latest_observed` 时 `news = memory.find_news([latest], k=10)`——**从记忆里**捞，不是从 buffer；捞到了就**不 drain buffer**（那一轮 buffer 原地不动）。锚点若已在记忆最近 10 条内则返回空，退回 drain。`_observe` 末尾 `latest_observed_msg = news[-1] if news else None`：这一轮没反应就把锚点清掉。
12. **空转判定**：`is_idle = not rc.news and not rc.todo and msg_buffer.empty()`；`env.run` 对空转角色**直接跳过**（账上记 `None`）。所以刚观测过的角色这一轮仍非空转，至少要**两轮**才全部空转。域约束（ValueError）：缺团队负责人、角色重名、轮数为负、直聊收件人不是已注册角色。

## 样例

```python
roles = [{"name": "Mike", "profile": "Mike", "watch": ["UserRequirement"],
          "cls": "metagpt.roles.role.Role"},
         {"name": "Alice", "profile": "Alice", "watch": ["WriteCode"],
          "cls": "metagpt.roles.role.Role"}]
msgs = [{"id": "m1", "content": "hi", "sent_from": "User",
         "send_to": ["Alice"], "cause_by": "WriteCode"}]
led = solve(roles, msgs, rounds=2)
p = led["publishes"][0]
# branch == "regular"；send_to == ["<all>", "Alice", "Mike"]
# delivered == ["Alice", "Mike"]；history_added == 2
# content_out == "[Message] from User to Alice, Mike: hi"
assert led["roles"]["Alice"]["news"] == [["m1"], []]
assert led["roles"]["Alice"]["idle"] is True
```

## 计分

15 个检查点：地址投递与 `<all>` 广播、四条分支（含 profile≠name 的转交判据与哑消息跳过）、文本改写与 role 归一、直聊簿记（忙/闲、私聊丢弃但仍记 history）、history 去重与投递去重分离、`_observe` 三道门、`observe_all`、`enable_memory` 开关、记忆门（等值锚点构造）、恢复口径（锚点 / 空转 / 已在最近 10 条）、轮次与空转、默认值，以及 50 组随机角色表与消息流对拍。
