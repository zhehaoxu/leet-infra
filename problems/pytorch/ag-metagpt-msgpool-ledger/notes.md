# notes（ag-metagpt-msgpool-ledger）

## 回源

- 仓库：`FoundationAgents/MetaGPT`（原 geekan/MetaGPT）main。核心文件：`metagpt/environment/base_env.py`（248 行，`Environment.publish_message` 在 175-195 行）、`metagpt/environment/mgx/mgx_env.py`（100 行，MGX 的三条额外规则）、`metagpt/roles/role.py`（593 行，`_observe` 在 399-428 行）、`metagpt/memory/memory.py`（113 行）、`metagpt/utils/common.py::is_send_to`、`metagpt/schema.py`（`Message` 的字段默认值与 `MessageQueue.pop_all`）。
- 论文：MetaGPT（arXiv 2308.00352）§2.2「Role Specialization / Publish-Subscribe」把 `_watch` + `_observe` 说成「订阅自己关心的 action 产出的消息」，但**代码里是两级**：环境按 `addresses` 做传输层投递（`is_send_to`），角色再按 `cause_by in watch` 或「名字在收件人里」做兴趣层过滤。论文与代码的口径差异是本题的第一课。
- MGX 是 MetaGPT 在 SaaS 产品里的分支（`mgx_env.py` 里的 team leader 转交、公开/私聊、直聊簿记），只在 GitHub 主仓里，不在论文里。

## 官方 quirk（逐条核实）

1. **两级路由**：`Environment.publish_message` 遍历 `member_addrs`（`role → addresses`），`is_send_to(message, addrs)` 命中就 `role.put_message(message)`（进私有 buffer）。`is_send_to` 先看 `MESSAGE_ROUTE_TO_ALL`，再逐个地址比对。默认地址 `{any_to_str(self), self.name}`（`check_addresses` 里设的），所以类路径与名字都能送。
2. **`<all>` 由环境层加**：`MGXEnv._publish_message` 先 `send_to.add(MESSAGE_ROUTE_TO_ALL)` 再转换——注意这一步在**转换之前**，所以 `send_to == {<all>}` 的「显示成团队负责人」分支只在收件人本来就只有 `<all>` 时触发。
3. **普通消息加 `tl.name`**：`MGXEnv.publish_message` 的 else 分支 `message.send_to.add(tl.name)`。加的是 name；私聊模式下若 TL 的地址里没有这个名字，TL 收不到自己该转交的消息。
4. **转交判据是 profile**：`elif publicer == tl.profile`。产品里 `publicer` 传的是 TL 的 profile（不是 name），两者在默认构造下相同，一改 profile 就能看出差异。
5. **哑消息整集合比较**：`if message.send_to == {"no one"}: return True`——早退在 `history.add(message)` 之前，所以这条消息**既不投递也不进 history**。`{"no one", "X"}` 不触发。
6. **直聊簿的两半**：入簿要求收件人**当前空转**（`self.get_role(role_name).is_idle`）；回包必须在 `sent_from` 命中时 `remove`（消费），私聊下不发布但 `history.add` 照跑。
7. **`move_message_info_to_content` 三个细节**：role 白名单 `["system","user","assistant"]`；`sent_from` 优先 `metadata[AGENT]`（AGENT = `"agent"`）；`send_to == {MESSAGE_ROUTE_TO_ALL}` 时显示 `TEAMLEADER_NAME`，否则 `", ".join({...})`——**集合拼接**，顺序依赖哈希（本账统一按字典序，见「保真性约定」）。
8. **`Memory.add` 去重**：`if message in self.storage: return`（pydantic 全字段等值）。`index[cause_by]` 只在真的 append 时更新。`get(k=0)` 是 `storage[-0:]` = 全量（`find_news` 的 `k=10` 取最近 10 条）。
9. **`_observe` 三道门**：`(cause_by in watch or name in send_to)` 且 `n not in old_messages`；`old_messages = [] if not enable_memory else memory.get()`。批内重复不会被门挡（门只跟记忆比）。
10. **`observe_all_msg_from_buffer`**：进记忆的是 `news`（**drain 出来的整批**），不是 `rc.news`（过滤后的）；所以「没反应的消息」也会被记住，但不会出现在 `news` 账里。
11. **恢复口径**：`if self.recovered and self.latest_observed_msg: news = memory.find_news(observed=[latest], k=10)`；`if not news: news = msg_buffer.pop_all()`。捞到了就**不 drain buffer** → 这一轮 buffer 原地不动，下一轮才读。末尾 `latest_observed_msg = rc.news[-1] if rc.news else None`：空手一轮就把锚点清掉。
12. **`is_idle` 三要素**：`not rc.news and not rc.todo and msg_buffer.empty()`。`env.run` 只对非空转角色 `await role.run()`，而 `run` 开头又无条件 `_observe`，所以「观测到的这一轮」结束时 `rc.news` 非空 → 下一轮还要再跑一次才回到空转。`self.rc.news` 每轮被整体重算（不是追加），这是「两轮才空转」的来源。

## 保真性约定（与官方不同的地方，逐条说明）

- **集合拼接顺序**：官方 `", ".join({...})` 走集合迭代序（CPython 下随 `PYTHONHASHSEED` 变化）。本账固定字典序，否则账不可复现；`send_to` 输出也一律排序。
- **pad/张量层全部剥掉**：官方投递是「往 buffer 里 push 同一个 Message 对象」，本账存深拷贝（内容相等），等价于 `MessageQueue` 的 FIFO。
- **角色不跑 LLM**：`_think`/`_act` 没有入参化，本题只到 `_observe` 与 `is_idle`；因此 `rc.todo` 恒为空，`is_idle` 化简为「没 news 且 buffer 空」。

## 出题工程

- 首轮 7 条 SURVIVED 的处理：
  1. `m-addr-name-only`（默认地址去掉 cls）——原用例里都是「普通消息 + 点名类名」，普通分支顺手加了 `tl.name`，掩盖了 cls 的作用；改成「走转交分支 + 只点名类名」才打出来。
  2. `m-regular-add-profile`、`m-tl-by-name`——都需要 **profile ≠ name** 的 TL 才能区分，原用例里两者相同；补了 `profile="Mike-bot"` 的角色。
  3. `m-observe-no-memory-filter`、`m-enable-memory-ignored`——「记忆门」在正常消息流里不可达（每轮 observe 都会 drain buffer，观测过的消息不会再进 buffer）。构造办法：给一个 `recovered` 角色一个**与投递件逐字段等值**的锚点（`_conv_anchor`：把 content 写成 `[Message] from User to Alice, Mike: hi`），于是「记忆里有、buffer 里也有」第一次可达。
  4. `m-latest-keep`、`eq-direct-unsorted`——确认为**等价**，有证明（下条）。
- 等价证明（2 条）：
  - `eq-direct-unsorted`：直聊分支遍历收件人只用 `is_idle`/`add`，输出 `direct_chat_added` 与 `direct_chat_roles` 都排序，故遍历序不可观测。
  - `eq-latest-keep`：把「本轮没捞到东西」时的 `latest` 保留旧值不改变任何输出。理由：`latest` 只在 `recovered and latest is not None` 且角色非空转时被读；`picked == []` 只有两条路径——①buffer 已被 drain 干净（下一轮 `is_idle` 为真、直接跳过，读不到 `latest`）；②锚点被记忆门挡下（要求锚点已在记忆里，而写进记忆那一轮的 `picked` 恒非空、`latest` 就等于它本身；此后锚点要么仍在最近 10 条内使 `find_news` 返回空、走 buffer 路径，要么已滑出窗口而 `latest` 早已被后续轮的 `picked` 覆盖成别的消息）。两条都不产生新输出。
- 变异统计：**35 条实质变异全 CAUGHT + 2 条等价（有证明）**。覆盖投递（4）、分支（4）、文本改写（6）、直聊（3）、去重（2）、观测门（6）、记忆开关/恢复（3）、轮次与空转（3）、域约束（3）、账面（2）。
- 随机对拍 50 组：随机角色表（随机 watch / 自定义 addresses / `enable_memory=False` / `observe_all` / `recovered` 锚点）+ 随机消息流（随机 `publicer`、直聊、`metadata["agent"]`）+ 随机 `is_public_chat` 与 `rounds 0..3`。
- 域约束检查点**直调 solve**（不经 oracle），否则参考解自己抛的 ValueError 会把边界检查的空变异兜住（730 那题踩过）。
