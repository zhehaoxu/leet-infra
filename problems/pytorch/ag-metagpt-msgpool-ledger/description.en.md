# MetaGPT Message-Pool Routing Ledger (ag-metagpt-msgpool-ledger)

**Difficulty**: Hard ｜ **Collection**: agentic-systems ｜ **order**: 731

Turn MetaGPT's publish-subscribe message pool into a pure function: given the role table (subscriptions, addresses, memory switch, recovery anchor) and a message stream, recompute **delivery routing** (who receives what, and why), **text rewriting** (the `[Message] from X to Y:` prefix and role normalization), **history deduplication**, and the per-round `_observe` **filtering + memory + idleness** decision. Sourced from [FoundationAgents/MetaGPT](https://github.com/FoundationAgents/MetaGPT) main: `metagpt/environment/base_env.py::Environment.publish_message`, `metagpt/environment/mgx/mgx_env.py::publish_message` / `_publish_message` / `move_message_info_to_content`, `metagpt/roles/role.py::_watch` / `_observe` / `put_message`, `metagpt/memory/memory.py::add` / `get` / `find_news`, `metagpt/utils/common.py::is_send_to`, `metagpt/schema.py::Message` / `MessageQueue.pop_all`.

## What you implement

`solve(roles, messages, tl_name="Mike", is_public_chat=True, rounds=1, all_tag="<all>", no_one_tag="no one")`, returning the ledger of the whole batch.

- `roles`: `[{"name", "profile", "cls", "watch", "addresses", "enable_memory", "recovered", "latest_observed", "observe_all"}]`. When `addresses` is `None` it defaults to `{cls, name}` (`cls` is the class path of `any_to_str(role)`).
- `messages`: in publish order, each `{"id", "content", "sent_from", "send_to", "cause_by", "role", "metadata", "publicer", "user_defined_recipient"}`.
- `rounds`: how many times `env.run()` runs; each round lets every **non-idle** role in table order run one `_observe`.

Returns `{"publishes", "roles", "history_ids", "direct_chat_roles"}`. Each `publishes` entry records `branch` (`direct_in`/`direct_out`/`tl`/`skipped`/`regular`), `send_to`, `delivered`, `direct_chat_added`, `history_added`, `content_out`, `role_out`; each `roles` entry records per-round `news` (`None` for rounds skipped by `env.run`), `memory`, the leftover `buffer`, and the final `idle`.

## Ledger points (line-by-line against the source)

1. **Delivery only looks at `addresses`**: `is_send_to` is `MESSAGE_ROUTE_TO_ALL in send_to` or `addresses ∩ send_to ≠ ∅`. The default addresses are `{any_to_str(role), name}`, so **naming the class path also delivers**; with custom `addresses` even the team leader's name may stop working.
2. **`<all>` is added at the environment layer**: `if self.is_public_chat: message.send_to.add(MESSAGE_ROUTE_TO_ALL)` inside `_publish_message`, so in public mode everyone receives — and the `send_to` set itself grows by that item (visible in the ledger).
3. **A regular message always detours through the leader**: `message.send_to.add(tl.name)` — it adds the **name**, not the profile, and adds it even when the recipient list already contains the leader (set semantics). In private mode (`is_public_chat=False`), if the leader's addresses do not contain that name, the leader really receives nothing.
4. **The takeover check uses the profile**: `elif publicer == tl.profile` (`publicer` is who relayed it; the UI passes the leader's profile). Name equalling profile is a coincidence; the predicate is the profile.
5. **Only `send_to == {"no one"}` is dropped**: the leader's dummy message hits `if message.send_to == {"no one"}: return True` — an exact **set** comparison, so one extra recipient prevents the drop; and the early return happens after `attach_images` but before `history.add`, so nothing is even recorded (`history_added == 0`).
6. **Direct-chat bookkeeping**: in the `user_defined_recipient` branch only **currently idle** recipients enter `direct_chat_roles` (a busy one means the user is helping it); the reply is matched by `sent_from in direct_chat_roles` and the flag is **consumed immediately** (`remove`), after which private mode does **not** publish it — yet `history.add` still runs (`history_added == 1`, `content_out is None`).
7. **Text rewriting**: `move_message_info_to_content` keeps only `system/user/assistant` (everything else collapses to `assistant`); `sent_from` prefers `metadata["agent"]`; when `send_to == {<all>}` the target is displayed as the team leader's name, otherwise as the recipients minus `<all>` joined in lexicographic order; an empty sender displays as `User`.
8. **History dedups whole messages**: `Memory.add` is `if message in self.storage: return` (pydantic equality over all fields). Every successful publish adds two entries (the rewritten copy and the original), and re-publishing the same message adds **nothing** — while delivery still happens (the buffer holds multiple copies).
9. **Two interest gates in `_observe`**: `cause_by in watch` **or** the role's name in `send_to`; then a third gate "not already in memory". Duplicates inside one buffer batch **all** pass (the third gate compares against memory, not within the batch).
10. **`observe_all_msg_from_buffer`**: **all** buffered messages enter memory (even ones nobody watches and that trigger no reaction), while only the watched ones enter `news`. With `enable_memory=False` the third gate is off (`old_messages = []`), so an already-observed message enters `news` again.
11. **Recovery semantics**: with `recovered and latest_observed`, `news = memory.find_news([latest], k=10)` — it reads **from memory**, not from the buffer; when it finds something the buffer is **not drained** that round. If the anchor sits within the last 10 memory entries, `find_news` returns nothing and the buffer is drained as usual. At the end `_observe` does `latest_observed_msg = news[-1] if news else None`, so a round with no reaction clears the anchor.
12. **Idleness**: `is_idle = not rc.news and not rc.todo and msg_buffer.empty()`; `env.run` **skips** idle roles entirely (the ledger records `None`). A role that just observed is therefore still non-idle, and at least **two rounds** are needed for everyone to idle. Domain constraints (ValueError): a missing team leader, duplicate role names, a negative round count, and a direct-chat recipient that is not a registered role.

## Example

```python
roles = [{"name": "Mike", "profile": "Mike", "watch": ["UserRequirement"],
          "cls": "metagpt.roles.role.Role"},
         {"name": "Alice", "profile": "Alice", "watch": ["WriteCode"],
          "cls": "metagpt.roles.role.Role"}]
msgs = [{"id": "m1", "content": "hi", "sent_from": "User",
         "send_to": ["Alice"], "cause_by": "WriteCode"}]
led = solve(roles, msgs, rounds=2)
p = led["publishes"][0]
# branch == "regular"; send_to == ["<all>", "Alice", "Mike"]
# delivered == ["Alice", "Mike"]; history_added == 2
# content_out == "[Message] from User to Alice, Mike: hi"
assert led["roles"]["Alice"]["news"] == [["m1"], []]
assert led["roles"]["Alice"]["idle"] is True
```

## Scoring

15 checks: address delivery and `<all>` broadcast, the four branches (including the profile-vs-name takeover predicate and the dummy skip), text rewriting plus role normalization, direct-chat bookkeeping (busy/idle, private drop that still records history), history dedup separated from delivery dedup, the three `_observe` gates, `observe_all`, the `enable_memory` switch, the memory gate (built with an equal-field anchor), recovery semantics (anchor / idle / already within the last 10), rounds and idleness, the defaults, and 50 randomized role tables with message streams.
