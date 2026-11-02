"""
MetaGPT 消息池路由账。

MetaGPT 的 Environment 是 publish-subscribe 消息池：角色发布 `Message`，
环境按角色的 `addresses` 投递到其私有 buffer，角色再在自己的 `_observe`
里按 `watch/cause_by` 与收件人过滤出一批 news。MGX 版本又在环境层叠了
团队负责人（TeamLeader）转交、公开/私聊、直聊簿记三条规则。

回源 FoundationAgents/MetaGPT main：
metagpt/environment/base_env.py 的 Environment.publish_message /
add_roles / set_addresses / is_idle，
metagpt/environment/mgx/mgx_env.py 的 publish_message / _publish_message /
move_message_info_to_content，metagpt/roles/role.py 的 _watch / _observe /
put_message / is_idle，metagpt/memory/memory.py 的 add / get / find_news，
metagpt/utils/common.py::is_send_to，metagpt/schema.py 的 Message /
MessageQueue.pop_all。

solve(roles, messages, tl_name="Mike", is_public_chat=True, rounds=1,
      all_tag="<all>", no_one_tag="no one")

roles：[{"name", "profile", "cls", "watch": [cause_by...], "addresses"
（None → {cls, name}）, "enable_memory", "recovered", "latest_observed",
"observe_all"}]；messages：[{"id", "content", "sent_from", "send_to",
"cause_by", "role", "metadata", "publicer", "user_defined_recipient"}]。

返回 {"publishes", "roles", "history_ids", "direct_chat_roles"}：
每次发布记分支、最终收件人、实际投递、直聊簿新增、history 增量、转成
文本后的 content/role；每个角色记逐轮 news、memory、残留 buffer 与最终
是否空转。

域约束（ValueError）：缺团队负责人、角色重名、轮数为负、直聊收件人
不是角色。
"""


def solve(roles, messages, tl_name="Mike", is_public_chat=True, rounds=1,
          all_tag="<all>", no_one_tag="no one"):
    raise NotImplementedError
