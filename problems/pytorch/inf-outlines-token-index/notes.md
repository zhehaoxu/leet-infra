# notes — inf-outlines-token-index（order 682）

## 回源

- `dottxt-ai/outlines-core` `src/index.rs`（GitHub API contents + base64；主仓 dottxt-ai/outlines 已把 FSM 逻辑外移到 outlines-core，主分支只剩 `tests/types/test_to_regex.py`）。
- 关键源码点：
  - `DFA::new(regex)`（regex_automata）→ `universal_start_state(Anchored::Yes)` 为起点。
  - 主循环 `while let Some(current) = next_states.pop()` —— **LIFO**，不是 BFS；`seen` 初始含 start。
  - token 循环 `for (token, ids) in vocabulary.tokens().iter()` —— Rust HashMap 迭代序不定；**出题时规定为词表插入序**，并把 vocab 顺序写进确定性契约。
  - 逐字节 `dfa.next_state`，遇 `is_dead_state || is_quit_state` → `continue 'token_loop`（整 token 作废）。dead 与 quit 处理完全一致 → 抽象合并为 sink_state 无损（记进题面）。
  - **useful 判定**（带全局缓存 `is_useful_state_cache`）：`is_full_match_state（next_eoi 为 match）|| check_is_intermediate_state（存在 byte class representative 使 next 非 dead）`。byte class representative 与全 256 字节等价（DFA 同类字节转移相同）。短路口径：先 match 后 intermediate。
  - **final 判定**：`dfa.is_match_state(dfa.next_eoi_state(current))` —— eoi 达到 match ⇔ state 本身是 match（对 anchored 语义等价），抽象为 `current in match_states`。
  - **IncompatibleVocabulary**：`!has_valid_transitions && !dfa.is_match_state(current)`；missing_tokens 用 `byte.is_ascii()` 分支（ascii 用 char，否则 `\x{:02x}`）。注意官方判据是 **`byte.is_ascii()`（<128）**，不是"可打印"——控制字符如 \x00 也会输出成原始字符。
  - **eos 自环**：循环结束后给每个 final 补 `eos -> final`；`next_state(eos)` API 恒返回 None（即便 transitions 里有该边）—— API 层与数据层的"矛盾"是官方设计，本题只考数据层。

## 出题抽象（与官方的差异，均已钉进题面/notes）

- regex→DFA 一步不考（Python 无 regex_automata 等价物），DFA 以显式转移表输入：`dict[state -> dict[byte -> next]]` + 未写字节落 sink + sink_state 参数。state 0 = anchored start。
- eoi 语义抽象为 match 集合直接给定。
- vocab 用 `dict[str, list[int]]`：插入序 = 官方 token 遍历序的可确定化替代；一词多 id 共享同一条转移（官方 `for token_id in ids` 全部 insert）。
- 报错改为返回 `{"incompatible": {...}}` dict 而非异常，保证输出确定性。

## 核心账点（测试逐一钉死）

1. **LIFO 处理序影响报错对象**：两个不兼容 state，先 pop 的先报（check_incompatible_dfs_order；变异 pop-fifo 抓）。
2. **死胡同不入队**：非 match 且无 DFA 出边的 state 不 useful → 不入队 → 不报错、不收录；到达它的 token 整体作废（check_dead_end_not_error；变异 push-not-useful 抓：入队后被 pop 会误报 incompatible）。
3. **未到达的 match 不进 final**（check_unreached_match_not_final；变异 final-all-match 抓）。
4. **match 非吸收**：越过 match state 继续推进；match state 的普通 token 边保留，eos 自环共存（check_over_match_continue / check_initial_match）。
5. **token 停在中段 useful state 也收边**（check_stop_at_intermediate）。
6. **含 eos id 的 token 整跳过**；eos 仅以自环形式出现（check_eos_token_skipped；变异 no-eos-skip 抓）。
7. **一词多 id 全建边**（check_multi_id_edge）。
8. **UTF-8 字节级推进**：任何字节落 sink 整体作废；中段 state 不被完整 token 到达则不收录（check_multibyte_token_walk）。**Python 陷阱**：`"\xf0"` 是字符 ð（encode 为 c3 b0 两字节），不是裸字节 0xF0 —— 测试构造时踩过。
9. **missing_tokens 格式账**：byte 升序、ascii/<128 用 chr、否则 `\xHH` 小写两位（check_incompatible_hex_and_order；变异 miss-no-hex / miss-reversed 抓）。
10. **首个不兼容即短路**：报错后整个 Index 不可观察（官方 Err 路径），报错用例只能断言 incompatible dict 本身。

## 变异记录

10 个可观察变异全 CAUGHT（`.mutate.py`）：
`useful-no-match`（useful 丢 match 分支）、`final-all-match`（final 取全部 match）、`no-eos-self-loop`、`eos-loop-to-init`（自环指 0）、`no-eos-skip`、`dead-reset-restart`（落 sink 重置到 0 续走）、`push-not-useful`、`drop-match-valid`、`miss-reversed`（missing 倒序）、`miss-no-hex`。

**等价变异分析（drop-match-valid 第一版）**：官方实现里 match 分支的 `has_valid = True` 与报错条件 `... and !dfa.is_match_state(current)` 互为冗余（双保险）——单独删任一处都不可观察，必须两处同时改才能构造"match state 无 token 边报错"的语义变体。因此保留可观察版（删 has_valid 赋值 + 豁免共存时不可达）确认其为等价，另以 miss-reversed 补足 10 个可观察变异。

## 手算锚点踩坑

- `transitions[2]` 里 "0" 从 state 2 出发是自环（'0'→2），不是回 state 0→1 的边 —— 初版断言错记为 {3:1}。
- str token 的 UTF-8 编码陷阱见第 8 条：最初设计"裸 \xf0 单字节 token"不可能用 str 表达，改为完整 😀 四字节 DFA。
- check_multibyte_token_walk 初版还想断言报错前的边——报错路径整个 Index 不可观察，断言只能落在 incompatible dict 上。

## 与相邻题划界

- `inf-grammar-mask-dfa`（order 458）：正则→DFA 后**运行期单步** mask（prefix 逐步推进）。
- `inf-grammar-trie-mask-resume`（order 681）：xgrammar **伪 trie + LCP 三分支**的跨 token 前缀复用。
- 本题：**编译期**把 DFA×词表压成索引的构建账（useful 过滤、eos 自环、不兼容检测）——运行期查询不再出现。
