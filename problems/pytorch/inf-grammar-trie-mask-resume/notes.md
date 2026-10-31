# notes — inf-grammar-trie-mask-resume（order 681）

## 回源

- 权威源：mlc-ai/xgrammar main 分支（2026-11 抓取）。`cpp/tokenizer_info_impl.h`（sorted_decoded_vocab / trie_subtree_nodes_range / TokenCharCounts 字段注释「A pesudo-trie. trie_subtree_nodes_range[i] stores how many nodes there are in the subtree.」——注释说的是节点数，**实际实现存的是下标边界**（子树 = [i, range[i])），注释与实现不符，题面按实现口径）、`cpp/tokenizer_info.cc` Impl 构造（分类 + std::sort + 单调栈）、`cpp/compiled_grammar_impl.h`（AdaptiveTokenMask::StoreType 现为 kAccepted/kRejected/kAcceptedBitset，USE_BITSET_THRESHOLD=1000——旧版的 kTree/kPrefix store 已重构掉，本题只取 matcher 消费侧）、`cpp/grammar_matcher.cc` FillBitmaskForStates（Step 2.1 LCP 三分支 / Step 2.2 walk / subtree skip 的 lower_bound 跳段 / prev_token 每轮更新 / atomic-retry 置 nullptr 不入题）。
- 检测表 DETECTION_STOP_TOKENS 8 条逐字转写（LLaMA2 `</s>`、LLaMA3 `<|end_of_text|>`/`<|eot_id|>`、Phi-2 `<|endoftext|>`、Gemma `<eos>`/`<end_of_turn>`、DeepSeek-V2 `<｜end▁of▁sentence｜>`、`<|eos|>`）。
- **官方 quirk（钉进题目）**：trie 弹栈条件用 `token.find(top) == npos`（子串包含）而非前缀判断——栈顶 `xab` 遇 `xxab` 误判为前缀不弹，range 被延长。matcher 后果：被延长的伪子树把本可接受的 token 整段 subtree-skip（过拒）。原样保留并出锚点用例。
- **官方注释/实现不符**：头文件注释说 range 值是「子树节点数」，实现是「下标边界」；按实现出题。
- 排序并列（相同 token 字符串）官方 std::sort 不稳定——题面钉死 id 升序（stable），notes 记录偏离。
- Earley 状态机抽象为 targets 前缀闭包 byte 状态机：advance(s) ⇔ s 是某 target 的前缀（含完全匹配；继续推进则失败）。判定只依赖当前串前缀 ⇒ LCP 短路的正确性（同前缀必同失败）在此口径下成立。
- 不入题（记 notes）：char budget / token_char_counts（GetTokenCharCounts 与 GetMaxTokenChars 的 bounded-string 路径）、FindMidTokenExcludeEdgeRow 中途分叉、atomic retry（PopLastStates(prev_matched_size+1) 路径）、kAcceptedBitset 存储与 USE_BITSET_THRESHOLD、vocab_type 解码（BYTE_LEVEL/BYTE_FALLBACK 的 DecodeToken，题面钉死 RAW 语义）。

## 判题面

- solve(encoded_vocab, targets, uncertain_token_ids, stop_token_ids=None) → 7 字段 + ledger(token_id/sorted_index/branch/lcp/matched_from)。
- oracle：独立重写——排序用「稳定排序按 token」（并列自动保持 id 升序）、trie 用 `top not in tok` 改写 quirk、lcp 用 os.path.commonprefix、主循环 while 结构；全字段对拍。另设独立 walk 第二 oracle（每 token 从零 walk，无 quirk vocab 上与增量结果等价）。
- 锚点：trie 构建（a/get/getter/go/gone/set/setter 全 range 手算）；quirk range [2,2] 与其 matcher 过拒后果；get→go 回退续推（lcp=1）；goa 失败后 gone 复用前缀（lcp==matched 不回退）；lcp 短路（lcp=3>2 且子树外）+ 短路置 range 使子树内 token 被 subtree-skip；abc 失败后整棵子树 skip；range 值边界（range[ab]=1 恰不含 ac）；检测表 vs 显式 stop 两种模式（显式时 "</s>" 等进 sorted）；空串 special 与 " " 非 special；并列 tie id 升序（含同 token 双 id 的 lcp==matched 直接 accepted）；targets 为空（全拒：首 token m=0 失败）；ValueError 组。
- 惰性弹栈口径：range[j] 赋的是「弹出时刻的 i」，同一 i 可连续弹出多个栈元素（go 弹 get/getter 同值 3）。

## 易错点（出题自踩）

- 手算 lcp("get","go")=1（g 相同、e≠o），错算成 2（与「已匹配长度」混淆）。
- sorted 序里 go 在 getter 之后（getter 是 get 的前缀扩展且 > get），错排会让 range[get] 手算成 2（实际 3）。
- range[abxyc]=4：弹出赋的是「弹出时刻的 i」（处理 aby 时 i=4），错算成 5（把「子树后第一个位置」直接当值——多数情况等价，但同 i 连弹多个栈元素时全部取同一 i）。
- quirk 用例里 xxab 的独立 walk 结果与增量结果相同（都拒）——必须靠 ledger branch 断言钉住；另造 targets 含 xxab 的版本让 accepted 集合也出现 quirk 差异。

## 变异记录（10 条全抓）

sort-tie-desc（并列 id 降序）、stop-detect-miss（检测表条目改字）、special-strip（special 判 strip 后空）、quirk-fix（find 改 startswith——「修正官方 quirk」反方向变异）、range-plus-one（弹栈赋值 +1）、lcp-gt-no-range（短路不置 range）、lcp-lt-nopop（回退丢失）、prev-not-updated（接受时不更新 prev_token）、advance-off-by-one（walk 判定错位一字节）、uncertain-no-sort（不按升序处理）。无等价变异。

## 红线

真 tokenizer/HF 模型、Earley parser 真实现、CUDA kernel 不出；byte 状态机以 targets 显式入参（与 ag-tool-loop-breaker 的 sha256 指纹、inf-grammar-mask-dfa 的 DFA 表同范式）。与 `inf-grammar-mask-dfa`（order 484）划界：单步 DFA mask vs 跨 token 前缀复用/子树跳过；与 `inf-radix-prefix-cache` 划界：KV 缓存复用 vs token mask 复用。纯 Python，无 torch 依赖。
