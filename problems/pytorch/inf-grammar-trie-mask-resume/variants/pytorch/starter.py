def solve(encoded_vocab, targets, uncertain_token_ids, stop_token_ids=None):
    """逐层转写 xgrammar（mlc-ai/xgrammar）的 token 前缀 trie 构建与
    matcher 前缀增量复用账（tokenizer_info.cc 的 sorted_decoded_vocab /
    trie_subtree_nodes_range 与 grammar_matcher.cc FillBitmaskForStates
    的 LCP 三分支），并把 Earley 状态机抽象为确定性的「目标串前缀闭包」
    byte 状态机：advance(s) 当且仅当 s 是 targets 中某串的前缀。

    === 第一层：TokenizerInfo 伪 trie 构建账 ===

    1. 分类：token ∈ 检测表（stop_token_ids=None 时，官方 DETECTION_STOP_TOKENS
       8 条）或 ∈ 显式 stop_token_ids → stop；token == ""（解码后空串）→ special；
       其余进 sorted_decoded_vocab。
    2. 排序：按 token 字符串字典序；并列（相同 token 字符串）钉死按 token id
       升序（官方 std::sort 不稳定，题面钉死 stable 口径）。
    3. trie_subtree_nodes_range[i]（值语义：子树 = [i, range[i])）：官方单调栈
       逐行转写——栈顶 token 不是当前 token 的「子串命中」（官方用
       `cur.find(top) == npos` 判断，**子串误判 quirk 原样保留**：如栈顶
       "xab" 遇 "xxab"，find("xab")==1≠npos 不弹，range 被错误延长）时弹出
       并赋 range[top]=i；收尾剩余栈全部赋 n。
       官方实现：
           while ((!prefix_stack.empty()) &&
                 (token.find(prefix_stack.top().first) == std::string::npos)) {
             trie_subtree_nodes_range_[top_pair.second] = i;  // 弹出
           }
           // 收尾全部 = size()

    === 第二层：FillBitmaskForStates 前缀增量复用账 ===

    uncertain 列表按 sorted 下标升序遍历（入参给 token id，内部转升序）：

    a. subtree-skip：cur_idx < last_rejected_range（前一被拒 token 的
       trie_subtree_nodes_range）→ 整段子树直接拒（官方 kRejected delta）；
    b. 与 prev_token 求最长公共前缀 lcp：
       - lcp > prev_matched（prev 匹配失败位置之后仍与 cur 共享前缀）
         → 状态机确定性 ⇒ cur 在同一字节处必然失败 → 短路拒
         （官方 `last_rejected_uncertain_range = subtree_range[cur_token_idx]`）；
       - lcp < prev_matched → 回退 PopLastStates(prev_matched - lcp) 步，
         prev_matched = lcp（从公共前缀续推，增量复用）；
       - 否则原地续推。
    c. 从 prev_matched 起逐字节 advance；失败 → 记 walk-rejected 并置
       last_rejected_range = trie_subtree_nodes_range[cur_idx]；
       prev_matched 停在最后成功推进长度（失败时不回退，供下一 token 短路）；
    d. prev_token 每轮更新为当前 token（接受/拒绝都更新；官方 atomic-retry
       置 nullptr 的路径不入题）。

    参数
    ----
    encoded_vocab      : list[str]  已解码 vocab（官方 vocab_type=RAW 语义：
                         字符即字节）；空串 = special。
    targets            : list[str]  byte 状态机的目标串集合（重复无害）。
    uncertain_token_ids: list[int]  待判定 token id（不得含 stop/special）。
    stop_token_ids     : None（用官方检测表）或 id 集合（显式语义）。

    返回
    ----
    dict：
      sorted_token_ids         list[int]  sorted_decoded_vocab 的 id 序列
      sorted_tokens            list[str]
      trie_subtree_nodes_range list[int]  子树 = [i, range[i])
      stop_token_ids           list[int]  升序
      special_token_ids        list[int]  升序
      accepted_token_ids       list[int]  uncertain 中被接受者（按处理序）
      ledger                   list[dict]  每 uncertain token 的处理账：
          token_id / sorted_index / branch（accepted | walk-rejected |
          lcp-short-circuit | subtree-skip）/ lcp（与 prev 的公共前缀长，
          无 prev 为 -1）/ matched_from（walk 起点，非 walk 分支为 -1）
    """
    raise NotImplementedError
