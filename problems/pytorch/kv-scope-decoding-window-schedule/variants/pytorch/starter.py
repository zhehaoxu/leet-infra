import torch


def solve(init_positions, attn, decoding_metric="discontinuous", num_hidden_layers=32,
          delta=15, max_capacity_prompt=512, decoding_recent_size=128,
          decoding_window_size=1024):
    """SCOPE 解码期的窗口预算调度 + 不连续驱逐：模拟四种 decoding_metric 分支。

    输入 `init_positions`（长度 P，预填充压缩后各槽位的位置，顺序即槽位顺序）给出
    每个 head 初始缓存；`attn` 形状 [T, H, P+T]，第 t 个解码步生成位置 P+t 的新 token。
    维护 H 份**位置列表**（head 之间独立，长度恒等），另有两个从 0 起的整数累加器
    `cds`（current_decoding_step）与 `jump`（jump_step）。

    对 t = 0 .. T-1，每步依次：
      1. 每个 head 的列表尾部追加位置 P+t，记追加后的长度为 L。
      2. 预算 dw：
         - "slide"          -> dw = decoding_window_size（恒定，不动 cds）
         - "adaptive" /
           "discontinuous"  -> dw = decoding_recent_size + cds // (delta * num_hidden_layers)，
                               然后 cds += num_hidden_layers（上游是每层调用 +1，一步恰好 L 次）
         - "None"           -> dw = 0，本步不判门不驱逐
      3. 驱逐门：L >= max_capacity_prompt + dw 才可能驱逐。
         "discontinuous" 额外叠一道：jump >= delta * num_hidden_layers 时把 jump 清零并放行，
         否则 jump += num_hidden_layers 且本步**不**驱逐。其余三个分支没有这道门。
      4. 放行则驱逐，keep = dw - decoding_recent_size：
         候选区 = slots[max_capacity_prompt : L - decoding_recent_size]
         对每个 head：按 attn[t][h][position] 降序取 keep 个候选，同分取候选区里靠前的；
         新列表 = slots[:max_capacity_prompt] + 选中的（保持分数降序，不重排）
                  + slots[L - decoding_recent_size :]
      5. 记录本步的 dw、本步结束后的长度、以及本步是否驱逐。

    返回 4 键 dict，值一律 int64 张量：
      window_sizes   : [T] 每步的 dw（"None" 全 0）
      cache_lens     : [T] 每步结束后的长度（各 head 等长，取 head 0）
      evict_steps    : [E] 发生驱逐的步号，升序
      kept_positions : [H, 最终长度] 每个 head 最终缓存里的位置，按槽位顺序

    非法输入抛 ValueError：
      - `decoding_metric` 不是 "None"/"slide"/"adaptive"/"discontinuous" 之一（区分大小写）；
      - 五个整数参数任一不是 int（bool 也算不合法）或 < 1；
      - max_capacity_prompt > P；
      - attn 不是三维浮点张量、宽不等于 P + T、或 T < 1；
      - init_positions 不是一维 / 长度为 0 / 含重复值 / 含落在 [0, P+T) 之外的值；
      - "slide" 且 decoding_window_size <= decoding_recent_size
        （此时 keep = dw - rec <= 0，滑窗分支的 topk 参数非法）。
    """
    # TODO
    raise NotImplementedError
