# 解题说明 · KVLink 段落掩码与 link token 布局

## 一句话

文档各自独立预计算 KV，靠「同段互看 + segment 0 目标全局可见」的掩码把不同文档隔开，再用每段末尾的 link token 把信息重新接到最终问题上。

## 算法步骤

1. 写 `segment_ids`：`prefix_len` 个 0；对第 `j` 个 chunk 依次写 `len_j` 个 `j+1`、`reencode_num` 个 0；`pad_to` 非空时尾部补 `-1`；
2. 建 `(L, L)` 的 float32 `bias`，先全 0；
3. 严格上三角（`j > i`）置 `-inf`；
4. `target == -1` 或 `source == -1` 的位置置 `-inf`；
5. 目标 segment 非 0 且与 source 不同段时置 `-inf`；目标 segment 为 0 时保持可见。

第 5 步的**方向性**是本题核心：`zero_mask` 只看 target，不看 source。

## 源码级细节

上游 `UCSB-NLP-Chang/KVLink`（master）：

- `src/data/titan_preprocessor.py::make_segment_mask`（L591-662）：
  - L614-615：`min_dtype = -float("inf")`；L618-627：`torch.triu(torch.full((B,S,S), min_dtype), diagonal=1)`；`add_causal_lm_mask=False` 时（L628-633）换成全 0；
  - L579-588 `_make_bool_segment_mask`：`torch.eq(source.unsqueeze(-2), target.unsqueeze(-1))`；
  - L642-645：`target_is_zero = (target_segments == 0).unsqueeze(-1)`，`zero_mask = target_is_zero`——**只按目标判断**；
  - L649-652：`source_invalid_mask = (source_segments == -1).unsqueeze(-2)`、`target_invalid_mask = (target_segments == -1).unsqueeze(-1)`，`invalid_mask` 为二者之或；
  - L654-656：`all_masks = invalid_mask | ((~bool_mask) & (~zero_mask))`，`masked_fill_(all_masks, min_dtype)`。
- 布局来源：`process_xsum`（L513-548）L530 `segment_ids = segment_ids + [j+1]*len(chunks[j]) + [0]*self.reencode_num`；推理脚本 `scripts/evaluation/wiki_eval.py::preprocess_fn`（L113）L132 同样处理，L134-135 为每段拼 `reencode_num` 个特殊 link token。
- link token 数默认 5：`scripts/evaluation/wiki_eval.py` L54 `parser.add_argument("--reencode_num", type=int, default=5, help="Number of the link tokens.")`，`titan_trainer_kvlink.py` 的配置也写 `reencode_num=5`。
- pad 语义：`BlockAttnCollator`（`titan_preprocessor.py` L548 起）与 `DataCollatorForGeneration`（`wiki_eval.py` L170 起）都把 `segment_ids` 右侧补 `-1`。
- 仓库 README 指向的 `src/data/titan_preprocess.py` 与 `src/data/attention.py` 实际不存在（前者真名是 `titan_preprocessor.py`，后者的 `construct_biased_attention_matrix` 是悬空 import）；本题只依据真实存在的 `make_segment_mask`，不引 `biased_index` 分支。

## 确定性边界

- `bias` 只有 `0.0` 与 `-inf` 两个取值，float32 可精确表示，测试用 `torch.equal` 逐位比较；不存在容差或舍入问题。
- `torch.triu(..., diagonal=1)` 与 `torch.eq` 都是确定性整数/布尔运算；无随机、无浮点累加。
- pad 必须补在尾部：上游 collator 是右 pad，`segment_ids + [-1]*residual`；本题不允许左侧 pad 或重排。
- 总长为 0 的输入被拒绝；`chunk_len=0` 也拒绝，因为上游 chunk 由非空切片产生。

## 与其他题的区别

- `sequence-pack-pad`：只产出打包后的 `input_ids/position_ids/segment_ids/attention_mask`，**不构造** `(L,L)` 掩码；本题不打包、只做段落偏置。
- `kv-sepllm-separator-cache`：按 token id 挑分隔符做 KV 淘汰；本题按 segment 关系做注意力偏置，不淘汰任何 KV。
- `kv-key-rerotation-rope`：处理跨文档 KV 拼接后的 RoPE 位置重旋转；本题处理**可见性**，不碰位置编码。
- `causal-mask`：纯 causal；本题在 causal 之上再叠加同段约束、segment-0 全局目标与 pad 无效位。

## 验证清单

- `check_example`：题面 6 token 例子，核对 `segment_ids` 与每行可看列。
- `check_matches_oracle`：5 组 chunk/prefix/link/pad 组合对拍独立的 numpy 双层循环实现。
- `check_link_tokens_are_global_readers`：link token 能看它之前的全部 token；非 0 段落的 token 看不到 link token。
- `check_nonzero_segments_are_isolated`：非 0 段落之间互不可见，即使已经越过 causal 下三角。
- `check_causal_and_diagonal`：严格上三角全 `-inf`，合法 token 的对角线恒可见。
- `check_padding_is_fully_masked`：`-1` 的整行整列（含对角）全 `-inf`。
- `check_zero_link_tokens`：`reencode_num=0` 时没有 link token，段间完全隔离。
- `check_prefix_only`：只有 prefix 时退化为纯 causal。
- `check_validation`：chunk 长度非法、负 prefix/link、`pad_to` 过小、bool 冒充 int、总长为 0 均抛 `ValueError`。
