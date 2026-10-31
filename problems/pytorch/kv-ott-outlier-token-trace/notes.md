# 解题说明 · OTT 低幅值 token 的追踪与回填

## 一句话

一段整数取整、一次门槛比较、一次整表重排、两次广播回填：分数是 `|K|` 跨 head 跨通道的均值，追的是**最小**的几条而不是最大的几条。

## 为什么是这套账

2-bit 组内 min-max 量化最怕「整组都很小」：组里幅值越低，`scale = (max - min) / 3` 越窄，组里其他 token 的相对误差越被放大。OTT 的做法是把幅值最低的 `sink_num` 条 token 留在 fp16 追踪集里，量化流中只留它们共同的均值做占位——反正注意力算分时这些位置会被追踪集里的精确值覆盖回去（`att_qkquant[:, :, :, sink_id] = att_sink`，L269-271），V 侧同理（L312-314 把 `sink_id` 那几列从量化路径清零，L323-324 改用 `sink_v` 加回来）。所以本题只需算清「谁进追踪集、下标是多少、占位值填什么」。

门槛写成 `sink_scores[sink_num - 1]`（L113）而不是 `sink_scores[n - 1]`：追到第 6 条以后，比的是**已追踪集合里第 `sink_num` 小**那条，也就是「最弱的在编成员」。这样新来的必须比当前最差成员更强（更幅值小）才置换得进来，且整表重排后最差的那条沉到表尾等着被下一轮比掉。上限判 `sink_id.shape[0] < sink_max_size`（L114）在**进入之前**，所以一步最多能把追踪集顶到 `sink_max_size + sink_num - 1` 条，上游没有裁剪。

拿题面第 2 步验算：`a = [2.5, 0.25, 4.0, 0.75, 3.0, 1.25, 3.5, 1.75]`，`F = 8`、`residual_length = 1`、`group_size = 2` → `quant_len = (8 - 1) // 2 * 2 = 6`，最后一个 token 留在近因段。`scores = 2 * a[:6] = [5.0, 0.5, 8.0, 1.5, 6.0, 2.5]`，门槛 `sink_scores[1] = 2.0` → 候选 `0.5 / 1.5` 即位置 `[1, 3]`，加偏移 `quantized_len = 4` 得绝对下标 `[5, 7]`。与旧的两条 `[1, 3]`（分数 `1.0 / 2.0`）拼起来按分数升序 → `[0.5, 1.0, 1.5, 2.0]` 对应 `[5, 1, 7, 3]`：旧条目被夹在新条目中间，`sink_id` 从来不是追加序。

## 四个易错口径

- **门槛下标是配置的 `sink_num - 1`**：`sink_num = 2` 追到 4 条之后门槛仍是 `sink_scores[1]`，用 `sink_scores[n - 1]`（表尾最大那条）会让几乎所有候选都进得来。
- **严格小于**：`scores == threshold` 不入选（L113 是 `<`）。等于门槛的位置一旦进来，同分集合会每步自我膨胀。
- **回填是跨 token 均值，不是跨 head 均值**：`chunk[0, :, idx, :]` 取出的形状是 `(H, m, D)`，`mean(dim=1)` 折掉的是那 `m` 个被追踪的 token，每个 head 各留一个代表向量（L139-140）。广播的是同一个向量，所以按分数序还是按下标序取数都同结果——别把它当成 head 平均，也别当成逐 head 逐通道的全局均值。
- **满额那步什么都不做，但仍要返回状态**：`sink_id.shape[0] >= sink_max_size` 时上游直接落进 `else`（L118-119），`sink_indices` 为 `None` → 既不追加也不回填，可 `scores`、`keys_out` 这些量照样得给出来。

## 本题自己的口径

- 只做追踪记账：`triton_quantize_and_pack_along_last_dim` 的 group-wise min-max 与 `quant-kivi-kv-asym` 逐行同，故意剥掉；`sink_k` / `sink_v` 是 fp16 原值收集，只服务 CUDA 侧的注意力覆盖，与「谁被追踪」无关，也剥掉。
- `quant_len` 外面套 `max(0, ...)`：上游 L104 没有这层保护，`F < residual_length` 时是个负数，切片会反向吃掉尾部；调用点 L361 的门槛 `F >= residual_length + group_size` 保证线上不会出现，本题把 `quant_len == 0` 定义成「整步空转、状态原样回吐」。
- 重排用**稳定**升序，同分时旧条目排在新条目前面；上游 `sink_scores.sort()`（L134）没传 `stable=True`，同分相对次序不保证。
- 首步与候选步的「取最小的 k 个」用 `argsort(stable)` 表达，同分取**下标小**的；上游用 `torch.topk(..., largest=False)`（L116、L121），同分次序同样未定义，这里给的是本题的确定性口径。
- `sink_id` 非空却少于 `sink_num` 条 → `ValueError`（上游 L113 会 `IndexError`）；`quant_len > 0` 却 `sink_num > quant_len` 同样报错，因为上游 `topk` 会直接崩。
- `batch` 维钉死为 1：`metric` 与两处回填都只写 `k[0]`，多 batch 时上游静默丢信息。

## 判分点

7 键：`quant_len` 是 int64 零维；`scores` / `sink_scores` / `keys_out` / `values_out` 与输入同 dtype（float32 进 float32 出）；`new_indices` 是 chunk 内下标、按分数升序，`sink_id` 是**绝对**下标、按 `sink_scores` 升序——两者语义不同，别混。输入张量一律不许就地改（chunk 先 `clone()` 再回填）。

## 复杂度

分数一遍 `O(H · quant_len · D)`，排序 `O(quant_len log quant_len)`，回填 `O(H · m · D)`。判题用例控制在 `H, D <= 4`、`F <= 32`，随机对拍另加 `quantized_len` 偏移与已有追踪长度两维。

## 出处

- 仓库 `yisunlp/OTT`（GitHub 检索亲见：「code for paper Accurate KV Cache Quantization with Outlier Tokens Tracing (ACL2025 Main)」，默认分支 `master`）。`models/utils_OTT.py`：`metric` L10-11，`quant_step` L90-166（取整 L104、门槛 L113、上限判定 L114、`topk` L116/L121、绝对下标偏移 L129、整表重排 L134-138、回填 L139-140），注意力侧覆盖 L269-271 与 L312-324，调用门槛 L361-362，浅层关掉追踪 L195-197。
- 默认超参取 `example.py` L17-20：`group_size` 128、`residual_length` 128、`sink_num` 5、`sink_max_size` 32。
- README L1 只给标题、正文没有 arXiv 链接；论文信息取 `HuangOwen/Awesome-LLM-Compression` KV Cache Compression 章节第 2186 行——arXiv 2505.10938，标注 ACL 2025，作者 Yi Su、Yuechi Zhou、Quantong Qiu、Juntao Li、Qingrong Xia、Ping Li、Xinyu Duan、Zhefeng Wang、Min Zhang（arXiv 摘要页亲见，v1 提交 2025-05-16）。`October2001/Awesome-KV-Cache-Compression` 的 Quantization 章节里**没有**这条，别按那份清单的编号引用。
- 仓库自己留的旧名还在：`pyproject.toml` 里包名仍是 `sinkq`，描述写着「Sink-aware 2/4/8 bit KV Cache quantization」，与题面无关，只是提醒追踪集和 attention sink 不是一回事。

## 相关题目

`quant-kivi-kv-asym`（被剥掉的量化本体，也是 OTT 改的那篇 baseline）、`kv-logquant-log-sparse-fold`（另一种「谁留全精度」的调度，纯索引不看幅值）、`kv-capkv-leverage-score`（按重要性留 K 的另一路判据）。
