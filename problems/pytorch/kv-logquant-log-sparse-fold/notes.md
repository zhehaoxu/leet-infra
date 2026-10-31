# 解题说明 · LogQuant 全精度索引折叠

## 一句话

两个整数列表 + 两条阈值：预填充按 `>= 3W` 反复「整段前 `2W` 个取偶数位」，解码按 `> 2W` 把最旧的 `W` 个并进同一列偶数位抽样；没有一行浮点运算。

## 为什么是这套账

「log-distributed」指的是**越老的位置被抽得越稀**：每一轮折叠都对上一轮的抽样结果再取一次偶数位，头部间隔每轮翻倍，刚从 `local` 吸进来的尾段还是稠密的。

- `L = 20, W = 4`：折叠 3 轮（`20 → 16 → 12 → 8`），结果 `[0, 8, 12, 14]`——相邻间隔 8 / 4 / 2，越靠后越密。
- `L = 50, W = 4`：折叠 10 轮，结果 `[0, 36, 40, 42]`，头部一步就跳到 36。
- `L = 12, W = 4`：只折 1 轮，`[0, 2, 4, 6]`。

轮数有闭式：`L >= 3W` 时是 `floor((L - 3W) / W) + 1`（每轮 `len(tmp)` 恰好减 `W`），`L = 20, W = 4` 代进去得 3，与实测一致。

解码段是同一件事的增量版：`log` 一直按 `2^r` 间隔抽稀，`local` 只装最近 `W`~`2W` 个连续位置。触发条件写成 `> 2 * W` 而不是 `>= 2 * W`（上游 L977），所以 `local` 最满时能同时装到第 `2W+1` 个才动手，砍掉 `W` 个后回到 `W + 1`，一步之内不可能连续触发两次。

## 四个易错口径

- **整段取偶数位，不是各抽一半**。`tmp = log + local` 先拼再 `tmp[0:2W:2]`；先折 `log` 再折 `local` 会丢边界位置。`solve(20, 0, 4)` 的第一轮 `log = [0,2,4,6]`、`local = [8..19]`，第二轮取的是 `tmp[0:8:2] = [0,4,8,10]`——`tmp[4] = 8`、`tmp[6] = 10` 都是从 `local` 借来的位置。
- **`>= 3W`（预填充）与 `> 2W`（解码）不能混**。用错一个符号，`L = 3W` 不折叠、或 `len(local) = 2W` 提前合并。
- **后置覆盖判据是原始 `L`**：`W < L < 3W`（上游 L969 用 `key_states.shape[-2]`，不是 `len(tmp)`）。`L >= 3W` 折完哪怕只剩 `2W` 个也不覆盖；`L == W` 两边都进不去，`log` 保持空。
- **`log` 为空时首次解码合并不抽样**（上游 L980-981 的 `else`）：直接搬 `local[:W]`。从第二次起才走 `(log + local[:W])[0::2]`。

## 本题自己的口径

- 只做索引调度，不算量化误差、不重建 KV——量化核与 `quant-kivi-kv-asym` 逐行同，故意剥掉。
- `prefill_log_index` / `prefill_local_index` 是**进入解码步之前**的快照，上游没有导出这个中间量，本题加它来钉住折叠过程。
- `prefill_len == 0` 表示「没有预填充调用」，位置从第 0 个解码 token 起算。上游不会出现空预填充（`update` 每步都带 `key_states`），这是本题为了把两个阶段解耦定的口径。
- `prefill_len == 1` 按上游 L975 的 `else` 分支当一次解码步处理，与 `L == 0` 加一个解码步同结果。

## 判分点

5 键 int64 一维张量，全部按升序；`final_fp_positions` 是 `final_log_index` 与 `final_local_index` 的并集去重（两段在语义上不相交，但断言写成集合更稳）。`window_length < 1` 必须报错——`W = 0` 时上游的 `while len(tmp) >= 0` 永远不收敛。

## 复杂度

朴素模拟即可：折叠轮数 `O(L / W)`，每轮切片 `O(L)`，总 `O(L² / W)`；`L` 到几千也瞬时。判题用例控制在 `L <= 400`。

## 出处

- 仓库 `Concyclics/LogQuantKV`（awesome list 的 `[Code]` 链接亲见）；`src/LogQuant.py`：`LogQuantizedCache.update` L950-1046，预填充索引块 L961-973，解码索引块 L976-982，`window_length` 默认 64 在 L853。
- README L1 直接给 arXiv 链接 2503.19950，L21-22 声明「全精度总量 = 3*window_length」（PartialLogQuant 为 2*window_length），Citation 段的 bibkey 是 `han2025logquant`、OpenReview `E26zsSXT3J`、ICLR 2025 SLLM workshop。
- `PartialLogQuantizedCache` 的索引块 L1258-1281 与 `LogQuantizedCache` 逐行相同，README 说的 2W/3W 差别只体现在索引块之外的每层内容缓存切分（L1291、L1341），本题取的是主类这条路径。

## 相关题目

`quant-kivi-kv-asym`（被剥掉的量化本体）、`kv-quest-page-bound`（另一种页级保留）、`kv-homer-chunk-merge`（按内容合并的树状归并，与本题的纯索引抽稀相对）。
