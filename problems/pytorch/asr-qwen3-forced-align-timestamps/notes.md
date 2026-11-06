# 踩坑记录（order 831）

## 回源口径

- 仓 `QwenLM/Qwen3-ASR`，`qwen_asr/inference/qwen3_forced_aligner.py`（483 行）。读源码时 grep 工具在该仓 .py 上持续返回空（原因不明），用 Read / python re.search 逐段读。
- **LIS 是非降（`<=`）不是严格递增（`<`）**：相等时间戳仍是正常点。变异 `<` 在 `[5,5,5]` 的 dp / is_normal 上直接抓（fixed 值不变，必须暴露中间量 dp / lis_indices / is_normal 才能抓——只断言 fixed 会让该变异幸存）。
- `parent` 只在 `dp[j]+1 > dp[i]` 严格大于时更新（平分留先）；`max_idx = dp.index(max(dp))` 平分取**最早**。构造 dp 平分数据 `[10,20,10,20,15]`（dp=[1,2,2,3,3]）钉死：取最早 → chain [0,1,3] → fixed [10,20,20,20,20]；取最后 → chain [0,2,4] → fixed [10,10,10,10,15]。
- **段长阈值 ≤2 / >2 是分治点**：≤2 用左右邻值按距离择一（`(k-(i-1)) <= (j-k)`，平分取左）；>2 用 `step=(right-left)/(anomaly_count+1)` 线性插值。分母是 **count+1**（含左右端点），不是 count。
- 左右值从 **`result`（已修复值）** 读而非 `data`——但正常点 `result[k] == data[k]`，这是真等价，变异不立（避免假红）。
- `int(v)` 截断（正浮点向零 = 向下取整），**不是 round**：插值造 12.5 / 17.5 → 12 / 17；`round` 会给 12 / 18，专设用例抓。
- 异常值必须**全部小于左邻正常值**才能构造长度 3 的异常段（否则 999/888 之类会自己接进 LIS 成为正常点，段长变 2 走错分支）——首版手算用 [0,10,999,888,777,20] 得到的实际 normal 集是 [0,1,2,5]，插值分支根本没触发。改用 [0,10,9,8,7,20] 才钉死 count=3 插值。
- `parse` 里 `round(v/1000.0, 3)`：fixed 已 int，`/1000` 恰 3 位小数，删 round 是**真等价**（剔除）；改 round 位数 3→2 可观测（1.666→1.67）。
- `align()` 里先 `parse_timestamp`（输出原始 int 值 dict）再 `/1000 + round(3)`；本题 parse mode 合并两步，但固定值断言同时暴露 fixed 与 spans。
- 日/韩分词走 nagisa / soynlp（判题环境无），Chinese/English 一律 `tokenize_space_lang`——判题只收 space-lang 分支，docstring 钉死。

## 测试设计踩坑

1. **手算锚点必须推 LIS 链**：例7/例8 首版把 `dp.index(3)` 的位置算成 2（实际是 3），lis_indices 与 fixed 全错——干跑一次即暴露，逐层推 dp 后再回填。
2. **`check_masked_timestamps` 的占位数断言**：ids 长度必须 == 2×词数，首版写死 6 配 2 词（4 位）导致 FAIL，改成 `2*len(wl)`。
3. **真等价变异 m-fix-walk 幸存**：把 `j = i` 改成 `j = i + 1`，while 终点不变 → `j-i` 与 `range(i,j)` 都不变，行为完全等价。剔除，换成 `j += 1` → `j += 2`（段边界与 count 都变）才抓。**教训：改循环起点未必改变语义，先验算终点**。
4. LIS parent 变异（`>` → `>=`）在 dp 上不可观测（dp 数组不变），靠 lis_indices 断言 `[0,1,3]` 抓——中间量暴露第二次生效。
5. 单侧缺失（开头无左 / 结尾无右）与距离平分（段长 1 时 `1 <= 1` 取左）分别设独立用例，防"左右都有"分支兜底掩盖。
6. 全链路一致性：encode 的占位 token 数 == masked_timestamps 的取值数 == 2×词数，用组合断言钉死三步衔接。

## mut 设计（20 变异全 CAUGHT + 1 真等价剔除）

- 分词族：撇号不保留 / 数字丢弃 / CJK 不逐字切 / flush 丢 token —— 中英混排 + 标点 + 空段用例；
- 编码族：词间单占位 / 尾部单占位 / 丢 audio 前缀 —— 精确字符串断言 + token 计数；
- NAR 族：掩码取反 / 乘改除 —— 非占位大值不入 + 双 segment_time 对照；
- LIS 族：非降改严格 / parent 平分更新 / max 平分取最后 —— dp / parent / lis_indices / is_normal / fixed 五重暴露；
- 修复族：阈值 2→1 / 距离 `<=`→`<` / 插值分母丢 +1 / int→round / 步长 1→2 —— 段长 1/2/3 三族用例 + 12.5/17.5 分叉；
- parse 族：start 错列 / /1000→/100 / round 3→2 —— spans 逐项精确断言。
