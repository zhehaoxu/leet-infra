# 详解：Offload 预取流水线与计算-传输重叠账

## 出处
ZeRO-Offload（2101.06840）把优化器状态与梯度放到 CPU、按层搬运，ZeRO-Infinity（2104.07857）进一步把参数与 KV 放到 NVMe 并靠**预取**掩盖搬运延迟；两者都用「计算与通信重叠 + 少量 staging buffer」这条流水线。本题把这条流水线抽象成单引擎、`buffers` 个槽的确定性时间线——没有真实 DMA 抢占、没有多流并发，只算「预取什么时候能开始、计算什么时候被拖住」。与 `distributed-zero-offload`/`distributed-zero-infinity` 分工：那两题算 offload 的**容量与通信量账**（多少字节、省多少显存），本题算**时间线**（重叠得好不好、空等多少拍）；与 `perf-kernel-launch-timeline` 分工：那题是 kernel 发射与流依赖，本题是 DRAM→HBM 的搬运槽位。

## 核心与易错点
1. **槽位「从预取开始占到计算结束」**：不是只占到预取结束。搬运完成后数据还躺在槽里等计算来读，槽在这期间不能给别人用。漏掉这一点会让 `buffers=2` 的第三层提前开搬，算出偏乐观的 makespan。
2. **两个约束取 max**：预取开始受「引擎空闲」和「同槽旧层算完」两个时刻约束，计算开始受「上一层算完」和「本层预取完成」两个时刻约束，各取 `max`。少取一个就会出现「计算等数据」或「引擎撞车」。
3. **传输耗时向上取整**：`ceil(bytes/bw)`；40 字节 / 30 字节每秒计 2 拍。用整除会把传输算快，makespan 偏小。
4. **`buffers=1` 是校验退化的锚点**：此时完全无法重叠，`stall` 恰好等于全部传输时间之和，`makespan = Σcompute + Σtransfer`。算不出这个恒等式说明槽位逻辑有误。
5. **0 字节层仍受槽位约束**：它不耗引擎时间，但 `buffers=1` 时仍要等上一层算完才能「发起」，所以它的 `transfer_start` 不等于 0——测试里专门钉住这条。
6. **空等 = makespan − Σcompute**：计算本身是串行累加的，任何空隙都只能来自预取没跟上。两条口径（逐层 `c_j - finish[j-1]` 累加 vs 总时长相减）必须一致。

## 对拍口径
参考解逐层推进三个时刻（`engine_free`、`finish[j-buffers]`、`finish[j-1]`），一行公式一层。判题 `_oracle` 不看公式，直接按**单位时间 tick** 推进状态机：引擎每拍搬 `bw` 字节、槽用「占用标记」跟踪、计算与传输在同一拍内并行、0 字节层在同一拍内链式发起；空等由「makespan − Σcompute」涌现。两套结构在 30 组随机用例上逐字段对拍；另设断言：单槽时 `stall` 等于全部传输耗时、0 字节层的 `transfer_start` 仍受槽位约束、`makespan = Σcompute + stall`、每层 `c_j >= e_j` 且计算不重叠、`transfer_start[j] >= finish[j-buffers]`、槽数增加 makespan 单调不增、计算远长于传输时只剩首层预取的空等、空输入退化。

## 复杂度
`O(L)`：每层三次取 max；判题 tick oracle 为 `O(makespan)`，规模很小。
