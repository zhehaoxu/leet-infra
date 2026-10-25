# 详解：Checkpoint I/O 账、异步落盘与最优周期

## 出处

- **CheckFreq**（Chen 等, arXiv:2301.13327）把深度训练的存盘代价与故障重跑代价写成一条可优化的曲线，
  其闭式沿用的是 Young（1974）/ Daly（2006）的经典结果：最优周期 $\approx\sqrt{2CF}$，$C$ 为一次检查点的
  **阻塞时长**、$F$ 为平均故障间隔。
- **异步保存**见 NVIDIA/Megatron-LM 的 `--async-save`（先把状态拷进 pinned host staging，再交给后台线程落盘）
  与 PyTorch 分布式 checkpoint 的 `torch.distributed.checkpoint.async_save`。
- 状态字节口径直接沿用题库里的 `training-state-memory`（混合精度 + Adam = 16 字节/参数）。

## 核心与易错点

1. **两级带宽要各除一次，别合并**：`copy_ms` 走 HBM→host（PCIe/NVLink，几十 GB/s），
   `write_ms` 走 host→存储（并行文件系统/NVMe，常常只有 1–10 GB/s）。异步的价值就是把**慢的那一级**移出关键路径：
   stall 从 `copy+write` 缩到只剩 `copy`。
2. **`period_ms = checkpoint_interval * step_ms`**：开销占比是 `stall/period`，不是 `stall/step_ms`。
   每步一存（interval=1）时两者相等，判题用这个退化点钉住口径。
3. **`overhead` 可以大于 1**：那表示「存盘比训练还慢」，配置根本不可行；不要 clamp。
4. **槽位占用是 copy+write 而不是只看 write**：pinned buffer 从拷贝开始就被占到落盘结束，因此
   「压不住」的判据是 `copy+write > B*period`（**严格**大于，持平刚好能轮转开）。写成 `write > B*period`
   或把 `>=` 当限流都会在边界用例上翻车。
5. **Daly 公式里的 C 要用「阻塞成本」**：异步时 $C=\text{copy}/\text{step}$，同步时 $C=(\text{copy}+\text{write})/\text{step}$。
   把被重叠掉的 `write` 也算进 $C$，会得出「异步反而该存得更稀」的错误结论。
6. **单位一致**：带宽是字节/**秒**，时长要乘 1000 才是 ms；$C$ 是「阻塞了多少**个 step**」，所以要除以 `step_ms`
   而不是乘。
7. **开方后要 floor 且钳到 1**：`floor` 而不是 `round`；结果 <1 表示「再稀也不如每步一存」，按定义取 1。
8. **分工**：读侧（训练期把参数预取进 HBM）在 `distributed-zero-offload`，重算侧在
   `distributed-sublinear-checkpoint` / `transformer-mem`，本题只管存侧的 I/O 账。

## 对拍口径

参考解走闭合式：`state = n_params*bps`、`stall = copy (+write)`、`writer_limited = copy+write > B*period`、
`optimal = floor(sqrt(2*(stall/step)*mtbf))`。判题 `_oracle` 用四路不同推导：state 按千元素块拆分再合并
（`n//1000*(1000*bps) + n%1000*bps`）、stall 用「同步总时长减去被异步藏掉的 write」、`writer_limited` 用
**逐事件模拟 60 个检查点的 B 路槽位轮转**（只要有一次必须等位就判限流）、`optimal` 把 `step_ms` 除到根号里面
（`sqrt(2*stall*mtbf/step)`）。另设退化断言：同步/异步 stall 之差恰为 `write_ms`、interval=1 时
`overhead=stall/step`、B 与 period 的严格边界表、带宽无穷大时全 0 且周期钳到 1、状态字节减半时各时长减半。

## 复杂度

参考解 $O(1)$。判题的槽位模拟为 $O(60\cdot B)$ 常数级，整套用例毫秒到几十毫秒完成。
