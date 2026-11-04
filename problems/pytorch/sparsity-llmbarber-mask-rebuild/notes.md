# 解题笔记

## 思路

官方 `mask_rebuild`（lib/opt.py L281-388）的确定性核心，按四个正交旋钮展开：

1. **metric 选择**（WD/W/D）——决定重要性怎么算；
2. **find_threshold**——growth/prune 双集合配对定双阈值；
3. **granularity**（block/layer/output1/input1）——阈值作用域；
4. **retain_reconstruction**——combined 权重的构造方式。

判题域取 `mask_rebuild` 全函数；上游依赖（block 前向 MSE 反传的 grad² 滑动平均 = `OPTAttentionGPT/OPTMLPGPT.add_batch`、SparseGPT 剪枝产生初始稀疏权重）按纪律显式入参。官方 N:M 结构化分支（逐行 threshold + 块内 `ii % prune_m == 0` 交错 topk）不实现，在文档标注边界。

## 关键实现点

- **find_threshold 的四个分支**（官方逐字复刻）：
  - 空集合 → `(inf, -inf)`，两个严格比较都不命中 → mask 不动（可观测：`thresholds` 里就是 inf/-inf，`rebuilt == combined` 经 mask 清零后全 0 或全保留）；
  - `idx = int(recons_num * threshold)`，`recons_num` 用**严格 >** 计数（差恰 0 不计）；
  - `idx == numel` 越界保护取末元素——注意两侧 numel 不同（growth 侧可能比 prune 侧短），分别独立判断；
  - 配对是**位置对齐**（growth 降序 vs prune 升序），不是同 rank 值比较——构造测试数据时要用 `gs[i] - ps[i]` 验证而不是想当然的「第 i 大 vs 第 i 小同值」。

- **交换顺序**：`mask[growth]=False` 先、`mask[prune]=True` 后。由于 growth_indices 在 metric > gt、prune_indices 在 metric < pt 且 gt ≥ pt（由配对构造保证 gt 从 growth 高位取、pt 从 prune 低位取），两集合交集通常为空；顺序语义上仍按官方先后写。

- **retain_reconstruction 的作用路径**：存活位 ref==weights 时（fixture 初版踩过这个坑）retain 是 no-op；只有存活位 ref ≠ weights（SparseGPT 重构后的权重）时 combined 才分化。rebuilt 只经 mask 清零，存活位取 combined——所以 retain=True 与 False 的 rebuilt 可能相同，分化点在**被剪位的 metric**（进而阈值与 mask）。

- **内部精度**：metric 全程 float64。combined 侧的 `.double()` 实为冗余（abs/where 无舍入、float32→float64 精确），scaler 侧的 `.double()` 才是唯一舍入点（sqrt）——变异测试实证：只去 combined 侧等价（剔除），只去 scaler 侧被抓（float32 sqrt 舍入进 float64 乘积）。

- **阈值返回**：`thresholds` 按 name 返回 `(gt, pt)`；output1/input1 返回逐行/逐列数组——测试对数组阈值用 `np.allclose`、标量用 `==`。

## 测试设计（12 例）

- 全对拍：layer/block/output1/input1 × retain 两态 × metric 三选，numpy oracle 独立复算 find_threshold 与记账；
- 交换可观测：revived（ref==0 且 rebuilt≠0）或 newly_pruned（ref≠0 且 rebuilt==0）至少一个发生；mask 位 rebuilt 必为 0；
- 空侧：全剪/全存矩阵 → inf/-inf → mask 不动（rebuilt 全 0 / 全保留）；
- 严格 > 配对：growth=[3,3]、prune=[5,4,3] 首对差恰 0 → recons=0（>= 变异会得 recons=1，pt 3→4）；
- 越界守卫：growth 3 个 9 全大于 prune 2 个 1 → idx=2==prune 数 → pt 取末元素 1.0（去守卫变异 → IndexError）；
- threshold=0/0.5/1.0 交换数单调不增；等值边界（metric==gt 不复活、metric==pt 不剪掉）；
- 纯函数：三个 dict 的张量不被修改，mask 是新张量；
- 校验：dict 空/键类型/键一致/形状/负 scaler/metric/granularity/threshold（bool、1.5、-0.1、inf、str 均拒；int 0 合法）。

## 变异收口

52 变异全 CAUGHT + 1 等价剔除（m-comb-f32：combined 侧 `.double()` 去掉后 abs/where 无舍入、scaler 侧 double 广播保精度，数值完全等价）。设计上每个分支至少一个专属用例：metric 三选互异（scaler 非常量）、路由四支互异（block 阈值 ≠ layer 阈值）、`m-order`（交换顺序）靠重叠位置断言、消息变异靠子串断言。

## 工程学备注

- fixture 初版 `ref = w * keep` 使存活位 ref==w，retain 退化为 no-op——mock 重构权重必须让存活位与稠密值不同（乘 0.9）。
- numpy oracle 初版把更新后的 mask 只写进局部变量没写回 dict，导致对拍假失败——oracle 与实现是两套代码，调试时先怀疑 oracle 自己。
- scaler 因子取无理数（π）提高 float32/float64 舍入分化概率，服务于抓 float32 精度类变异。
