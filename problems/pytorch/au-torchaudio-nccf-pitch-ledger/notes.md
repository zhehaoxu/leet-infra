# 858 解答笔记：torchaudio NCCF 音高检测账

## 1. 官方回源

### 源码清单与依赖

```text
.workbuddy/scratch/captum855/functional.py   2511 lines / 95067 bytes
.workbuddy/scratch/torchaudio858/official858.py   ast.get_source_segment x5
WANT = [_compute_nccf, _combine_max, _find_max_per_frame,
       _median_smoothing, detect_pitch_frequency]
NS injects only: math typing torch Tensor Tuple Optional List
```

### 逐字执行环境

五个函数整段 `exec` 到共享 `NS`，驱动层为 `comp_nccf` / `comp_combine_max` / `comp_find_max` / `comp_median_smooth` / `comp_pitch`，统一 float64。`comp_pitch` 先跑逐字 `detect_pitch_frequency` 保存 `freq_official`，再用同一批中间量按 float64 重算频率，并把 `nccf` / `indices` / `smoothed` 从 packed `[B, ...]` reshape 回输入 lead 形状。

## 2. 十二段官方机制探针

### 探针结论

P1–P4 钉分帧 meta、pad 恒等式、四次方归一与尾帧补零；P5 钉 `mask = a > thresh * b` 的严格边界；P6–P8 钉 `lag_min`、半窗偏好与两次偏移；P9–P11 钉左侧 replicate pad、unfold 窗口、偶数窗下中位与帧数缩短；P12 钉全链路频率账与批次 unpack。
首轮 3 处断言写错：P8 双校准应是 `0 + 2 + 1 = 3`、P10 `w=30` 输出 25 帧而非 26、P12 频率因 float32 恰为 200.0；均按官方实测订正。P10 顺带记录 `win_length=1` 触发 `torch.cat([])` 崩溃。后续 `comp_pitch` 改 float64 后再复跑一次，P12 改为同时钉 `199.99999995` 与 `freq_official = 200.0`。

## 3. 三层对拍

### official 层

逐字五函数 + torch float64 张量；`_median_smoothing` 的 left replicate pad 与 `unfold` 单独复算，便于对账 `padded` / `roll`。

### reference 层

纯 Python `solution_staged.py`，五个 mode；`pack` / `unflatten` 处理任意前导维，`lower_median` 取 `sorted[(n-1)//2]`，`argmax` 取首次最大。

### oracle 层

独立定义式：单行内核 + `math.fsum` 一次除法、`max((v, -k))` 取首个最大、独立的 `_rows` / `_rebuild`，不复用参考解的任何 helper。

### 对拍结论

```text
VERIFY 858 DONE checks=471 fails=0 ALL GREEN
coverage: nccf x6 / combine x5 / find x5 / median x6 / pitch x4
          includes 1D / 2D / 3D leading dimensions
```

## 4. 判题用例与补丁

### 首轮红灯

首轮 `driver858.py` 12 个用例中 2 红：`check_median_smooth` 把 `w=3, n=7` 的输出误写成 5 帧（实为 6 帧 `[9,8,2,7,3,6]`）；`check_nccf` 误以为 NCCF 尺度不变（波形乘 2 后实为除以 4）。两者都用官方逐字环境复核后订正期望值，参考解与官方逐字环境始终一致。

### 硬化补丁

`patch858.py` 追加 9 个 `check_*`：非整数 ceil 边界（lags / frame_size / num_frames）、全非零波形的逐点 NCCF、标量 / bool / 缺键 / bool 标志的域约束、combine_max 四个数组逐一校验、find_max 奇数 lag 维与 `0.99` / `0.98` 阈值边界、find_max 自身下界、median_smooth 非数值输入、pitch 校验顺序、小采样率下 `freq_high` 必须原样透传。最终 21/21 ACCEPTED。

## 5. 变异测试

### 真等价清单

11 个判真等价：① `pack` 的两个错误分支与 `shape_of([])` 分支被更早的 `require_numbers` 遮蔽（4 项）；② `unflatten` 叶子切片恒为单元素（1 项）；③ `argmax` 初始哨兵 `-1e300` 低于所有判题有限值（1 项，另用全负行实测）；④ `sum(a*b)` 与 `sum(b*a)` 恒等（1 项）；⑤ 少补一个零不改变 dot 与两个范数（1 项）；⑥ 两个平方范数交换除法，400 组确定性扫描 `max |diff| = 6.94e-18`（1 项）；⑦ `find_max` 索引恒为整数，`int(v) == float(v)`（1 项）；⑧ `_pitch` 的 `freq_high` 下界与 `_find_max` 的下界消息完全相同（1 项）。

### 最终结果

```text
PROBE  858 DONE ok=57 fail=0 ALL GREEN
VERIFY 858 DONE checks=471 fails=0 ALL GREEN
DRIVER 858 21/21 ALL ACCEPTED
MUT    858 202 total / 11 equivalent / 191 active / 191 CAUGHT
       0 SURVIVED, 0 ERROR, ALL MUTANTS CAUGHT
```

首轮 202 项中有 37 项 SURVIVED，全部通过追加硬化用例或判真等价收口；其中 `find_max` 原本暴露了一个官方并不存在的 `thresh` 入参，为严格对齐 `_find_max_per_frame` 里硬编码的 `_combine_max(half, best)`，已从参考解移除该入参并改写对应变异锚点。

## 6. 官方 quirk 清单

① docstring 写 `dot / sqrt(E1*E2)`，源码是两次连除平方范数；② 归一化非尺度不变（乘 2 → 除 4）；③ `_combine_max` 严格 `>` 且 `0.99` 阈值在 `_find_max_per_frame` 中硬编码；④ 两次索引校准 `+= lag_min` 与 `+= 1`；⑤ `_median_smoothing` 只左侧 pad、右侧不补，输出比输入短；⑥ 偶数窗取下中位；⑦ `win_length=1/2` 触发 `torch.cat([])` 崩溃；⑧ 频率换算的 float32 降级吃掉 `1e-9` epsilon；⑨ 输出帧数 `n - ceil((w-1)/2)`。

## 7. 查重与复用

落地前扫描确认 `max_order = 857`、858 空闲、835 题 order 唯一、无同名 slug；全库 `meta.json` 检索 `nccf` / `pitch_frequency` / `griffinlim` 零命中。与 838（librosa YIN）按算法族划界，与 855（频谱后处理）、857（波形算子）虽同源于 `functional.py`，但函数集合互不重叠。
