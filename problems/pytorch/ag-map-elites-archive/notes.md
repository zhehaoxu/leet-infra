# notes（ag-map-elites-archive）

## 回源

- 论文：MAP-Elites（arXiv 1504.04909，Mouret & Clune）。原始算法就是「网格 + 每格精英」：
  解被映射到行为描述子空间的一个格子，若该格为空或新解更优则替换。
- 工程侧对应物：AlphaEvolve / OpenEvolve 的 `archive`（`programs` 字典按分格键存最优程序），
  以及 QD 系（CVT-MAP-Elites、CMA-ME）的网格变体；本题只取「网格 + 严格更优替换」这层，
  不做 CVT 分区与梯度归档。
- 本题**没有对应的官方源码可钉**（各实现的描述子归一化与越界处理不一致），因此把分格公式、
  越界丢弃、严格改进、行优先扁平下标四条全部写进题面并作为判据。

## 本题的规范化

- 分格：`slot = floor((v - lo)/(hi - lo) * bins)`，`v == hi` 夹到 `bins - 1`；越界丢弃。
  这两条是不同实现最容易分歧的地方，故显式钉死。
- 扁平下标行优先、最后一维变化最快，使 `elite_score` 的形状与 `bins` 的笛卡尔积顺序一致。
- 替换条件是**严格大于**；同分保留先到者，`best_index` 也取并列时的先到下标。
- `accepted` 只标记「成为该格新精英」，与「是否落进某格」区分开。
- 校验门：`bins` 标量也要 `>=1`（首轮就是漏了这条，导致 `bins=0` 走到 `IndexError`）；
  `ranges` 每项 `lo < hi`；`scores` 不含 `NaN`；`N = 0` 合法。

## 变异

14 条全部 CAUGHT：

```
m-slot-clamp-drop            上界不夹取               → 边界用例抓
m-slot-ge                    夹取条件改成 >           → 边界用例抓
m-outside-gate               越界不丢弃               → 越界用例抓
m-strict-improve             同分也替换               → 同分用例抓
m-flat-order                 扁平下标改逆序           → 多维用例抓
m-bins-len                   列表长度不校验           → validation 抓
m-bin-zero                   每项 >=1 不校验          → validation 抓
m-range-eq                   lo == hi 放行            → validation 抓
m-best-tie                   并列取后者               → 并列最优用例抓
m-best-index                 记录格号而非候选下标     → best_index 断言抓
m-nan-gate                   放行 NaN 分数            → validation 抓
m-bins-bool                  标量 bins 放过 bool      → validation 抓
m-occupancy                  占用数按总格数算         → occupancy 断言抓
m-cells-discard-zero         越界记 0 而不是 -1       → 越界用例抓
```

## 查重

- `ag-got-operations-ledger`(723)：操作链的选择/合并，不是网格档案。
- `ag-reflexion-memory-ledger`(725)、`ag-memorybank-forget-ledger`(722)：记忆与遗忘，
  不做行为描述子分格。
- `ag-lats-uct-ledger`(724)：树搜索的 UCT 选择，没有精英档案替换规则。
- 本题只考「网格路由 + 每格精英替换」这一层，与上述均不重叠。
