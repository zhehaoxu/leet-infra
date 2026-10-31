# 参考思路：签名、分带、复核、聚簇四段串联

## 实现

四段各自只依赖上一段的输出，串错任何一段都会在下一段暴露，所以本题把五个中间结果全部交出来。

1. **shingle 折叠**：`sid = (sid * 31 + t) % MOD` 是逐位 Horner，与幂次展开
   \(\sum_k t_k 31^{n-1-k} \bmod \text{MOD}\) 同余但不同实现；结果是**集合**，重复窗口不加重数。
2. **签名**：每格取 `min`，空集填 `MOD`。这里没有随机数——置换系数 `(a, b)` 由入参给出，
   因此签名是纯函数。
3. **分带**：第 `k` 带取**连续**切片 `k*rows:(k+1)*rows`，桶内两两成对，跨带取并集去重后排序。
   连续与跨步是两种不同的分带方案，`bands * rows` 相同、位置集合相同，但撞车结果不同；
   下面的推演就是判题用来区分它们的那组数据。
4. **复核与聚簇**：只复核提名到的对，`inter / union >= threshold`（并集为空记 `0.0`），
   然后对通过的边取连通分量，标签与代表都取**分量的最小下标**。

分带方案要能看出差别，签名不能全由 `min(S)` 决定。取 `MOD = 2**61-1` 时
`a = 2**31` 这类系数正好把乘法变成 61 位循环左移，同一个集合在不同置换下的 argmin 会错开：

```text
# multiplication by 2**k modulo 2**61-1 is a cyclic left shift on 61 bits,
# because 2**61 = 1 (mod 2**61-1).  With n = 1 the shingle id is the token.
# PERM_ROT = [(2**31,0), (2**32,0), (2**33,0), (2**34,0)]

A = {2**30}            # k = 30
   k+31 = 61 = 0  mod 61 -> 2**0 = 1
   k+32 = 62 = 1  mod 61 -> 2**1 = 2
   k+33 = 63 = 2  mod 61 -> 2**2 = 4
   k+34 = 64 = 3  mod 61 -> 2**3 = 8            sig_A = [1, 2, 4, 8]

B = {2**30, 2**28}     # k = 30 gives 1, 2, 4, 8 as above; k = 28 gives
   k+31 = 59 -> 2**59   (larger, not the min)
   k+32 = 60 -> 2**60   (larger, not the min)
   k+33 = 61 = 0  -> 1  (beats 4)
   k+34 = 62 = 1  -> 2  (beats 8)               sig_B = [1, 2, 1, 2]

contiguous bands, rows = 2:  (1,2) == (1,2) hit ; (4,8) != (1,2) miss
strided    bands, rows = 2:  (1,4) != (1,1) miss ; (2,8) != (2,2) miss
exact Jaccard = 1/2, so the pair is a duplicate only if banding proposed it
```

参考解用并查集拿分量，但**输出前会把每格标签改成分量的最小下标**：并查集的根编号取决于
合并顺序，写进返回值就变成实现细节泄漏。判题里 `labels[i] <= i` 与 `labels[labels[i]] == labels[i]`
两条通用不变量正是抓这一点的。

离线环境里没有现成实现可比，于是把这套算法**应当**满足的两条数学性质自己跑了一遍：
签名一致率应当逼近精确 Jaccard，分带召回率应当逼近 \(1-(1-s^r)^b\)。

```text
# 140 documents, 40 tokens each, 7 groups of 20 sharing r of their tokens
# (J = r / (80 - r)); 400 random permutations; 9730 pairs.
# signature agreement rate per exact-Jaccard bucket, averaged over pairs:
  exact 0.0000 (8590 pairs)  ->  0.0000
  exact 0.1111 ( 190 pairs)  ->  0.1468
  exact 0.2500 ( 190 pairs)  ->  0.2567
  exact 0.4286 ( 190 pairs)  ->  0.4406
  exact 0.6667 ( 190 pairs)  ->  0.6662
  exact 0.8182 ( 190 pairs)  ->  0.8229
  exact 1.0000 ( 190 pairs)  ->  1.0000
  largest bucket deviation 0.0357 (the 0.1111 bucket, 400-perm sampling noise)

# banding recall over the 190 pairs with exact J = 0.8182, theory 1-(1-s^rows)^bands
  bands  4 rows 10   measured 0.195   theory 0.439   candidate pairs 236
  bands  8 rows  5   measured 1.000   theory 0.974   candidate pairs 524
  bands 10 rows  4   measured 1.000   theory 0.997   candidate pairs 642
  bands 20 rows  2   measured 1.000   theory 1.000   candidate pairs 846
  bands 40 rows  1   measured 1.000   theory 1.000   candidate pairs 1140
  bands  2 rows 20   measured 0.016   theory 0.036   candidate pairs 193
```

第一张表逐档贴合，`rows` 越大提名越严、候选越少，`bands` 越大越松——题面里那句「旋钮」是
真的。第二张表只有 `bands=4, rows=10` 那档实测明显低于理论（0.195 对 0.439）：这一组的
190 个对来自同 20 篇文档，**对与对之间不独立**（一篇文档的签名整体偏向哪一侧，它与谁都撞不上），
有效样本量是文档数而不是对数，误差棒因此比二项估计宽得多。题面不考这个统计量，本题判题
全部走精确对拍，随机性只出现在上面这段自测里。

## 正确性检验

十二条 `check_*`，全部与结构独立的 oracle 对拍：oracle 用幂次展开算 sid、把哈希值排序取首元、
用 all-pairs 逐带元素比较找候选、用邻接表 BFS 取连通分量；参考解是 Horner、边扫边比 min、
建桶、并查集——四条路径不共用任何中间结果。`_verify` 除五元组逐位比对之外还强制：候选与重复
两个列表必须是 `tuple`、`p < q`、去重且字典序升序；`duplicates ⊆ candidates`；
`labels[i] <= i` 且 `labels[labels[i]] == labels[i]`；`kept == sorted(set(labels))`；
短于 `n` 的文档整行等于 `MOD`、其余格值落在 `[0, MOD)`；`docs` 与 `perms` 未被修改。

- `check_example`：题面两个例子的五元组逐位手算值。
- `check_matches_oracle`：23 组 fixture（20 组手工 + 3 组确定性 LCG 语料），覆盖空语料、单文档、
  全同文档、全空文档、`num_perm=1`、`bands=1`、`rows=num_perm`、`perms` 传 list 而非 tuple、
  重复置换、`a = MOD-1`、token 大于 `MOD`、增补平面级大 token、阈值 0.0/0.4/0.5/1.0。
- `check_banding_nominates_and_exact_check_disposes`：`2**30/2**28` 那组旋转签名，`rows=1` 提名两对、
  阈值 0.5 时全不通过；`rows=2` 时两对都掉出候选——提名与判定是两个独立开关。
- `check_true_duplicate_never_patched_in`：真重复但从未被提名，**不许**补进 `duplicates`（抓「顺手
  做成全量两两比对」的答法）。
- `check_empty_shingle_set_rules`：三篇取不出窗口的文档互为候选，`threshold > 0` 时零重复、
  `threshold = 0.0` 时全并成一簇（`0.0 >= 0.0` 取等号）；空文档与非空文档不同桶。
- `check_sid_is_folded_modulo`：`n=1` 时 `MOD` 折成 `0`；`n=2` 时 `(MOD+5, 7)` 与 `(5, 7)` 同为 `162`；
  另给 `(2*MOD+5, 7)` 让「不取模」的答法算出不同集合。
- `check_shingle_set_not_multiset`：`[1,2,1]` 与 `[1,2,1,2,1]` 的 sid 集相同 → `J = 1.0`；
  按多重集会算成 `2/4 = 0.5`，在阈值 1.0 下掉出重复。
- `check_threshold_boundary_is_inclusive`：`2/4 = 0.5` 取等号算重复、`0.5000001` 不算；
  `4999/10000 = 0.4999` 在阈值 0.5 下**不算**重复（先四舍五入再比就错）；`threshold` 传整数 1 等价 1.0。
- `check_clusters_are_transitive`：`0~1`、`1~2` 通过而 `0~2` 只是候选，仍并成一簇；
  唯一的重复边落在靠后的下标上，代表必须是最小下标。
- `check_bands_are_contiguous_slices`：`sig_A = [1,2,4,8]`、`sig_B = [1,2,1,2]`，连续分带提名、
  跨步分带不提名，`rows=4` 时两者都不提名——三个断言把切片方式钉死。
- `check_scale`：300 篇 LCG 语料 × 8 个置换 × 四种 `(bands, rows)` 组合，外加 300 长文档。
- `check_invalid`：38 个非法入参 + 4 个合法边界。阈值那五条必须配 `bands=1, rows=1`，
  否则 `bands * rows != num_perm` 会先抛错，把阈值校验**掩盖**成通过——这一版就修掉过这个坑。

变异测试 28 个 mutant 全部被抓：不取模折叠、base 改 32、小端折叠、多重集 Jaccard、取 max、
取第一个窗口、`a`/`b` 对调、置换不取模、空集签名填 0 与填空、跨步分带、每带少取一行、候选对
不按带去重、候选对不排序、跳过 LSH 全量两两比对、`>` 代替 `>=`、并集长度写成两边之和、
空并集记 1.0、交叉相乘、先四舍五入再比、标签取并查集根、标签取最大下标、`kept` 改成「去掉每对的
第一个元素」、完全不聚簇，以及三条放宽校验（不查 `bands*rows`、负 token 截成 0、阈值夹到 `[0,1]`）。
最后三条之所以能被抓，是因为判题断言「必须抛 `ValueError`」本身，而不是抛错之后的返回值。

## 范围与来源

- **离线没有任何 MinHash/LSH 实现可对照**：`.venv-eval` 里 `datasketch`、`sklearn`、`scipy`、
  `xxhash`、`mmh3`、`datatrove` 全部没装，也没有对应源码。因此本题**不做归属声明**，不复现任何
  具体库的常数与接口：`BASE = 31`、`MOD = 2**61-1`、空集签名填 `MOD`、连续分带、跨带候选去重、
  `inter / union >= threshold` 的浮点判定、`labels` 取分量最小下标，都是**题面自定**。
  真实管线在这些点上各不相同（签名长度、带行划分、是否用 b-bit 压缩、用什么哈希族），
  本题只固定一套自洽口径。
- **算法性质靠自测背书**：上面两张表是本地跑出来的实测数据（`random.Random(20261027)` 造语料与置换，
  140 篇 × 400 置换 × 9730 对），用来确认「签名一致率 ≈ Jaccard」与「分带召回 ≈ \(1-(1-s^r)^b\)」
  在这套常数下成立；判题本身不依赖任何统计量，全部是确定性对拍。
- **与 `eval-shingle-contamination`（order 300）的分工**：那一题算的是评测集对训练集的
  **shingle 包含率与 Jaccard 稠密矩阵**，重点是「分母钉在哪一侧」与命中阈值标记，规模小到可以直接
  两两求交；本题考的是**规模化的近似检索骨架**——定长签名、分带桶、候选提名、精确复核、
  传递聚簇与保留清单，`Jaccard` 只是其中一步的判据。两题的 shingle 折叠方式也不同
  （300 直接用 id 元组，本题把窗口折成单个整数再取模）。
