# 解题笔记

## 思路

官方 shizhediao/active-prompt 的 uncertainty_estimation.py（L49-143）+ answer_extraction 的输出端（utils.py L248-294）：

1. **上游不入判题域**：GPT k 次采样、answer_extraction（正则找数字/A-E/yes-no、gsm8k round、last_letters 截尾）——抽取后答案串直接作入参；dataloader 截断（dataset_size>1000 取 1000）与 IO 层不入；
2. **判题域**：occurrence 计数（空答案哨兵 -10086）→ numerical 的 variance / 全任务的 entropy、disagreement → sort_by 三口径排序 → qes_limit 截断。

与现有题划界：
- **775 cot-auto-cluster-demos**：按聚类代表**选题拼 demo**，无采样统计；本题按采样**不一致度选题**（todos ② 对 ① 的划界即此）；
- **rs-self-consistency / rs-ttrl-pseudo-reward-ledger**：多数投票折答案；本题不做投票，只算「不一致度」排序；
- **rs-prm-best-of-n**：步级聚合产答案，本题产「标注优先级」。

## 关键实现点（官方口径逐条钉死）

- **NO_SOLUTION 混账**：空答案进哨兵键 `-10086`，numerical 的 variance 展开时它**原样进方差**（float("-10086")）——「全无解的题方差巨大」是官方真实行为，不是 bug；generic/strategyqa 的 disagreement 也把它算一个 distinct 键。
- **strategyqa 双特例**：occurrence 预置 yes/no 零频键（entropy 无贡献但 disagreement 恒 ≥2）+ disagreement 排序按 `abs(yes-no)` **升序**（唯一一个升序口径，其他全 -x 降序）。官方 try/except 兜底 `-len(occurrence)` 在预置键存在下是死分支，原样保留。
- **variance 排序的静默跳过**：官方 `elif sort_by == "variance" and dataset in (...)` 不匹配时无 else——variance + generic 不排序保持输入序，不是报错。
- **scipy entropy 口径**：`-Σ(c/n)·ln(c/n)`、0 频率贡献 0（rel_entr(0,·)=0）、全零 nan。判题环境无 scipy，自实现 `math.log` 对齐（验证锚点：entropy([2,2])=ln2、entropy([5])=0）。
- **np.var 是 ddof=0 总体方差**（非样本方差）。
- **stable 排序**：三个口径的平局都保持输入序（dataloader 顺序）——python sort 天然稳定；`reverse=True` 也保持平局稳定序（不是反转拼接），与 `-key` 恒等（变异 m-sort-tie-rev 因此等价剔除）。

## 测试设计（9 例）

- `generic_oracle`：3 题 k=5 三 sort_by 与 Counter 版 oracle 对拍 + occurrence/disagreement/entropy 零散断言；
- `numerical_variance`：NO_SOLUTION 混入方差（-10086 展开、variance >1.6e7 精确式对拍）+ 全同票方差 0；
- `strategyqa_sort`：abs(yes-no) 升序 + 预置零频不进熵（-(0.8ln0.8+0.2ln0.2)）但进 disagreement（=3）；
- `entropy_ledger`：降序三档熵值精确断言（ln5 / -0.6ln0.6-0.4ln0.2 / 0）；
- `sort_variants`：variance+generic 静默不排序、entropy 降序、numerical+entropy 对拍；
- `qes_limit`：=2 截断子集内排序 / 超限=全部 / 0=全部 / =1 单元素；
- `tie_stability`：disagreement 平局保持输入序 + entropy 平局同；
- `purity`；`args_validation`（类型/空子列表/三枚举/limit 负数与 bool 与浮点 + 消息）。

## 变异收口

29 条设计 → **27 条 CAUGHT + 2 等价剔除**（m-limit-cap：`>` 改 `>=` 在 limit==n 时同结果；m-sort-tie-rev：python reverse=True 平局保持原序，与 -key 恒同）。
覆盖：哨兵键丢失/计数不累加/首现计 0/预置键丢失、方差 ddof=1/每键只展开一次/numerical 键扩大、熵换底/不归一化/0 频 crash/正负翻转、排序六变异（dis 升序/sqa 差值反转/sqa 特例路由删除/variance 条件放宽/entropy 升序）、limit 两变异（0 语义丢/dataset_idx 偏移）、校验八变异 + 消息两条。

## 工程学备注

- **变异脚本片段缩进**：776 solution 的校验循环体缩进 12 空格（在 `for i, samples in enumerate(...)` 内），变异串写 16 空格会 count=0 断言失败——替换串从文件实际行拷贝。
- **scipy 熵锚点先钉死再自实现**：entropy([2,0,3])=0.673、entropy([0,0])=nan（全零不是 0）——0 频率与全零是两个独立边界，前者进账（strategyqa 预置键），后者只在 num_trails=0 出现（测试避开的 nan 语义在实现里保留）。
- 手算断言先跑 oracle 再写期望值（variance >2e7 是拍脑袋，实际 1.63e7；entropy_ledger 的期望顺序写反）——两处假失败全因手算不验算。
