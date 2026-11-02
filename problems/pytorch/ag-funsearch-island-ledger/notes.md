# notes（ag-funsearch-island-ledger）

## 回源

- 官方实现 `google-deepmind/funsearch`，`implementation/programs_database.py`（297 行）与
  `implementation/config.py`：
  - `ProgramsDatabase.__init__`：每个岛一个 `Island`，另有 `_best_score_per_island` /
    `_best_program_per_island` / `_best_scores_per_test_per_island` 三个平行列表。
  - `_register_program_in_island`：先 `register_program` 进岛，再按 `_reduce_score` 更新岛内最优。
  - `register_program`：`island_id is None` 时广播到所有岛；随后按 `reset_period`（真实墙钟）判重置。
  - `reset_islands`：`np.argsort(best + randn * 1e-6)[:num_islands // 2]` 取弱岛，清空后
    `np.random.choice(keep_islands_ids)` 抽创始人，再 `_register_program_in_island` 克隆。
  - `Island.register_program`：`_get_signature` 作簇键；新键建 `Cluster(score, program)`。
  - `Island.get_prompt`：`temperature = init * (1 - (num_programs % period) / period)`，
    `_softmax(cluster_scores, temperature)`，`functions_per_prompt = min(len(clusters), fpp)`，
    `np.random.choice(..., p=probabilities)` 有放回取样，`np.argsort(scores)` 排序后编号。
  - `Cluster.sample_program`：`normalized = (lengths - min) / (max + 1e-6)`，`softmax(-normalized, 1.0)`。
- 论文侧只用于定「这是 FunSearch 的机制」：arXiv 2306.12424（FunSearch）与 2506.13131 的岛屿式
  变体；论文本身不给池管理细节，故一切都以官方 `programs_database.py` 为准。

## 本题的规范化

- **随机段全部显式化**：`events` 里的 `island`（prompt 用哪个岛）、`clusters`/`programs`
  （取样结果）、`founders`（创始人岛）都是入参，不含任何 `np.random` 调用。
- **`np.argsort` 不稳定** → 版本排序归一化为「按 `(簇分, 取样位置)` 升序」。`sorted` 本身稳定，
  所以同分时天然按取样位置升序，与「显式加位置作次级键」等价。
- **重置噪声**：官方给岛内最优加 `1e-6` 高斯噪声破平，本题归一化为「按 `(岛内最优, 岛号)` 升序」，
  同分取小岛号。
- **`scipy.special.softmax` 的 argmax 再归一化**（`result[argmax] = 1 - sum(others)`）在
  float64 下只改动 1e-17 量级，测试用 1e-12 相对容差比对，故该步**不纳入判据**（记在此处备查）。
- **墙钟 `reset_period`** → 改成显式 `{"kind": "reset"}` 事件；`_last_reset_time` 不进账本。
- **代码改写**（`rename_function_calls`、docstring 替换、`Program.__str__`）不考，只把版本名
  与 docstring 作为账本字段输出。
- 显式钉死的边界：空岛取提示词报错（否则会静默产出空提示）；`num_islands=1` 时
  `1 // 2 == 0`，`founders` 必须为空；创始人必须落在保留岛**且**手上有程序（两个条件都查，
  因为被重置的岛也可能有程序）。

## 变异

35 条全部 CAUGHT；1 条实证等价剔除：

```
m-sig-unsorted          簇键取 dict 值序而非测试名序     → 多测试用例抓
m-sig-desc              簇键按测试名降序                 → 多测试用例抓
m-reduce-max            簇分取 max 而非最后插入键        → 题眼用例抓
m-reduce-first          簇分取第一个键                   → 插入序用例抓
m-reduce-lastkey        簇分取字典序末位键               → 插入序用例抓
m-cluster-score-overwrite  后来者改簇分                  → 冻结用例抓
m-best-ge               同分也替换最优                   → 同分用例抓
m-num-programs-stuck    计数器不涨（温度不衰减）         → 温度用例抓
m-broadcast-first-only  广播只发第一个岛                 → 广播用例抓
m-temp-fixed            温度恒为初值                     → 温度用例抓
m-temp-plus1            相位偏一                         → 温度用例抓
m-take-no-min           不夹到簇数                       → 少簇用例抓
m-version-plus2         版本号加二                       → 版本断言抓
m-header-doc-index      空壳头的 docstring 指错版本      → 版本断言抓
m-sort-version-desc     版本按簇分降序                   → 排序用例抓
m-prog-denom-nudge      分母去掉 1e-6                    → 比值用例抓
m-prog-denom-span       分母改成 max-min                 → 比值用例抓
m-prog-sign             负号丢掉（长程序更受宠）         → 概率单调性抓
m-reset-min-one         num_islands=1 也重置一个         → 单岛用例抓
m-reset-strongest       重置最强的一半                   → 重置用例抓
m-reset-keep-swapped    保留 / 重置两半互换              → 重置用例抓
m-reset-founder-any     不校验创始人在保留岛             → 新增用例抓
m-reset-founder-empty   不校验创始人手上有程序           → 新增用例抓
m-reset-count-check     不校验 founders 数量             → 负数用例抓
m-reset-keep-counter    重置后计数器不清零               → 重置用例抓
m-prompt-empty-ok       空岛放行                         → 空岛用例抓
m-prompt-len-check      只比长度不比起数                 → 起数用例抓
m-int-bool              整数校验放过 bool                → validation 抓
m-real-bool             实数校验放过 bool                → validation 抓
m-real-finite           不校验有限性                     → validation 抓
m-temp-nonpos           只拒负数不拒零                   → validation 抓
m-kind-check            不校验 kind                      → 未知 kind 用例抓
m-scores-empty          空 scores 放行                   → validation 抓
m-softmax-ignore-temp   概率忽略温度                     → 温度比值抓
m-softmax-no-norm       概率不归一化                     → 和为一用例抓
```

等价变异（不入清单）：`normalized = (x - lo)/denom` → `x/denom`。softmax 对整体平移不变，
而分母只依赖 `max`，减不减 `min` 把概率整体挪一个常数，输出逐位相同（已用 `_eq` 全量比对实证）。

## 已知坑

- 首轮 `check_softmax_over_clusters` 用 `exp(2.0 / 0.5)` 当期望，忽略了温度已被
  `num_programs % period` 从 `0.5` 削到 `0.49996…`，比值差在第 5 位——要从 `rec["temperature"]`
  取实际温度再算。
- `m-prog-no-min` 之所以「抓到」，前提是时间温度那一步先把 `temperature` 打进账本；
  否则 softmax 平移不变会让它变成等价变异。
- `-inf` 与 `-inf` 相减得 `nan`，比较函数必须**先判相等**再算容差，否则空岛的最优分会对拍失败。
- 广播种子让每个岛都有程序，于是「保留岛没程序」这条分支必须先构造「只在部分岛注册」的用例
  才能覆盖；同理「创始人指向被重置的岛」需要被重置的岛手上**有**程序才可观测。

## 查重

- `ag-map-elites-archive`(745)：质量-多样性网格档案，按行为描述子分格、每格留精英；本题是
  **按分数向量聚类 + 温度采样 + 岛屿重置**，簇内还留全部成员而非只留最优。
- `ag-self-instruct-filter`(747)：合成指令的过滤与相似度去重，池子线性增长；本题池子分岛、
  有重置与克隆，且取样概率是显式的温度 softmax。
- `ag-generative-memory-retrieval`、`ag-memorybank-forget-ledger`：记忆检索与衰减，不涉进化池。
- `opt-*` 与 `inf-cache-aware-routing`：调度 / 路由口径，无「簇 + 岛 + 重置克隆」结构。
- `kv-morphkv-island-fusion` 只是名字里有 island，考的是 KV 页融合，与本题无关。
