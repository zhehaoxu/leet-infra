# Darwin Gödel Machine：存档进化的父代选择与入档账

DGM 让编码 agent 改写自己的代码库，改完跑一遍基准，**只有真正变强的自我修改才进存档**；下一轮的
父代从存档里按「分数 × 稀有度」抽出来，每个父代再挑一个自改方向（补空补丁、治随机性、治超长
上下文，或盯住某个未解决的题）。本题把 `jennyzzt/dgm` 的一轮进化搬成纯函数账：候选表、父代概率、
条目分派、编译过滤、入档门槛、满评门槛。

## 接口

```python
def solve(commits, children, method="score_child_prop", size=2,
          update="keep_all", noise_leeway=0.1, num_issues=60,
          num_full_eval=500, parent_picks=None, pad_picks=None,
          draws=None, baseline=None) -> dict
```

`commits` 是存档顺序（必须含 `"initial"`），`children` 是本代子代，每条都是
`{"id", "score", "resolved", "unresolved", "empty", "submitted", "parent",
"metadata_ok", "context_exceeded"}`。返回 `candidates`、`probabilities`、`parents`、
`entries`、`draws_used`、`compiled`、`rejected`、`archive`、`threshold`。

## 输入输出

```
输入：commits=[initial 0.30, a 0.42, b 0.51, c 0.61, d 0.20]
     children=[e 0.65, f 0.90(补丁全空)]
     method="best", size=2
输出：parents = ["d", "initial"]           # 升序取前二，等于取最差的两个
     compiled = ["e"]；rejected = [{id:"f", reason:"no_patch"}]
     archive  = [..., "e"]
```

`best` 这条最反直觉：官方写的是 `sorted(candidates, key=score)[:size]`，**升序**，所以它挑的是
分数**最低**的那几个父代。照抄，不修。

## 规则

- **候选表**：只收 `metadata_ok` 的提交，顺序与存档一致；`children_count` 数的是「存档内父代
  链接」，父代不在候选表里就不计数（那条链接静默作废，子代本身仍是候选）。
- **父代概率**：`sigmoid(10·(score − 0.5))` 归一化；`score_child_prop` 再把每项除以
  `1 + children_count` 后重新归一化。`random` 是均匀，`no_darwin` 恒取候选表最后一条。
- **`best`**：按分数**升序**取前 `size` 个；名额不足时用 `pad_picks` 从已选中的里面补（下标指
  向「已选列表」）。
- **条目分派**：每个父代按序过三道闸门，硬币从 `draws` 里顺序取，**短路的分支不消耗硬币**：
  1. `empty >= 0.1 · (empty + resolved + unresolved)` 成立才抛第一枚，`< 0.25` 派
     `solve_empty_patches`；
  2. 无条件抛第二枚，`< 0.25` 派 `solve_stochasticity`；
  3. 只有 `context_exceeded` 为真才抛第三枚，`< 0.25` 派 `solve_contextlength`；
  4. 都没中：`unresolved == 0` 就记 `skipped`（不抛第四枚），否则抛一枚取下标
     `min(int(coin · unresolved), unresolved − 1)`。
- **编译过滤**：子代依次检查 `metadata_ok` / `resolved + unresolved > 0` / `submitted >= num_issues`，
  失败分别记 `no_metadata` / `no_patch` / `not_enough_tests`（先中的先记，不叠计）。
- **满评门槛**：池子先把**初始分无条件放进去**，再把 `submitted >= 0.9 · num_full_eval` 的存档
  成员分数放进去（于是 `initial` 满评时会被数两次）；取降序第二高（只有一个元素就取它自己），
  最后与 `0.4` 取大。
- **入档**：`keep_all` 收下全部编译通过的子代；`keep_better` 只收 `score >= initial − noise_leeway`
  的（等号算过）。
- **校验**：`size >= 1`、`num_issues >= 0`、`num_full_eval >= 1`、`noise_leeway` 有限（`bool` 拒收）、
  `method`/`update`/`baseline` 必须在枚举里、硬币必须落在 `[0, 1)`、存档必须有 `initial` 且 id
  不重复、`children` id 不重复、`parent_picks` 数量必须等于 `size`、硬币不够用——一律 `ValueError`。

## 提示

- 三道闸门是**短路**的：`empty` 占比不够时第一枚硬币根本不存在，第二道会先把它拿走。少算一枚
  硬币，后面的条目分派就全错位了。
- `children_count` 只数存档内的父子链接，本代 `children` 不参与——它是「历史上被当过几次父代」，
  不是「有几个孩子」。
- 满评门槛里 `initial` 被数两次是官方实现的真实行为（先无条件 append，再在循环里 append 一次），
  不是笔误，照抄。
