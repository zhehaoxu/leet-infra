# ExpeL 洞见账：规则操作的解析、计数增减与满额淘汰

ExpeL（arXiv 2308.10144）不训练模型，它让 agent 在训练任务上跑一堆成功 / 失败轨迹，两两对比后请
LLM 输出一批**规则层面的操作**（新增、同意、改写、删除），再把这些操作折进一份会自我修剪的规则表：
常被同意的规则爬上来，常被否掉的规则掉到 0 就被剔除。`LeapLabTHU/ExpeL` 的 `agent/expel.py`
就是这本账，本题把它搬成纯函数。

## 接口

```python
def solve(rounds, initial_rules=None, max_num_rules=10) -> dict
```

- `rounds`：逐批的 LLM 复盘输出（原始字符串，一批一轮）。
- `initial_rules`：初始规则 `[(文本, 计数)]`，默认空表。
- `max_num_rules`：**只**用来决定「列表算不算满」，从不裁剪。

返回 `rules`（`[文本, 计数]`，已修剪并排序）、`prompt`（编号渲染）、`log`（每轮的解析 / 实际落地 /
满额标记 / 快照）、`count`。

## 输入输出

```
输入：rounds=["ADD: Check the drawer.\nADD: Wipe the shelf.",
              "EDIT 3: Wipe the shelf first."]
     max_num_rules=1
输出：log[0].parsed  = [["ADD","Check the drawer."], ["ADD","Wipe the shelf."]]
     log[0].rules   = [["Check the drawer.",2], ["Wipe the shelf.",2]]
     log[1].parsed  = [["EDIT 3","Wipe the shelf first."]]
     log[1].applied = []                       # 编号 3 越界（只有 2 条）
     rules          = [["Check the drawer.",2], ["Wipe the shelf.",2]]
     prompt         = "1. Check the drawer.\n2. Wipe the shelf."
```

## 规则

- **解析**：每行按 `(REMOVE|EDIT|ADD|AGREE)( 数字)?: [标签: ]正文` 抽两列。操作名可带一个数字
  （`EDIT 3:`），`ADD` 后面的数字**丢弃**；`标签: ` 前缀（字母 / 空格 / 数字组成的串再接冒号空格）
  会被吞掉。匹配**不锚定**行首，所以 `PADD:` 也会命中成一个 `ADD`。
- **三道过滤门**：正文去空白后必须非空、必须**以句号结尾**、且正文里**不能出现** `ADD`、`AGREE`、
  `EDIT` 这三个子串（`REMOVE` 不在禁词里）。同一行写两条操作会双双丢失——第一列的正文是贪婪的
  `.*`，会把后半句一起吃进来，于是撞上禁词。
- **修复一遍**（不动账，只改操作本身）：
  - `ADD`：若新正文**包含**某条已有规则的全文（旧文本是新文本的子串），整条丢弃。
  - `EDIT`：若新正文包含某条已有规则，改判成 `AGREE <首个匹配序号>` 并用**旧文本**（等价于给那条
    老规则 +1，新文本作废）；若没有编号、或编号大于当前条数，整条丢弃。
  - `REMOVE` / `AGREE`：正文匹配不到任何已有规则就整条丢弃。
- **落账顺序**固定 `REMOVE → AGREE → EDIT → ADD`：REMOVE 计数减 1（列表满额时一次减 3）、
  AGREE 加 1、EDIT 按**编号**（`编号 - 1`）换成新文本并加 1、ADD 追加且**初始计数为 2**。
- **满额判定**：`len(rules) >= max_num_rules + 5` 为满，取的是**本批开始前**的条数；一批里新加的
  规则要到下一批才可能把列表顶满。
- **收尾**：先丢弃计数 `<= 0` 的规则，再按计数**降序稳定排序**（同分保持原有先后）。
- **渲染**：`"<序号>. <文本>"` 逐行拼接、序号从 1 起、无尾换行；空账渲染成空串。
- **校验**：`rounds` 非序列或含非字符串、`max_num_rules` 非整数或为负（`bool` 拒收）、
  `initial_rules` 元素不是 `(文本, 整数)` 二元组，一律 `ValueError`。

## 提示

- 去重是「旧文本 ⊂ 新文本」，方向反了不算重复：老规则更短时被新规则包住才算撞车，老规则更长时
  新规则照样能加进去。
- 同一批里两条一模一样的 `ADD` 都会落地（去重只对照批前的旧账），于是账上出现两条同文本、同为
  计数 2 的规则。
- `max_num_rules` 没有任何截断作用：它只影响 REMOVE 的力度。以为它会裁掉多余规则是最常见的误读。
- 顺序不能换：AGREE 靠**文本**定位、EDIT 靠**编号**换文本；若先 EDIT 把文本换掉，后面的 AGREE 就
  找不到原文了。
