# Voyager 技能库：同名覆盖、版本化落盘与 top-k 检索账

Voyager（arXiv 2305.16291）让一个 LLM 玩家在《我的世界》里不停写新技能（一小段 `async function`
形式的 JS），写完就塞进技能库，下次干活前按任务描述检索出最相关的几段代码拼进提示词。
`MineDojo/Voyager` 的 `voyager/agents/skill.py` 就是这间库房，本题把它搬成纯函数账。

## 接口

```python
def solve(skills, events, retrieval_top_k=5, dump_files=None, primitives=None) -> dict
```

- `skills`：库房初始内容，`[{"name", "code", "description"}]`，按**插入顺序**。
- `events`：依次发生的事件，两种形态
  `{"kind": "add", "task", "program_name", "program_code", "llm_description"}`
  与 `{"kind": "retrieve", "query", "distances"}`（`distances` 与**当前**技能列表一一对应）。
- `dump_files`：落盘目录里已存在的文件名（含扩展名）。
- `primitives`：控制原语代码段，拼提示词时垫在技能后面。

返回 `order`（技能名插入序）、`skills`（name → `{code, description}`）、`dumps`（每次落盘的结果）、
`retrieved`（每次检索的结果）、`skipped`、`programs`（拼好的代码串）、`count`、`synced`。

## 输入输出

```
输入：skills=[{name:"mine", code:"M1", description:"..."}]
     events=[{kind:"add", task:"Deposit useless items into the chest at 0,64,0",
              program_name:"deposit", ...},
             {kind:"add", task:"Craft a wooden pickaxe", program_name:"craft", ...},
             {kind:"add", task:"Mine iron", program_name:"mine",
              program_code:"M2", llm_description:"mine iron"},
             {kind:"retrieve", query:"dig", distances:[0.9, 0.1]}]
     retrieval_top_k=2, dump_files=["mine.js"]
输出：order     = ["mine", "craft"]        # 覆盖不改位置，新技能追加到尾部
     dumps     = [{"event":1, "program_name":"craft", "file":"craft", "rewritten":False},
                  {"event":2, "program_name":"mine",  "file":"mineV2","rewritten":True}]
     skipped   = [{"event":0, "program_name":"deposit"}]
     retrieved = [{"event":3, "query":"dig", "k":2,
                   "names":["craft","mine"], "codes":[...]}]
```

## 规则

- **跳过闸门**：任务的 `task` 以 `Deposit useless items into the chest at` 开头就整条丢弃，
  只在 `skipped` 里留个记号——不落盘、不改库、不产生新名字。前缀必须**逐字**匹配（大小写、
  空格、末尾 `at` 都要对上）。
- **同名覆盖**：`program_name` 已在库里时，条目**内容换新但位置不动**，`count` 不变；
  落盘名不再是原名，而是从 `V2` 起逐个试 `"<name>V<i>.js"` 找第一个**不在** `dump_files`
  里的下标（`mine.js`、`mineV2.js` 都占了就是 `mineV3`）。
- **新技能落盘**：直接用 `program_name` 当文件名，**不**做 V 号探测——即使
  `craft.js`、`craftV2.js` 已经存在也不管。
- **描述模板**：技能描述不由 LLM 原文直出，而是包成
  `async function <name>(bot) {\n    // <llm_description>\n}`——注意四空格缩进与收尾花括号。
- **检索条数**：`k = min(当前技能数, retrieval_top_k)`；库空时 `k = 0` 且直接返回空表，
  不去看 `distances`。排序是「距离升序，距离相同按插入序」，取前 `k` 个。
- **programs 拼接**：先按插入序拼所有技能代码，再拼控制原语，**每段**后跟 `"\n\n"`；
  全空时是空串。
- **校验**：`skills` 内 name 重复、`retrieval_top_k` 非正整数、`distances` 条数与当前技能数
  不等、事件类型未知、`distances` 里出现 `bool`，一律 `ValueError`。

## 提示

- 「覆盖」与「新增」在 `count` 上不可分辨，只能看 `order` 有没有变长——这是本题最容易写错的一处。
- V 号探测只看**落盘文件名**，不看库里有没有同名技能：一个已经存在 `mineV2.js`、
  但库里没有 `mine` 的目录，第一次落盘仍然是 `mine`（因为走的是「新技能」分支）。
- `programs` 的最后一段也带 `"\n\n"`，所以非空结果永远以两个换行结尾，别 `rstrip`。
