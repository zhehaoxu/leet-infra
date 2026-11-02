# TravelPlanner 约束沙盒：行程的 8 条常识判据与填充率账

TravelPlanner（arXiv 2402.01622）不训模型也不打分：LLM 吐出一份逐日的行程 JSON，沙盒用两组**硬编码
checker** 判它过不过——`hard_constraint.py` 管偏好与预算，`commonsense_constraint.py` 管「行程说不说得
通」。后者就是本题：八个纯函数拼成的判据表，输入是行程文本与几张数据库表，输出是每条判据的
`(ok, 原因)`。`OSU-NLP-Group/TravelPlanner` 里它是脚本级的（全局 pandas 表 + `os.chdir`），本题把
表、城市-州映射、距离矩阵全部提成显式入参。

## 接口

```python
def solve(query, itinerary, city_state_map, sandbox) -> dict
```

- `query`：`{org, dest, days, visiting_city_number, ...}`——出发城、目的州、天数、要去的城市数。
- `itinerary`：逐天的
  `{current_city, transportation, breakfast, lunch, dinner, attraction, accommodation}`；
  `current_city` 常写成 `from A to B`，景点常写成 `"X, A;Y, A;"`（**分号收尾**）。
- `city_state_map`：`{城市: 州}`，对应上游的 `citySet_with_states.txt`。
- `sandbox`：`restaurants` / `accommodations` / `attractions` / `flights` / `distances` 五张表。

返回 `{"checks": {名字: {"ok", "msg"}}, "failed": [...], "passed": bool}`，八条判据按上游
`evaluation()` 的顺序排列，失败项也按同一顺序收集。

## 输入输出

```
输入：query  = {org:"A", dest:"S1", days:3, visiting_city_number:1}
     day[0] = current_city "from A to B", transportation "Self-driving, from A to B",
              breakfast "R1, A", lunch "R2, A", dinner "R3, A",
              attraction "X, A;", accommodation "H2, B"
     day[1] = current_city "B", transportation "-", breakfast "R4, B", lunch "R5, B",
              dinner "R6, B", attraction "Y, B;", accommodation "H2, B"
     day[2] = current_city "from B to A", transportation "Self-driving, from B to A",
              breakfast "R7, B", lunch "R8, B", dinner "R9, B",
              attraction "V, B;", accommodation "-"
输出：passed = True
     failed = []
     city nodes = [A,B,B,B,A]

输入：day[0].dinner = "R1, A"
输出：failed = ["is_valid_restaurants"]
     checks["is_valid_restaurants"] = {"ok": False,
         "msg": "The restaurant in day 1 dinner R1, A is repeated."}
```

## 规则

- **城市结点**：当天 `current_city` 含子串 `from` 时，用
  `from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)` 拆出两城，否则只取一个；两地都过一遍
  `extract_before_parenthesis`（`^(.*?)\([^)]*\)`，**只剁配对括号**）。拆不出来就抛 `ValueError`。
  所以「三天行程」实际会得到 5 个结点。
- **首日与闭环**：第 0 天用 `from` 写法时，第一个结点必须等于 `query['org']`；全部结点首尾必须相同。
- **序列合法性**：结点数 `< 3` 直接非法；把序列压成连续段后，段首若是**已经出现过的城**且不在首尾
  位置 → 非法；某个城只占一段一个结点（单例段）且不在首尾 → 非法。城市必须在 `city_state_map` 里；
  `days > 3` 时还要每个**中间**结点所属州等于 `query['dest']`。
- **餐厅**：三餐按**整串文本**（含城市）跨天共用一张去重表，同一天午晚餐撞同一家也算重复。
- **景点**：`attraction.split(";")[:-1]` —— 最后一段被丢掉，所以**没有分号收尾的景点等于没填**；
  剩下的按整串跨天去重。
- **住宿**：`accommodation` 键**必须存在**（缺键报 `No Accommodation Info.`，值为 `-` 不算）；
  按 `count_consecutive_values` 压成 run，用 run 长度比库里该住宿的 `minimum_nights`；只有当
  「同名 + 同城」在库里**恰好命中一行**时才比，命中多行就跳过这条判据。
- **交通**：第一天必须非空且非 `-`；类型归一按 `taxi → self-driving → flight` 的**先后**做子串匹配
  （大小写不敏感，认不出则记 `None`）；冲突只有两种：自驾×飞机、出租×自驾。
- **城市归属**：当天每个字段都得提到「当天城市」（子串即可）——交通要**同时**提到两个城，三餐与每个
  景点只需提到其中一个，住宿只认**当天末城**。
- **沙箱查证**：餐厅 / 住宿 / 景点按「名字是库里名字的子串 + 城市**精确相等**」匹配，名字切在
  **最后一个逗号**上（`(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$`，城市再去括号并 strip）；航班号取
  `Flight Number: ` 之后到第一个逗号之间，且起降城市要双匹配；自驾 / 出租的起讫城必须在距离表里。
- **信息缺失**：先做两道短路（`days` 数不满 → `Invalid Days`；除出发地外的城市数不等于
  `visiting_city_number` → `Invalid City Number`）；再要求 6 个字段键存在；跨城日要交通、单城日要
  景点、非末日要住宿、非跨城日要三餐；最后算填充率 = 「非空且非 `-` 的键数 / (6 × days)」，
  `< 0.5` 报 `The absent information is more than 50%.`
- **校验**：`days` / `visiting_city_number` 必须是非 `bool` 整数且 `days >= 1`，`org` / `dest` 必须是
  字符串，`itinerary` 必须是非空 list 且每项是含 `current_city` 字符串的 dict，否则 `ValueError`。

## 提示

- `from ... to ...` 会把一天拆成**两个**结点，这是最容易算错的账：三段行程得到 5 个结点，不是 3 个。
- 序列判据要求「中间城至少连住两天」，所以单日游、两日游（结点数 < 3）**永远过不了**这条判据。
- 填充率的分子按 `unit` 的**全部键**数，`current_city` 也在里面；分母却固定是 `6 × days`。字段填满时
  比例会超过 1，而只空三餐时仍能到 11/18。
- 景点串忘了分号收尾不会报错，它会**静默消失**：不查重、不查城市归属、也不查沙箱存在性。
- 库里的「名字」用子串匹配，名字切出来是空串时（写成 `", A"`）`'' in 任何名字` 恒真，等于任何餐厅都
  算命中。
