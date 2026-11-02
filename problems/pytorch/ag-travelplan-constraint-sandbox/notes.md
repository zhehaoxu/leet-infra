# notes（ag-travelplan-constraint-sandbox）

## 回源

官方实现 `OSU-NLP-Group/TravelPlanner`，核过三份文件（GitHub contents API + `Accept:
application/vnd.github.raw`，`wc -l` 已验证非空）：

- `evaluation/commonsense_constraint.py`（547 行）
  - `evaluation`(L521-531)：8 条判据的**注册顺序**，就是本题返回的 `checks` / `failed` 顺序。
  - `is_reasonable_visiting_city`(L128-160)：L133 的 `min(days, len(tested_data))` 截断；L136 用
    `'from' in city_value` 判定要不要拆两城；L140 首日 `city1 != org` 就报错；L148 闭环；L155 城市必须
    在 `city_state_map`；L157 `idx not in [0, len-1] and days > 3 and city_state_map[city] != dest`。
  - `is_valid_city_sequence`(L85-124)：L98 `len < 3` 直接 False；L109 重复城（且不在首尾）False；
    L119 `count == 1 and 0 < i - 1 < len - 1`——**`i` 此时是下一个段的起点**，所以 `i - 1` 才是本段起点。
  - `is_valid_restaurants`(L163-194)：三餐共用 `restaurants_list`，命中即返回。
  - `is_valid_attractions`(L196-213)：L204 `unit['attraction'].split(';')[:-1]`，最后一段被丢。
  - `is_valid_transportation`(L215-234)：L217 第一天必须有交通（且被 append 两次）；L231 冲突表只有
    `(Self-driving, Flight)` 与 `(Taxi, Self-driving)` 两组。
  - `transportation_match`(L57-66)：taxi → self-driving → flight 的**短路顺序**，认不出返回 `None`。
  - `is_valid_information_in_current_city`(L236-320)：交通要 `city not in transportation` 逐城检查
    （即两城都得提到）；三餐只要 `any`；景点逐个查；住宿只查 `final_city_list[-1]`。
  - `is_valid_information_in_sandbox`(L323-398)：L330-332 从交通串取起讫、取不到退回
    `current_city`；L340 航班号 `value.split('Flight Number: ')[1].split(',')[0]` + 起讫双匹配；
    L353/356 自驾与出租查距离矩阵，`cost == None` 即无效。
  - `is_valid_accommodaton`(L401-424)：L411 `count_consecutive_values(data)`；L418
    `len(...) == 1 and unit[1] < minimum nights`——**只有恰好一行时才比**。
  - `is_valid_days`(L454-464)：占位串 `You don't need to fill in the information for this or later days.`
    不算一天。
  - `is_valid_visiting_city_number`(L426-452)：L447 `city_set.discard(question['org'])` 后再比个数。
  - `is_not_absent`(L466-518)：L470/473 先短路 `is_valid_days` 与 `is_valid_visiting_city_number`；
    L497-507 的四条逐日规则；L510-515 的填充率 `total_valid_info * 1.0 / (6 * days) < 0.5`，
    分子按 `for key in unit` 数，**`current_city` 也在里面**。
  - L24-25：`city_state_map` 由 `database/background/citySet_with_states.txt` 按 `\t` 切出。
- `utils/func.py`（192 行）
  - `get_valid_name_city`(L127-135)：`(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$`，名字切在**最后一个**逗号，
    城市再过 `extract_before_parenthesis` 并 strip；解析失败返回 `("-", "-")`。
  - `extract_before_parenthesis`(L172-174)：`^(.*?)\([^)]*\)`，**只剁配对括号**。
  - `count_consecutive_values`(L176-192)：与 `commonsense_constraint.py` L37-54 同源。
- `evaluation/hard_constraint.py`（266 行）：`evaluation`(L245-252) 的另外 5 条判据
  （cuisine / house rule / transportation / room type / 成本账）、`get_total_cost`(L71-128)、
  `is_valid_cuisine`(L158-203)、`is_valid_room_type`(L221-242)。**本题不取**（理由见下）。

## 本题的规范化

- **数据库 → 显式 dict**：上游五个 `Flights()/Accommodations()/Restaurants()/Attractions()/
  GoogleDistanceMatrix()` 都是模块级单例，内部是 pandas DataFrame + `os.chdir`。这里换成
  `sandbox` 里的五张 list/dict，`Name.astype(str).str.contains(...)` 的**子串语义**与
  `City == city` 的**相等语义**原样保留（这两者的不对称正是考点之一）。
- **城市-州表 → `city_state_map` 入参**：上游读 `citySet_with_states.txt`，本题由调用方给。
- **硬约束整层剔除**：`hard_constraint.py` 的五条判据要么依赖航班 / 距离定价表做成本账
  （`get_total_cost` 里 `people_number × price`、`ceil(people / 5)` 自驾分车、`ceil(people /
  max_occupancy)` 住宿分房），要么是偏好匹配（cuisine 子串、house rule、room type）。它们与本题
  的「文本一致性 / 序列结构」判据正交，混进来会把题变成算术账，故只留 `commonsense_constraint.py`
  的 8 条。
- **pandas / numpy / tqdm 剔除**：`convert_bool_values`(L36-51) 的 `np.bool_` 转换是给 json 落盘用的，
  本题返回普通 bool。
- **消息文本保留**：8 条判据的 `msg` 逐字照抄（含 `minumum`、`attaction` 两处上游拼写错误），
  因为它们是判据的一部分，也让「哪条先命中」可观测。

## 变异

39 条，38 条 CAUGHT，1 条实证等价：

```
m-regex-anchored         from-to 正则加 ^ 锚定          → 基座（交通串以 Self-driving 开头）抓
m-lookahead              前瞻只剩 (?=[,])              → 基座抓（串尾的 B 匹配不上）
m-from-to-nomatch        解析失败返回 ("","") 而非 None → unparsable_current_city 抓
m-paren-anchored         括号正则去掉右括号要求          → unbalanced_parenthesis 抓
m-name-greedy            (.*?) → (.+)                  → empty_name 抓（空名字不再命中）
m-name-no-strip          名字/城市不 strip              → name_whitespace 抓
m-transport-order        self-driving 抢在 taxi 前      → taxi_wins 抓
m-transport-no-none      认不出时返回 "Flight"          → unknown_kind 抓
m-city-pair-space        'from' → 'from '               → unparsable_current_city 抓
m-runs-init              run 计数从 0 起                → min_nights / base 抓
m-seq-len                长度阈值 3 → 2                 → one_day_trip（from A to A）抓
m-seq-singleton          单例段边界 <= len-1            → 基座 [A,B,B,B,A] 抓
m-seq-first-repeat       去掉 `i != 0`                  → 等价（见下）
m-closed-circle          闭环判据置 False               → closed_circle 抓
m-days-gate              days > 3 → >= 3                → state_gate_only_above_three 抓
m-unknown-city           城市存在性置 False             → city_must_be_known 抓
m-place-substring        名字改成相等比较               → place_substring（R1 vs R10）抓
m-place-city             城市改成子串比较               → place_city_must_equal（AB vs A）抓
m-restaurant-seen        去重表只看 seen[:-1]           → restaurant_repeat_same_day 抓
m-attraction-split       split(';') 不丢尾段            → attraction_semicolon 抓
m-min-nights-rows        len(rows) == 1 → >= 1          → min_nights_ambiguous 抓
m-min-nights-eq          nights < min → <=               → min_nights 抓
m-transport-pair         冲突表 Taxi×Self-driving 换掉  → transport_conflicts 抓
m-day1-skip              第一天只查「非空」不查 '-'     → transport_day1 抓
m-city-consistency-any   any → all                      → city_consistency 抓
m-accommodation-city     cities[-1] → cities[0]         → city_consistency 抓
m-flight-origin          航班只比目的城                 → flight_both_cities 抓
m-flight-split           split 少一个空格               → flight_lookup 抓
m-distance-key           距离键漏掉 mode                → distance_missing 抓
m-needed                 6 * days → 7 * days            → fill_ratio_boundary 抓
m-ratio                  < 0.5 → <= 0.5                 → fill_ratio_boundary（恰 9/18）抓
m-last-day               末日判断差一                   → not_absent_day_rules 抓
m-meal-space             'from ' → 'from'               → meal_rule_with_newline 抓
m-days-placeholder       占位串不再排除                 → placeholder_day 抓
m-discard-org            不再 discard org               → city_number_discards_org 抓
m-bool-int               整数校验放过 bool              → input_validation 抓
m-window                 不再按 days 截断               → extra_days_are_ignored 抓
m-city-number-org        首日基线 org → dest            → 基座抓（city_number 连带 not_absent 挂）
m-field-missing          空串也算缺键                   → empty_string 抓
```

**等价变异的实证**：`if city in visited and i != 0 and i != len(cities) - 1` 里的 `i != 0` 是死代码
——`i` 只会前进，第一次迭代 `visited` 必然为空，所以「起点在 0 且城市已访问」不可达。用穷举核对：
字母表 `{A,B,C}`、长度 1–7 的全部序列（3279 条）上，带与不带 `i != 0` 的结果**逐条相同**（0 差异）。
这不是漏测，删掉该子句也能通过全部用例。

## 已知坑

- **结点数是 2N−1 而不是 N**：`from A to B` 一天贡献两个结点，所以三天的基座行程得到
  `[A,B,B,B,A]`。第一版按「3 个结点」推期望，闭环与单例判据全推错。
- **单日游永远过不了序列判据**：`len < 3` 先拦一层，`from A to A` 这种两结点的写法也拦。写用例时别
  指望「1 天 1 城」能通过 `is_reasonable_visiting_city`。
- **填充率的分子分母不同源**：分母是 `6 * days`，分子按 `unit` 的**全部键**数——上游的 unit 有 7 个
  键（含 `current_city`），所以「只空三餐」是 11/18 = 0.61 仍然通过。第一版用例直接断言「空三餐就该
  跌破 50%」，实际不会，必须再清掉景点才到 8/18。
- **恰好 9/18 = 0.5 是通过**：`< 0.5` 才是失败。这条边界同时钉住 `<` 与分母 6，两个变异都靠它抓。
- **景点串没分号 = 静默消失**：`split(';')[:-1]` 把尾段丢掉，所以 `"Y, B"` 在去重、城市归属、沙箱三处
  都不参与，写错也不报错。
- **`minimum_nights` 只在恰好命中一行时比**：库里同名同城两行时整条判据跳过。想要这条判据生效，测试
  数据必须保证唯一。
- **空名字匹配一切**：`", A"` 切出 name=`""`，`"" in 任何库名` 为真。这是上游 pandas `.str.contains`
  的后果，照抄并在提示里点明。
- `_check_restaurants` 的三条上游消息被合成了统一格式（`The restaurant in day %d %s %s is repeated.`），
  因为本题的断言只看「哪条判据挂」与消息可读性，不需要三种措辞。

## 查重

- `ag-expel-insight-ledger`(752)、`ag-ace-skillbook-ledger`(750)：都是「LLM 产出的文本操作 → 落到
  一份账上」，本题不落账，只做**校验**并返回 per-check 的 `(ok, msg)`。
- 与 `context-compression` / `retrieval` 侧无关：本题不检索、不打分、不做摘要。
- `eval-*`（含 `eval-judge-score-parse`(746)、`eval-judge-swap-consistency`(684)）判的是 LLM 输出文本
  的解析与胜率聚合，本题判的是**结构化约束**（城市序列、去重、沙箱存在性、填充率），没有一行重合。
- TravelPlanner 的硬约束（cuisine / house rule / room type / 成本账）**未立项**：它的判据要么是
  `people_number × 单价` 这类算术（与 `optimization` 侧的账同赛道但更薄），要么是偏好子串匹配，
  且需要再引入航班定价 / 距离定价两张表；本题只留「常识」这一半。
