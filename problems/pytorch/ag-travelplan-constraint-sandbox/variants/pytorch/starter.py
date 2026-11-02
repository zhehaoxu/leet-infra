def solve(query, itinerary, city_state_map, sandbox):
    # TODO: TravelPlanner 的常识约束校验（8 条判据，按上游 evaluation() 的顺序）。
    #   query      = {org, dest, days, visiting_city_number, ...}
    #   itinerary  = 逐天的 {current_city, transportation, breakfast, lunch, dinner,
    #                attraction, accommodation}；文本原样，不做清洗
    #   city_state_map = {城市: 州}；sandbox = {restaurants / accommodations / attractions /
    #                flights / distances} 五张显式表
    #   - 城市结点：当天 current_city 含 'from' 时用 `from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)`
    #     展开成两个城，否则只取一个；城市末尾的 `(State)` 要剁掉。首日为 'from' 时第一城
    #     必须等于 org；首尾城必须相同（闭环）；结点序列里「中间只出现一次」的城判非法。
    #   - 餐厅：三餐按整串文本跨天去重；景点：`split(';')[:-1]`（最后一段被丢掉）。
    #   - 住宿：按 run 长度与 minimum nights 比；景点/住宿/餐厅在沙箱里按「名字是库中名字的
    #     子串 + 城市精确相等」匹配，名字切在**最后一个**逗号上。
    #   - 交通：第一天必须有；taxi / self-driving / flight 三类归一后，自驾×飞机、出租×自驾
    #     算冲突；航班号取 'Flight Number: ' 到第一个逗号之间，并按起降城市双匹配。
    #   - 信息缺失：6 个字段必须存在；跨城日要交通、单城日要景点、非末日要住宿、非跨城日要
    #     三餐；最后按「非空且非 '-' 的键数 / (6 * days)」算填充率，低于 0.5 判失败。
    #   - 返回 {"checks": {名字: {"ok", "msg"}}, "failed": [...], "passed": bool}。
    raise NotImplementedError
