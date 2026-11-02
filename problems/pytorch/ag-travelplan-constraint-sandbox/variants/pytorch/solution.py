"""TravelPlanner 约束沙盒：行程的常识约束校验（8 条判据）。

对照 OSU-NLP-Group/TravelPlanner：
- `evaluation/commonsense_constraint.py`：`evaluation`(L521-531) 的 8 条判据顺序、
  `is_reasonable_visiting_city`(L128-160)、`is_valid_restaurants`(L163-194)、
  `is_valid_attractions`(L196-213)、`is_valid_transportation`(L215-234)、
  `is_valid_information_in_current_city`(L236-320)、`is_valid_information_in_sandbox`(L323-398)、
  `is_valid_accommodaton`(L401-424)、`is_not_absent`(L466-518)；
  以及 `is_valid_city_sequence`(L85-124)、`count_consecutive_values`(L37-54)、
  `transportation_match`(L57-66)、`extract_from_to`(L69-81)、`is_valid_days`(L454-464)、
  `is_valid_visiting_city_number`(L426-452)。
- `utils/func.py`：`get_valid_name_city`(L127-135)、`extract_before_parenthesis`(L172-174)。

数据库（flights / accommodations / restaurants / attractions / distance matrix）与城市-州表
全部换成显式入参 dict；LLM 生成的行程文本原样作入参。硬约束（`hard_constraint.py` 的
cuisine / house rule / room type / 交通偏好 / 预算成本账）不在本题范围。
"""
import re

_FROM_TO_RE = re.compile(r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)")
_PAREN_RE = re.compile(r"^(.*?)\([^)]*\)")
_NAME_CITY_RE = re.compile(r"(.*?),\s*([^,]+)(\(\w[\w\s]*\))?$")

_FIELDS = ("transportation", "breakfast", "lunch", "dinner", "attraction", "accommodation")
_DAYS_PLACEHOLDER = "You don't need to fill in the information for this or later days."
_NAMES = (
    "is_reasonable_visiting_city",
    "is_valid_restaurants",
    "is_valid_attractions",
    "is_valid_accommodation",
    "is_valid_transportation",
    "is_valid_information_in_current_city",
    "is_valid_information_in_sandbox",
    "is_not_absent",
)


def _as_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("%s 必须是整数" % name)
    if value < minimum:
        raise ValueError("%s 必须 >= %d" % (name, minimum))
    return value


def _extract_from_to(text):
    match = _FROM_TO_RE.search(text)
    return match.groups() if match else (None, None)


def _before_paren(text):
    match = _PAREN_RE.search(text)
    return match.group(1) if match else text


def _name_city(info):
    if not isinstance(info, str):
        return "-", "-"
    match = _NAME_CITY_RE.search(info)
    if match is None:
        return "-", "-"
    return match.group(1).strip(), _before_paren(match.group(2).strip()).strip()


def _transport_match(text):
    low = text.lower()
    if "taxi" in low:
        return "Taxi"
    if "self-driving" in low:
        return "Self-driving"
    if "flight" in low:
        return "Flight"
    return None


def _city_pair(unit):
    text = unit["current_city"]
    if "from" in text:
        first, second = _extract_from_to(text)
        if first is None or second is None:
            raise ValueError("current_city 无法解析: %r" % text)
        return [_before_paren(first), _before_paren(second)]
    return [_before_paren(text)]


def _runs(values):
    if not values:
        return []
    result = []
    current, count = values[0], 1
    for value in values[1:]:
        if value == current:
            count += 1
        else:
            result.append((current, count))
            current, count = value, 1
    result.append((current, count))
    return result


def _valid_city_sequence(cities):
    if len(cities) < 3:
        return False
    visited = set()
    i = 0
    while i < len(cities):
        city = cities[i]
        if city in visited and i != 0 and i != len(cities) - 1:
            return False
        count = 0
        while i < len(cities) and cities[i] == city:
            count += 1
            i += 1
        if count == 1 and 0 < i - 1 < len(cities) - 1:
            return False
        visited.add(city)
    return True


def _place_rows(sandbox, key, info):
    name, city = _name_city(info)
    return [row for row in (sandbox.get(key) or [])
            if name in row["name"] and row["city"] == city]


def _window(query, itinerary):
    return itinerary[:min(query["days"], len(itinerary))]


def _check_visiting_city(query, itinerary, city_state_map):
    cities = []
    for index, unit in enumerate(_window(query, itinerary)):
        current = unit["current_city"]
        if "from" in current:
            first, second = _city_pair(unit)
            if index == 0 and first != query["org"]:
                return False, "The first day's city should be %s." % query["org"]
            cities += [first, second]
        else:
            cities.append(_before_paren(current))
    if cities[0] != cities[-1]:
        return False, "The trip should be a closed circle."
    if not _valid_city_sequence(cities):
        return False, "The city sequence is invalid."
    for index, city in enumerate(cities):
        if city not in city_state_map:
            return False, "%s is not a valid city." % city
        if index not in (0, len(cities) - 1) and query["days"] > 3 and city_state_map[city] != query["dest"]:
            return False, "%s is not in %s." % (city, query["dest"])
    return True, None


def _check_restaurants(query, itinerary):
    seen = []
    for index, unit in enumerate(_window(query, itinerary)):
        for field in ("breakfast", "lunch", "dinner"):
            value = unit.get(field)
            if not value or value == "-":
                continue
            if value in seen:
                return False, "The restaurant in day %d %s %s is repeated." % (index + 1, field, value)
            seen.append(value)
    return True, None


def _check_attractions(query, itinerary):
    seen = []
    for index, unit in enumerate(_window(query, itinerary)):
        value = unit.get("attraction")
        if not value or value == "-":
            continue
        for attraction in value.split(";")[:-1]:
            if attraction in seen:
                return False, "The attraction '%s' in day %d is repeated." % (attraction, index + 1)
            seen.append(attraction)
    return True, None


def _check_accommodation(query, itinerary, sandbox):
    names = []
    for unit in _window(query, itinerary):
        if "accommodation" not in unit:
            return False, "No Accommodation Info."
        names.append(unit["accommodation"])
    for value, nights in _runs(names):
        if value in ("", "-"):
            continue
        rows = _place_rows(sandbox, "accommodations", value)
        if len(rows) == 1 and nights < rows[0]["minimum_nights"]:
            return False, "The accommodation %s do not obey the minumum nights rule." % value
    return True, None


def _check_transportation(query, itinerary):
    first = itinerary[0].get("transportation")
    if not first or first == "-":
        return False, "The transportation in day 1 should not be empty."
    kinds = [_transport_match(first)]
    for unit in _window(query, itinerary):
        value = unit.get("transportation")
        if value and value != "-":
            kinds.append(_transport_match(value))
    if ("Self-driving" in kinds and "Flight" in kinds) or ("Taxi" in kinds and "Self-driving" in kinds):
        return False, "The transportation is conflicting."
    return True, None


def _check_city_consistency(query, itinerary):
    for index, unit in enumerate(_window(query, itinerary)):
        cities = _city_pair(unit)
        value = unit.get("transportation")
        if value and value != "-":
            for city in cities:
                if city not in value:
                    return False, "The transportation in day %d is invalid city choice." % (index + 1)
        for field in ("breakfast", "lunch", "dinner"):
            value = unit.get(field)
            if not value or value == "-":
                continue
            if not any(city in value for city in cities):
                return False, "The %s in day %d is invalid city choice." % (field, index + 1)
        value = unit.get("attraction")
        if value and value != "-":
            for attraction in value.split(";")[:-1]:
                if not any(city in attraction for city in cities):
                    return False, "The attraction in day %d is invalid city choice." % (index + 1)
        value = unit.get("accommodation")
        if value and value != "-":
            if cities[-1] not in value:
                return False, "The accommodation in day %d is invalid city choice." % (index + 1)
    return True, None


def _check_sandbox(query, itinerary, sandbox):
    flights = sandbox.get("flights") or {}
    distances = sandbox.get("distances") or {}
    for index, unit in enumerate(_window(query, itinerary)):
        value = unit.get("transportation")
        if value and value != "-":
            org, dest = _extract_from_to(value)
            if org is None or dest is None:
                org, dest = _extract_from_to(unit["current_city"])
            if "flight number" in value.lower():
                parts = value.split("Flight Number: ")
                if len(parts) < 2 or org is None or dest is None:
                    raise ValueError("The transportation %s in day %d can not be parsed." % (value, index + 1))
                number = parts[1].split(",")[0]
                row = flights.get(number)
                if row is None or row["origin"] != _before_paren(org) or row["dest"] != _before_paren(dest):
                    return False, "The flight number in day %d is invalid in the sandbox." % (index + 1)
            elif "self-driving" in value.lower() or "taxi" in value.lower():
                if org is None or dest is None:
                    org, dest = "-", "-"
                else:
                    org, dest = _before_paren(org), _before_paren(dest)
                mode = "self-driving" if "self-driving" in value.lower() else "taxi"
                if distances.get((org, dest, mode)) is None:
                    return False, "The %s in day %d is invalid in the sandbox." % (mode, index + 1)
        for field, table in (("breakfast", "restaurants"), ("lunch", "restaurants"),
                             ("dinner", "restaurants"), ("accommodation", "accommodations")):
            value = unit.get(field)
            if value and value != "-":
                if not _place_rows(sandbox, table, value):
                    return False, "The %s in day %d is invalid in the sandbox." % (field, index + 1)
        value = unit.get("attraction")
        if value and value != "-":
            for attraction in value.split(";")[:-1]:
                if not _place_rows(sandbox, "attractions", attraction):
                    return False, "The attraction %s in day %d is invalid in the sandbox." % (attraction, index + 1)
    return True, None


def _check_days(query, itinerary):
    days = 0
    for unit in _window(query, itinerary):
        if unit and unit.get("current_city") != _DAYS_PLACEHOLDER:
            days += 1
    if days != query["days"]:
        return False, "The number of days should be %d." % query["days"]
    return True, None


def _check_city_number(query, itinerary):
    cities = set()
    for index, unit in enumerate(_window(query, itinerary)):
        current = unit["current_city"]
        if "from" in current:
            first, second = _city_pair(unit)
            if index == 0 and first != query["org"]:
                return False, "The first day's city should be %s." % query["org"]
            cities.add(first)
            cities.add(second)
        else:
            cities.add(_before_paren(current))
    cities.discard(query["org"])
    if len(cities) != query["visiting_city_number"]:
        return False, "The number of visiting cities should be %d." % query["visiting_city_number"]
    return True, None


def _check_not_absent(query, itinerary):
    needed = 6 * query["days"]
    filled = 0
    if not _check_days(query, itinerary)[0]:
        return False, "Invalid Days"
    if not _check_city_number(query, itinerary)[0]:
        return False, "Invalid City Number"
    for index, unit in enumerate(_window(query, itinerary)):
        for field in _FIELDS:
            if field not in unit:
                return False, "No %s Info." % field.capitalize()
        current = unit["current_city"]
        if ("from " in current or "to " in current) and unit["transportation"] in ("", "-"):
            return False, "No transportation in day %d is not allowed." % (index + 1)
        if ("from " not in current and " to " not in current) and unit["attraction"] in ("", "-"):
            return False, "No attaction in day %d is not allowed." % (index + 1)
        if index != query["days"] - 1 and unit["accommodation"] in ("", "-"):
            return False, "No accommodation in day %d is not allowed." % (index + 1)
        if (unit["breakfast"] in ("", "-") or unit["lunch"] in ("", "-")
                or unit["dinner"] in ("", "-")) and "from " not in current:
            return False, "No meal in day %d is not allowed." % (index + 1)
        for key in unit:
            if unit[key] and unit[key] != "-":
                filled += 1
    if filled * 1.0 / needed < 0.5:
        return False, "The absent information is more than 50%."
    return True, None


def solve(query, itinerary, city_state_map, sandbox):
    if not isinstance(query, dict):
        raise ValueError("query 必须是字典")
    if not isinstance(itinerary, list) or not itinerary:
        raise ValueError("itinerary 必须是非空列表")
    if not isinstance(city_state_map, dict):
        raise ValueError("city_state_map 必须是字典")
    if not isinstance(sandbox, dict):
        raise ValueError("sandbox 必须是字典")
    _as_int(query.get("days"), "query['days']")
    _as_int(query.get("visiting_city_number"), "query['visiting_city_number']", minimum=0)
    for name in ("org", "dest"):
        if not isinstance(query.get(name), str):
            raise ValueError("query[%r] 必须是字符串" % name)
    for index, unit in enumerate(itinerary):
        if not isinstance(unit, dict) or not isinstance(unit.get("current_city"), str):
            raise ValueError("itinerary[%d] 必须是含 current_city 的字典" % index)

    results = [
        _check_visiting_city(query, itinerary, city_state_map),
        _check_restaurants(query, itinerary),
        _check_attractions(query, itinerary),
        _check_accommodation(query, itinerary, sandbox),
        _check_transportation(query, itinerary),
        _check_city_consistency(query, itinerary),
        _check_sandbox(query, itinerary, sandbox),
        _check_not_absent(query, itinerary),
    ]
    checks = {}
    for name, (ok, msg) in zip(_NAMES, results):
        checks[name] = {"ok": bool(ok), "msg": msg}
    failed = [name for name in _NAMES if not checks[name]["ok"]]
    return {"checks": checks, "failed": failed, "passed": not failed}
