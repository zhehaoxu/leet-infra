"""Active-Prompt 的不确定度选样（k 次采样答案 → 计数 → 排序挑 top）。

官方 shizhediao/active-prompt 的 uncertainty_estimation.py 确定性核心：
（GPT 采样与 answer_extraction 不入判题域——每题 k 次采样的**抽取后**
答案串直接作入参；answer 为空串表示「无解」。）

1. occurrence 计数：逐题逐采样，非空答案按**原串**为键计数（首现序）；
   空答案计入哨兵键 NO_SOLUTION = '-10086'（官方常量）；
2. variance（仅 dataset_kind="numerical"）：occurrence 按 `for ans, occurs
   in occurrence.items(): for i in range(int(occurs)): append(float(ans))`
   展开成数值列表（**NO_SOLUTION 也以 -10086 计入**——官方原样），
   variance = np.var(ans_list)（总体方差，ddof=0）；occurrence 为空时
   ans_list 为空，np.var([]) 发出 RuntimeWarning 且返回 nan——按官方
   原样保留 nan 语义（num_trails≥1 时不会发生）；
3. entropy：scipy.stats.entropy(frequency) 口径——
   entropy = -Σ_{c>0} (c/n)·ln(c/n)，0 频率项贡献 0（yes/no 预置零频
   不影响），频率全零（n=0）时为 nan；
4. disagreement = len(occurrence)（含 NO_SOLUTION 键）；
5. 排序（stable，平局保持输入序）：
   - disagreement：key = -len(occurrence)（distinct 答案多者在前）；
     dataset_kind="strategyqa" 特例：key = abs(yes - no) **升序**
     （两票越接近越不确定；yes/no 键预置恒存在，官方 except 兜底
     -len(occurrence) 属死分支，原样保留）；
   - variance：key = -variance，**仅 numerical**（官方 elif 条件带
     dataset 过滤，不匹配时静默不排序、保持输入序）；
   - entropy：key = -entropy（全任务）。
6. qes_limit：只计算前 qes_limit 题（0 = 全部；超过题数即全部），
   排序只作用于被计算的子集。

返回 records 列表（排序后），每条
`dict(dataset_idx, variance?, entropy, occurrence, disagreement)`，
dataset_idx 为原输入下标。
"""

import math

_NO_SOLUTION = "-10086"


def _entropy(freq):
    """scipy.stats.entropy 口径：-Σ (c/n)·ln(c/n)，0 频率贡献 0。"""
    n = 0
    for c in freq:
        n += c
    if n == 0:
        return float("nan")
    total = 0.0
    for c in freq:
        if c > 0:
            q = c / n
            total -= q * math.log(q)
    return total


def _validate(name, value, n=None):
    if not isinstance(value, list):
        raise ValueError(f"{name} 必须是 list")
    if not value:
        raise ValueError(f"{name} 不能为空")
    for i, item in enumerate(value):
        if not isinstance(item, list):
            raise ValueError(f"{name}[{i}] 必须是 list")
        for j, s in enumerate(item):
            if not isinstance(s, str):
                raise ValueError(f"{name}[{i}][{j}] 必须是字符串")
    if n is not None and len(value) != n:
        raise ValueError(f"{name} 长度必须为 {n}")


def solve(samples_per_question, sort_by="disagreement", dataset_kind="generic",
          qes_limit=0):
    n = len(samples_per_question)
    _validate("samples_per_question", samples_per_question)
    if sort_by not in ("disagreement", "variance", "entropy"):
        raise ValueError("sort_by 必须是 disagreement/variance/entropy 之一")
    if dataset_kind not in ("generic", "numerical", "strategyqa"):
        raise ValueError("dataset_kind 必须是 generic/numerical/strategyqa 之一")
    if isinstance(qes_limit, bool) or not isinstance(qes_limit, int) \
            or qes_limit < 0:
        raise ValueError("qes_limit 必须是 >= 0 的整数")
    for i, samples in enumerate(samples_per_question):
        if not samples:
            raise ValueError(f"samples_per_question[{i}] 不能为空（num_trails >= 1）")

    if qes_limit == 0 or qes_limit > n:
        qes_limit = n

    records = []
    for idx in range(qes_limit):
        samples = samples_per_question[idx]
        occurrence = {}
        if dataset_kind == "strategyqa":
            occurrence = {"yes": 0, "no": 0}
        for ans in samples:
            if ans != "":
                if ans in occurrence:
                    occurrence[ans] += 1
                else:
                    occurrence[ans] = 1
            else:
                if _NO_SOLUTION in occurrence:
                    occurrence[_NO_SOLUTION] += 1
                else:
                    occurrence[_NO_SOLUTION] = 1

        record = {"dataset_idx": idx, "occurrence": occurrence}
        if dataset_kind == "numerical":
            ans_list = []
            for ans, occurs in occurrence.items():
                for _ in range(int(occurs)):
                    ans_list.append(float(ans))
            if ans_list:
                mean = sum(ans_list) / len(ans_list)
                record["variance"] = sum((v - mean) ** 2 for v in ans_list) \
                    / len(ans_list)
            else:
                record["variance"] = float("nan")
        record["entropy"] = _entropy(list(occurrence.values()))
        record["disagreement"] = len(occurrence)
        records.append(record)

    if sort_by == "disagreement":
        if dataset_kind == "strategyqa":
            try:
                # 官方：两票差越小越不确定（升序）；yes/no 预置恒在,
                # except 兜底属死分支，原样保留
                records.sort(key=lambda x: abs(x["occurrence"]["yes"]
                                              - x["occurrence"]["no"]))
            except Exception:
                records.sort(key=lambda x: -len(x["occurrence"]))
        else:
            records.sort(key=lambda x: -len(x["occurrence"]))
    elif sort_by == "variance" and dataset_kind == "numerical":
        records.sort(key=lambda x: -x["variance"])
    elif sort_by == "entropy":
        records.sort(key=lambda x: -x["entropy"])
    # 其余组合（如 variance + generic）：官方 elif 不匹配即不排序
    return records
