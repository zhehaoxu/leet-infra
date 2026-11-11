#!/usr/bin/env python3
"""leet-infra 加题脚手架：交互式生成题目、自动挂进合集，并固化两道人工步骤。

用法:
  .venv-backend/bin/python scripts/add_problem.py                     # 交互式加题
  .venv-backend/bin/python scripts/add_problem.py --selftest <slug>   # 用参考解自测某题
  可选 --framework pytorch（默认）/ --mode run（只跑样例，默认）/ submit（全部用例）

生成 problems/<framework>/<slug>/ 下 meta.json、description.md，以及
variants/pytorch/ 三件套（starter.py、test_cases.py、solution.py），
并把 slug 追加进所选合集的 collections.json。
多实现算子题可选：meta.json 写入 pytorch(可用)+triton/cuda(置灰) 声明，
不可用的 variant 不建目录，仅前端选择器置灰展示。
加题流程内固化两步（事项 49）：① 校验题目领域标签是否已在前端
 problemFilter.ts 的 TAG_FAMILY 登记（未登记则提示要补的那一行）；
② 提供 `--selftest <slug>` 直接复用后端 judge 跑一遍用例、回显逐用例结果。
新题落盘即被后端热加载（题库与合集均按 mtime 失效），免重启。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS_DIR = ROOT / "problems"
FILTER_TS = ROOT / "frontend" / "src" / "utils" / "problemFilter.ts"
# 非领域标签：这些是元/结构标签，不参与 family 归组，登记校验时忽略
META_TAGS = {"operator", "framework", "todo"}

TEMPLATE_META = '''{{
  "title": "{title}",
  "difficulty": "{difficulty}",
  "tags": {tags},
  "order": {order}{variants}
}}
'''

# 多实现算子题的 variants 声明：仅 pytorch 可用，triton/cuda 置灰占位
VARIANTS_JSON = ''',
  "variants": {
    "pytorch": {"label": "PyTorch", "available": true},
    "triton": {"label": "Triton", "available": false},
    "cuda": {"label": "CUDA", "available": false}
  }'''

TEMPLATE_DESC = '''# {title}

## 题目描述

TODO：用一两句话说明要实现的算子/函数及其意义。

## 规则

- 实现 `solve({params})`，返回 TODO。

## 示例

```
输入：TODO
输出：TODO
```

## 约束

- TODO（张量形状、取值范围、数据类型）
'''

TEMPLATE_STARTER = '''import torch


def solve({params}):
    # TODO: 在这里实现
    raise NotImplementedError
'''

TEMPLATE_TEST = '''import torch


# 样例用例（点「运行」只执行标了 example = True 的用例）
def check_basic(solve):
    out = solve(TODO)
    assert out.shape == TODO, f"形状错误: {{out.shape}}"


def check_edge_case(solve):
    out = solve(TODO)
    assert torch.allclose(out, TODO), "边界用例不匹配"


# 样例用例：点「运行」只执行这些；「提交」执行全部用例
check_basic.example = True
'''

TEMPLATE_SOLUTION = '''"""参考答案：{title}（TODO：实现后移除本占位并自测）。"""

import torch


def solve({params}):
    raise NotImplementedError("TODO: 参考实现待补全")
'''


def _tag_family_pairs() -> tuple[dict[str, str], list[str]] | None:
    """解析前端 problemFilter.ts，返回 (TAG_FAMILY 映射, GROUP_ORDER 族顺序)；文件缺失返回 None。"""
    if not FILTER_TS.exists():
        return None
    text = FILTER_TS.read_text(encoding="utf-8")
    tm = re.search(r"TAG_FAMILY[^{]*\{(.*?)\}", text, re.S)
    pairs = dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*'([^']*)'", tm.group(1))) if tm else {}
    gm = re.search(r"GROUP_ORDER\s*=\s*\[(.*?)\]", text, re.S)
    order = re.findall(r"'([^']*)'", gm.group(1)) if gm else []
    return pairs, order


def check_family_registered(tags: list[str]) -> None:
    """算子题领域标签若未在前端 TAG_FAMILY 登记则提示补哪一行（事项 49）。"""
    if "operator" not in tags:
        return  # 非算子题统一落「框架 / 其他」，无需登记
    parsed = _tag_family_pairs()
    if parsed is None:
        print("  ⚠ 未找到 problemFilter.ts，跳过族登记校验。")
        return
    pairs, order = parsed
    domain = [t for t in tags if t not in META_TAGS]
    bad_ident = [t for t in domain if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", t)]
    if bad_ident:
        print(f"  ⚠ 标签 {bad_ident} 不是裸标识符（含连字符/空格等）："
              f"TAG_FAMILY 的 key 必须为 [A-Za-z_][A-Za-z0-9_]*，否则后端完整性护栏抽不到。")
    hit = [t for t in domain if t in pairs]
    if hit:
        fams = {pairs[t] for t in hit}
        missing_order = [f for f in fams if f not in order]
        if missing_order:
            print(f"  ⚠ 标签已登记但族名不在 GROUP_ORDER：{missing_order}，"
                  f"请在 GROUP_ORDER 补上以控制分组顺序。")
        else:
            print(f"  ✓ 族登记 OK：{', '.join(f'{t}→{pairs[t]}' for t in hit)}")
    else:
        line = ", ".join(f"{t}: '族名'" for t in domain) or "标签: '族名'"
        print(f"  ⚠ 该算子题的领域标签 {domain} 未在 problemFilter.ts 的 TAG_FAMILY 登记，"
              f"前端会把它归入「框架 / 其他」。如需单独成族，请在 TAG_FAMILY 增加一行：")
        print(f"        {line},")
        print("     并把新族名插入 GROUP_ORDER 期望位置。")


def run_selftest(slug: str, framework: str, mode: str) -> int:
    """复用后端 judge 跑参考解对拍，逐用例回显（事项 49）。返回退出码：0=AC，2=未 AC，1=环境/文件错误。"""
    vdir = PROBLEMS_DIR / framework / slug / "variants" / "pytorch"
    sol, tests = vdir / "solution.py", vdir / "test_cases.py"
    if not sol.exists() or not tests.exists():
        print(f"  ✗ 未找到 {sol} 或 {tests}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT))
    try:
        from backend import judge
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ 无法导入后端 judge（请用 .venv-backend/bin/python 运行）：{exc}", file=sys.stderr)
        return 1
    res = judge.run_judgement(sol.read_text(encoding="utf-8"),
                              tests.read_text(encoding="utf-8"), mode=mode)
    cases = res.get("cases") or []
    if not cases and res.get("error"):
        print(f"  ✗ 评测未完成 [{res.get('status')}]：{res.get('error')}", file=sys.stderr)
        if res.get("stdout"):
            print(res["stdout"])
        return 1
    for c in cases:
        icon = "✓" if c["passed"] else "✗"
        msg = f"  — {c['message']}" if c.get("message") else ""
        print(f"  {icon} {c['name']}{msg}")
    if res.get("stdout"):
        print("  ── 用户 stdout ──")
        print(res["stdout"])
    ok = bool(res.get("accepted"))
    print(f"  => {res.get('passed')}/{res.get('total')} 通过；"
          f"{'ACCEPTED' if ok else 'NOT accepted'}（mode={mode}）")
    return 0 if ok else 2


def ask(prompt: str, default: str = "") -> str:
    while True:
        v = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
        if v:
            return v
        if default:
            return default
        print("  该项不能为空。")


def main() -> int:
    frameworks = sorted(p.name for p in PROBLEMS_DIR.glob("*/") if p.name != "collections.json" and p.is_dir())
    print("== leet-infra 加题 ==")
    framework = ask("框架目录", frameworks[0] if frameworks else "pytorch")
    fw_dir = PROBLEMS_DIR / framework
    fw_dir.mkdir(parents=True, exist_ok=True)

    while True:
        slug = ask("题目 slug（目录名，kebab-case）")
        if (fw_dir / slug).exists():
            print(f"  已存在：{fw_dir / slug}，换一个。")
            continue
        break

    title = ask("题目标题", slug)
    difficulty = ""
    while difficulty not in {"Easy", "Medium", "Hard"}:
        difficulty = ask("难度（Easy/Medium/Hard）", "Medium")
    tags = [t.strip() for t in ask("标签（逗号分隔）", "").split(",") if t.strip()] or ["todo"]
    params = ask("solve 参数列表（如 x: torch.Tensor, eps: float = 1e-5）", "x: torch.Tensor")

    orders = []
    for meta in fw_dir.glob("*/meta.json"):
        try:
            orders.append(json.loads(meta.read_text(encoding="utf-8")).get("order", 0))
        except json.JSONDecodeError:
            pass
    order = max(orders, default=0) + 1

    problem_dir = fw_dir / slug
    variant_dir = problem_dir / "variants" / "pytorch"
    variant_dir.mkdir(parents=True)
    multi = ask("多实现算子题（声明 triton/cuda 置灰）？(y/N)", "n").lower() == "y"
    files = {
        "meta.json": TEMPLATE_META.format(
            title=title,
            difficulty=difficulty,
            tags=json.dumps(tags, ensure_ascii=False),
            order=order,
            variants=VARIANTS_JSON if multi else "",
        ),
        "description.md": TEMPLATE_DESC.format(title=title, params=params),
        "starter.py": TEMPLATE_STARTER.format(params=params),
        "test_cases.py": TEMPLATE_TEST,
        "solution.py": TEMPLATE_SOLUTION.format(title=title, params=params),
    }
    (problem_dir / "meta.json").write_text(files.pop("meta.json"), encoding="utf-8")
    (problem_dir / "description.md").write_text(files.pop("description.md"), encoding="utf-8")
    for name, content in files.items():
        (variant_dir / name).write_text(content, encoding="utf-8")
    print(f"已生成 {problem_dir.relative_to(ROOT)}：meta.json、description.md、"
          f"variants/pytorch/{{{', '.join(['starter.py', 'test_cases.py', 'solution.py'])}}}")
    if multi:
        print("  variants 已声明 pytorch(可用) + triton/cuda(置灰)；后续实现时建对应目录即可。")

    # 挂进合集（可多选，回车跳过）
    collections_file = PROBLEMS_DIR / "collections.json"
    if collections_file.exists():
        collections = json.loads(collections_file.read_text(encoding="utf-8"))
        picked = []
        for col in collections:
            answer = ask(f"加入合集「{col['name']}」？(y/N)", "n")
            if answer.lower() == "y":
                col["problems"].append(slug)
                picked.append(col["slug"])
        if picked:
            collections_file.write_text(
                json.dumps(collections, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"已加入合集: {', '.join(picked)}")

    # 固化步骤①：校验领域标签是否已在前端 problemFilter 登记
    check_family_registered(tags)

    print("\n下一步：")
    print("  1) 补全 description.md / variants/pytorch/ 三件套中的 TODO"
          "（solution 填参考实现、test_cases 用独立纯 Python oracle 写 check_*）。")
    print(f"  2) 用参考解自测判题链路（复用后端 judge，跑例值 dump）：")
    print(f"       .venv-backend/bin/python scripts/add_problem.py --selftest {slug} --mode run")
    print(f"     （--mode run 只跑样例用例；--mode submit 跑全部）")
    if ask("\n现在就用占位参考解跑一次自测冒烟（验证链路接得通）？(y/N)", "n").lower() == "y":
        run_selftest(slug, framework, "run")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="leet-infra 加题脚手架：交互式加题，或 --selftest 复用后端 judge 自测。")
    p.add_argument("--selftest", metavar="SLUG",
                   help="用该题的参考解对拍 test_cases 并逐用例回显（不进入交互式加题）")
    p.add_argument("--framework", default="pytorch", help="题目所在框架目录（默认 pytorch）")
    p.add_argument("--mode", choices=["run", "submit"], default="run",
                   help="selftest 模式：run 只跑样例（默认），submit 跑全部用例")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.selftest:
        sys.exit(run_selftest(args.selftest, args.framework, args.mode))
    sys.exit(main())
