#!/usr/bin/env python3
"""leet-infra 加题脚手架：交互式生成题目五件套并自动挂进合集。

用法:  .venv-backend/bin/python scripts/add_problem.py
生成 problems/<framework>/<slug>/ 下 meta.json、description.md、starter.py、
test_cases.py、solution.py，并把 slug 追加进所选合集的 collections.json。
新题落盘即被后端热加载（题库与合集均按 mtime 失效），免重启。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS_DIR = ROOT / "problems"

TEMPLATE_META = '''{{
  "title": "{title}",
  "difficulty": "{difficulty}",
  "tags": {tags},
  "order": {order}
}}
'''

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
    problem_dir.mkdir()
    files = {
        "meta.json": TEMPLATE_META.format(
            title=title, difficulty=difficulty, tags=json.dumps(tags, ensure_ascii=False), order=order
        ),
        "description.md": TEMPLATE_DESC.format(title=title, params=params),
        "starter.py": TEMPLATE_STARTER.format(params=params),
        "test_cases.py": TEMPLATE_TEST,
        "solution.py": TEMPLATE_SOLUTION.format(title=title, params=params),
    }
    for name, content in files.items():
        (problem_dir / name).write_text(content, encoding="utf-8")
    print(f"已生成 {problem_dir.relative_to(ROOT)} / {', '.join(files)}")

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

    print("\n下一步：补全 description.md / starter.py / test_cases.py / solution.py 四个 TODO，"
          "再用参考解跑一次 POST /api/problems/{slug}/submit 自测判题链路。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
