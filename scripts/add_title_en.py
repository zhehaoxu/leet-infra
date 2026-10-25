#!/usr/bin/env python3
"""Insert `"title_en": "..."` right after `"title"` in a problem's meta.json."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    slug, title_en = sys.argv[1], sys.argv[2]
    framework = sys.argv[3] if len(sys.argv) > 3 else "pytorch"
    path = ROOT / "problems" / framework / slug / "meta.json"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().startswith('"title":'):
            break
    else:
        print(f"no title key in {path}")
        return 1
    if any(l.lstrip().startswith('"title_en":') for l in lines):
        return 0
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines.insert(i + 1, f'{indent}"title_en": {json.dumps(title_en, ensure_ascii=False)},\n')
    path.write_text("".join(lines), encoding="utf-8")
    json.loads(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
