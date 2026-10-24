"""backend/tests 级 conftest：路径注入 + self_ac 冒烟档开关。

- 把仓库根塞进 sys.path，使 `from backend import ...` 在任意 pytest 起始目录/CWD 都可导入。
- 提供 `--smoke`：让全量自测只跑每题样例用例（run 模式），命中题面/示例数值的最快冒烟档；
  不带该选项时维持原样（submit 全量）。与 pytest-xdist `-n auto` 可叠加。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help="self_ac 仅执行每题样例用例（run 模式），跳过耗时的全量/随机大对拍",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: 仅样例用例的快速自测档")
