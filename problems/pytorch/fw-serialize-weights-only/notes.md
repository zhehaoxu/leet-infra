# notes — fw-serialize-weights-only（order 712）

## 回源

- 出处：本地判题环境 `.venv-eval` 的 `torch/serialization.py` 与 `torch/_weights_only_unpickler.py`（v2.14.1）。判定表从 `_get_allowed_globals` + protocol-2 实际序列化输出反推（oracle 用真 `torch.load(weights_only=True)` 逐场景闭环验证，判定表与官方行为一致即锚定）。
- 关键事实：torch.save 是 zip 容器（新 zipfile 格式），pkl 成员在 `archive/data.pkl`，默认 protocol 2——GLOBAL 是两行文本指令（`module\nname\n`），genops 的 `str(arg)` 形如 `"collections OrderedDict"`；纯 primitive 容器无 GLOBAL。
- 与 `fw-resume-checkpoint`/`fw-state-dict-remap` 划界：本题是 weights_only 白名单判定层 + load_state_dict 键级差集账，不是键重映射业务。

## 判题面

- solve() 无参固定 17 场景；oracle 用真 `torch.load(weights_only=True)` 判 verdict + ok 场景值 roundtrip（torch.equal）+ 真 `Module.load_state_dict` 出 missing/unexpected + 2.6 默认账（裸 load nn.Linear 被拒、裸 load 纯 tensor dict 通过）。
- 红线检查：扫描 solution 源码禁 `torch.load(`/`.load_state_dict(` 调用形态（harness 无 `__file__` 时跳过；注意不能用宽松子串——docstring 里的字样会误伤，曾致变异误判 CAUGHT）。

## 锚点（全部真 torch 实测）

- 放行组：tensor dict（OrderedDict+FloatStorage+_rebuild_tensor_v2）、嵌套 primitive（零 GLOBAL）、OrderedDict、Counter、Size、device、dtype（torch float32）、set/bytearray（+_codecs encode）/complex、Parameter（_rebuild_parameter，无独立 GLOBAL）。
- 拒绝组：nn.Linear（torch.nn.modules.linear Linear）、SimpleNamespace（types SimpleNamespace）、**range → `__builtin__ xrange`**（protocol 2 映射 quirk）、numpy（dtype/ndarray/_reconstruct）。
- lambda：save 阶段 PicklingError → save_fail 分类（非 load 拒）。
- load_state_dict：`0.bias < 0.weight` 字典序 sorted；异构账 missing=["0.bias","0.weight"] / unexpected=["bias","weight"]；+extra.weight → unexpected=["bias","extra.weight","weight"]；strict RuntimeError 首行前缀；根模块键无属性名前缀。

## 变异记录（11 条全抓）

linear-allow、xrange-allow、np-allow、dtype-drop、counter-drop、storage-drop、verdict-flip、scan-wrong-member（读错 zip 成员）、missing-unexpected-swap、savefail-to-ok、sort-drop（去 sorted——unexpected 三键自然序 ≠ 字典序，场景已覆盖）。

等价变异 2 条记 notes：STACK_GLOBAL 扫描分支删除（protocol 2 无 STACK_GLOBAL 指令，死码；实测 17 场景均未出现）、strict-flip（恒等式写废变异）。STACK_GLOBAL 分支保留以对齐官方 `find_class` 的 protocol-4 兼容意图。

## 红线

solution 禁 torch.load/load_state_dict（判定层手写）；safe_globals 动态注册（_add_safe_globals）、torch.package、jit.save 不出；多进程/分布式 checkpoint（dcp）不出；与 `fw-resume-checkpoint`（键重映射业务）不重叠。
