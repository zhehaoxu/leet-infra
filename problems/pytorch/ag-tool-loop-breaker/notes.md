# notes — ag-tool-loop-breaker（order 667）

## 回源

- 题型背景：agent 运行时的工具循环护栏（loop guard）。上游普遍做法是对 `(tool, canonical(args))` 做指纹并按出现次数熔断（等价于 cache 语义 + 计数闸门），如各类 agent 框架的 step budget / duplicate-call 检测；本题把口径抽象钉死，不绑定具体框架。
- 两个规范化决策是可观察契约：
  - `sort_keys=True`：JSON 对象键序不参与指纹（同一语义调用不因 dict 构造顺序漂移）；
  - `float round 6`：模型生成参数的浮点抖动（1e-7 级）不产生新指纹，但 1e-5 级差异视为不同调用。
- 判定语义：**调用前计数** `c`（counts_at_call）与 `c < threshold` 比较，放行才计数 +1；threshold=0 一步即熔断。

## 判题面

- oracle：`test_cases._py` 独立重写（Counter 计数、sorted keys 另一条递归路径、列表推导求首个熔断），随机 150 组对拍（n 1–10、threshold 0–3、args 深度 2、tool 池含 `Read`/`read` 大小写对）。
- 锚点：同指纹三步 threshold=2/3/0 三档账目；tool/args 不同独立计数；键序无关（含嵌套）；float 1e-7 同指纹、1e-5 异指纹；多处熔断取首个；指纹格式 `tool|` + 64 位十六进制（与手算 sha256 对照）。
- 非法输入：空 calls、条目缺/多键、空 tool、非 str tool、args 非 dict、threshold bool/负/浮点/str → ValueError。

## 变异记录（9 条全抓）

threshold-off-by-one（`<` → `<=`）、no-sort-keys、no-float-round、tool-lower（大小写合并，随机池 Read/read 抓）、sha1-instead、no-count-increment（恒置 1 永不熔断）、first-block-last（只记最后一步）、separators-default（json 默认分隔符）、allow-threshold-zero（0 阈值放行）。

## 红线

不引入真实工具执行、网络调用或框架依赖；与 `agent-react-loop`（循环状态机）划界：本题是**参数级指纹账**，不判定 ReAct 步进语义。
