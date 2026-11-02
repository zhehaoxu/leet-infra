# TextGrad 文本反传图账

TextGrad（arXiv 2406.07496）把「自然语言反馈」当成梯度：`Variable` 是计算图节点，
每次算子调用建一个新节点并挂一个 `grad_fn`，`root.backward()` 沿图把反馈反过来推。
本题把这套自动微分剥成纯函数——LLM 生成、LLM 评分全部变成显式入参，
只保留**建图、遍历、梯度累积与归约**。

官方实现（`zou-group/textgrad` main）里所有容器都是 `set`：`predecessors`、
`gradients`、角色描述集合都不保证迭代序。本题把这三处统一规范成
「按加入顺序去重」，其余规则逐条对齐官方代码。

实现 `solve(nodes, grads, reduces=None, root=None, rounds=1)`。

## 图与节点

`nodes` 按**建图顺序**给出，`id` 唯一：

```
{"id": "q", "kind": "input", "value": "3 * 7", "role": "question",
 "requires_grad": True, "inputs": [...], "system": "sys", "group": "G"}
```

- `kind = "input"`：叶子，`value` / `role` 必给，`requires_grad` 默认 `True`。
- `kind = "llm"`：一次 LLM 调用，`inputs` 是前驱 id 列表，可选 `system` 指向系统提示词节点；
  `value` 是这次调用的生成结果。
- `kind = "sum"` / `"aggregate"`：把若干前驱的文本拼起来，`inputs` 可为空列表。
- `kind = "add"`：`Variable.__add__`，**恰好两个**操作数。
- 只有 `aggregate` 用 `group`（默认等于自己的 id），其余节点忽略该字段。

## 前向建图

- 值：`input` / `llm` 取声明里的 `value`；`sum` / `add` / `aggregate` 是把前驱的值用换行
  `"\n"` 连起来。
- 角色文本三者不同（它会原样嵌进后向的反馈模板，所以必须精确）：
  - `sum`：`a combination of the following: <去重后的角色，逗号+空格>`
  - `add`：`<左操作数角色> and <右操作数角色>`（**不去重**）
  - `aggregate`：`a combination of the following variables: <去重后的角色>.`（结尾有句点）
- `requires_grad`：`sum` / `aggregate` 是「任一前驱为真」，`add` 是左右相或。
- `llm` 的前驱序是 `[system] + inputs`——系统提示词排在最前面。
- `aggregate` 把自己的归约标记 `{"op": "mean", "id": group}` 挂在**自己**身上。
- 硬约束：`requires_grad=False` 却带需要梯度的前驱、`add` 的操作数不是 2、`id` 重复、
  前驱 id 未知、`root` 未知、空图、`rounds < 1`，一律 `ValueError`。

## 反传顺序

从 `root`（默认最后一个声明的节点）开始，沿前驱按声明序做 DFS **后序**，再**逆序**逐节点处理。

- 只清空**根节点**的梯度；中间节点保留上一轮残留（`rounds > 1` 时会继续累积）。
- `requires_grad=False` 的节点整段跳过：不清梯度、不调 `grad_fn`，在 `skipped` 里记一笔。
- 其余节点先「检查并归约自己的梯度」，再调自己的 `grad_fn` 往后推。
- 每轮都重新做一遍拓扑；`order` 是各轮处理顺序的拼接。

逆后序意味着兄弟支路是**先声明、后处理**，这会直接影响同一个输入收到的梯度先后。

## 三个后向

- `sum` / `add`（幂等广播）：设本节点梯度文本为 `T`。对**每个**前驱推
  - `T == ""` 时推空串，否则推
    `Here is the combined feedback we got for this specific <前驱角色> and other variables: <T>.`
  - 而且每个前驱要推**两条完全一样**的梯度（官方连着 `add` 两次）；
    只有**第一条**带本节点的归约标记，同时把标记追加到前驱节点上，第二条不带。
- `aggregate`：只给**声明序最后一个**前驱推一条，带自己的归约标记
  （官方 `for` 循环里只赋值不推送，循环结束后才推，于是只有一个前驱真正收到梯度）。
- `llm`：响应自身梯度文本为空走 `base` 路，否则走 `chain` 路；跳过 `requires_grad=False`
  的前驱；每个前驱只推一条，值取自 `grads["<本节点>:<base|chain>:<前驱>"]`。

## 梯度归约

处理某节点时，若它带归约标记、梯度文本非空、且**梯度多于一条**，就进入归约：

1. 按每条梯度**自带**的标记 id 分组，组按首次出现排序；
2. 每组发一次总结调用，提示词是把组内每条梯度包成 `<FEEDBACK>值</FEEDBACK>` 后用换行连接；
3. 归约结果的值取自 `reduces[group]`，角色是组内角色去重后逗号连接；归约出的梯度**不带标记**；
4. 没有标记的梯度在这一步被丢弃；一条组都凑不出来时结果为**空列表**，也不会发总结调用。

## 输出

```
{
  "order": ["a", "q"],
  "skipped": [],
  "steps": [{"round": 1, "id": "a", "skipped": false, "in": [], "out": [],
             "reduced": false, "called": true, "path": "base",
             "pushed": [{"to": "q", "count": 1}]}],
  "grad_text": {"q": "...", "a": ""},
  "grad_count": {"q": 1, "a": 0},
  "values": {"q": "3 * 7", "a": "20"},
  "roles": {"q": "question", "a": "answer"},
  "reduce_prompts": ["<FEEDBACK>...</FEEDBACK>"],
  "rounds": 1
}
```

- `steps` 按处理顺序记录：`in` 是归约前的梯度值列表，`out` 是归约后，`reduced` 表示这一步
  是否走了归约分支，`called` 表示是否调了 `grad_fn`，`path` 只有 `llm` 节点有。
- `grad_text` 是每节点最终梯度文本（多条用换行连接），`grad_count` 是条数。
- `reduce_prompts` 是**实际发出**的总结提示词，按调用顺序。

## 样例

```
nodes = [
  {"id": "q", "kind": "input", "value": "3 * 7", "role": "question"},
  {"id": "a", "kind": "llm", "inputs": ["q"], "value": "20", "role": "answer"},
]
grads = {"a:base:q": "the question asks for 21"}
led = solve(nodes, grads)
led["order"]        # ["a", "q"]
led["grad_text"]    # {"q": "the question asks for 21", "a": ""}
led["grad_count"]   # {"q": 1, "a": 0}
```

## 约束

- 纯函数：不联网、不读写文件、不修改入参；同样的输入必须给同样的账本。
- 归约结果、LLM 梯度都只能来自入参表，缺键时抛 `KeyError`。
- 建图期违规抛 `ValueError`。
