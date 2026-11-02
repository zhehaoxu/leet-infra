"""TextGrad 的文本反传图账。

把 TextGrad（arXiv 2406.07496）的 `Variable.backward()` 摊成一份可逐项核对
的账本：建图 → 逆后序遍历 → 逐节点先归约自己的梯度、再把梯度推给前驱。

节点用声明序（= 建图序）的字典列表给出：

```
{"id": str, "kind": "input"|"llm"|"sum"|"add"|"aggregate",
 "value": str,              # 只有 input / llm 需要给
 "role": str,               # 只有 input / llm 需要给
 "inputs": [id, ...],       # llm / sum / add / aggregate 的前驱（按声明序）
 "system": id,              # llm 可选的系统提示词节点
 "requires_grad": bool,     # 默认 True
 "group": str}              # aggregate 的归约分组，默认等于自己的 id
```

你实现的 `solve(nodes, grads, reduces=None, root=None, rounds=1)` 返回：

```
{
  "order": [str],                  # 逐节点处理顺序（DFS 后序的逆序，按轮拼接）
  "skipped": [str],                # 因 requires_grad=False 整段跳过的节点
  "steps": [{"round", "id", "skipped",
             "in": [str], "out": [str], "reduced": bool,
             "called": bool,       # 是否调用了 grad_fn
             "path": "base"|"chain",   # 仅 llm 节点
             "pushed": [{"to", "count"}]}],
  "grad_text": {id: str},          # 每节点最终的梯度文本
  "grad_count": {id: int},         # 每节点最终的梯度条数
  "values": {id: str},
  "roles": {id: str},
  "reduce_prompts": [str],         # 每次「总结」调用实际发出的提示词
  "rounds": int,
}
```

`grads` 是 LLM 后向产出的梯度文本表，键为 `"<节点>:<base|chain>:<前驱>"`；
`reduces` 是归约分组的总结结果表，键为 group id。

参数与规则见 `solution.py` 的 docstring 与题面。
"""


def solve(nodes, grads, reduces=None, root=None, rounds=1):
    raise NotImplementedError
