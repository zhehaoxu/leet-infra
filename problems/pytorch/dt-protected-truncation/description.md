# 分词：保留特殊 Token 的预算截断

## 任务

句对模板已经插入特殊 token。给定每个位置的 token ID、特殊标记和序列归属，在总长度预算内裁剪普通 token，同时保留全部特殊位置和所有字段的对齐关系。这里不执行 tokenizer、模板拼接或 batch padding。

三个列表等长，长度为 0..200。特殊位置的 `sequence_ids` 为 -1，普通位置为 0 或 1；归属可以交错。ID 为 0..10^6 的整数。特殊身份仅由 `special_mask` 决定，即使普通 token 与特殊 token 数值相同，也不获得保护。

## 预算分配

`max_length` 必须是非负 Python 整数，不接受 bool。令原长为 L，需要删除 `max(0,L-max_length)` 个普通 token。

- `only_first`：只删除序列 0 的普通 token。
- `only_second`：只删除序列 1 的普通 token。
- `longest_first`：每一步从**剩余普通 token 较多**的一侧减去一个，数量相同时选序列 1。特殊位置不计入两侧长度。

若特殊位置本身超过预算，或指定侧没有足够的可删 token，抛出 `ValueError`。未知策略、未知保留模式、非法预算也抛出 `ValueError`，包括无需裁剪的情况；其余输入合法。

## 位置选择

先确定每侧保留数 q，再按该侧原有顺序选择位置：

- `head`：保留前 q 个。
- `tail`：保留后 q 个。
- `middle`：**删除中部**，保留前 `ceil(q/2)` 个与后 `floor(q/2)` 个；不是保留中间一段。

q 为零时该侧一个普通 token 也不保留。最后把两侧选中的位置与全部特殊位置按原始下标合并。不要移动特殊 token、重新编号序列或修改输入。

返回恰有五键的字典，值全部为 Python 列表：`input_ids`、`special_mask`、`sequence_ids` 是保留字段；`kept_indices` 是升序原始位置；`removed_counts` 是 `[序列0删除数,序列1删除数]`。

## 实跑样例

```text
input_ids    = [101,10,11,12,13,14,102,20,21,22,23,102]
special_mask = [true,false,false,false,false,false,true,false,false,false,false,true]
sequence_ids = [-1,0,0,0,0,0,-1,1,1,1,1,-1]
max_length = 8
strategy = "longest_first"
keep = "middle"
```

```text
input_ids      = [101,10,11,14,102,20,23,102]
special_mask   = [true,false,false,false,true,false,false,true]
sequence_ids   = [-1,0,0,0,-1,1,1,-1]
kept_indices   = [0,1,2,5,6,7,10,11]
removed_counts = [2,2]
```

三个特殊位置占用预算，普通位置由 (5,4) 减为 (3,2)。第一侧保留头两个与尾一个，第二侧保留头尾各一个。

## 范围限定

这是明确约定的保护位置与句对预算算法。`middle`、已插入模板上的保护掩码及任意交错序列是本题规则，不声称等价于某个 tokenizer 的完整截断接口。它区别于 `sequence-pack-pad` 的多序列打包，也区别于 few-shot 示例级预算装填。
