# notes — ag-generative-memory-retrieval（order 666）

## 回源

- 出处：`joonspk-research/generative_agents` 的 `cognitive_modules/retrieve.py`（`new_retrieve`；论文 2304.03442「Generative Agents: Interactive Simulacra of Human Behavior」）。
- 官方 quirk 忠实转写：`extract_recency` 用 `decay ** i for i in 1..n` 配按时间升序的 nodes，**最旧记忆 recency 值最大**——与"越新越近因"的直觉相反，但这是上游原样行为，题面钉死不修正。
- `normalize_dict_floats(d, 0, 1)`：min-max 到 [0,1]；`range == 0` 时全部置目标区间中点 0.5（单节点记忆由此得 0.5 而非 0/1）。
- 权重 `gw = [0.5, 3, 2]`（recency_w/relevance_w/importance_w 均为 1 时的默认组合）。

## 判题面

- solve 入参：query_emb、memories（emb + importance）、k、可选 recency_decay/gw。返回 scores/top/components。
- oracle：`test_cases._py` 独立重写（手动展开归一化与排序），随机 150 组对拍（n 1–12、decay ∈ {0.995, 0.99, 1.0}、k 可越界截断）。
- 锚点：手算三分量与总分（cos 精确取 1/2：emb=[1,√3]、q=[1,0]，1+3=4 → √4=2 全程浮点精确）；importance 全等 range=0 → 全 0.5；单节点 → score=(0.5+3+2)*0.5=2.75；全并列 + gw[0]=0 → 总分全并列 top=[0,1,2]（钉稳定排序）；k 截断前缀性质。
- 非法输入：空 query/memories、维度不齐、非数、NaN/Inf importance、k 非正/bool、decay 越界、gw 长度≠3 → ValueError。

## 变异记录（9 条全抓 + 1 条等价）

recency-direction-reversed（新旧序反转）、gw-rel-imp-swap、norm-range0-to-zero（0.5 改 0）、cos-no-norm（点积代余弦）、sort-ascending、k-plus-one、importance-no-norm、recency-no-norm、tie-reverse-index（并列取大下标，靠 gw[0]=0 的总分全并列用例抓住）。

**等价变异**：`decay ** (i+1)` → `decay ** i` 不可观察——整体乘常数被 min-max 归一化约掉。已换成方向反转变异。

## 红线

embedding 模型调用、检索工程化（向量库/ANN）不出；与 `int-*`（interpretability 读出）划界：本题是**智能体记忆账**，无模型内部信号。
