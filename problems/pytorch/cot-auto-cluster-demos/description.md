# Auto-CoT 聚类选代表拼 demo

## 题意

few-shot CoT 的效果高度依赖示例质量。Auto-CoT（arXiv:2210.03493）用**聚类**自动构造覆盖面广的 demo 集：把问题池按语义向量聚成 k 簇，每簇选**一条**代表推理链拼成 demo——既省人工，又自动覆盖问题空间的各个分区。

给你（编码与聚类已在池外完成，`retrieval-kmeans-lloyd` 占 k-means 算子，故分配与距离显式入参）：

- `questions` / `rationales` / `pred_anses`：候选池（问题原文、zero-shot-CoT 推理链、预测答案），等长 list[str]；
- `cluster_assignment`：每条样本的簇编号（官方 KMeans.labels_）；
- `dist`：每条样本到各簇质心的欧氏距离矩阵 (n, k)（官方 clustering_model.transform）；
- `max_ra_len`：rationale 行数上限（默认 5，last_letters 类任务官方传 7）；
- `arithmetic`：是否启用算术任务的答案回验；
- `gold_anses`：可选金标答案（官方 debug 路径才收集）。

## 你要实现的管线

### 1. 分簇

第 i 簇收集 assignment==i 的样本，按下标升序 append，记录到质心 i 的距离与全局下标。

### 2. 簇内排序

按距离**稳定升序**（官方 sorted 默认 stable，距离平局保持全局下标序）。

### 3. 逐候选过滤（任一失败换下一候选）

```
a. 问题词数   len(question.strip().split()) <= 60
b. rationale 行数 len(r.replace("\n\n", "\n").split("\n")) <= max_ra_len
              # r = rationale.strip()
c. 以句点结尾 r.endswith(".")            # 官方 r[-1]=="."；空串不通过
d. 答案非空   pred_ans.strip() != ""
e. 算术回验   arithmetic 时：pred_ans 必须出现在
              r.split(".")[-2]（倒数第二个句子，前置条件 c 保证至少两段）
              或 r.split()[-10:]（最后 10 个词）中，否则跳过
```

### 4. 每簇至多一个

首个通过全部过滤的候选收为该簇 demo 并 `break`；簇内全部被过滤（或空簇）则该簇无 demo。簇按编号 0..k-1 依次处理。

### 5. demo 组装

```
rationale_clean = " ".join(r.replace("\n\n", "\n").replace("\n", " ").strip().split())
                # 官方清洗链；split() 按任意空白折叠成单空格
question / pred_ans 原样透传（官方的 "\nA:" 后缀与 "A: Let's think..." 替换属 log 解析层）
gold_ans = gold_anses[sid].strip()（给出时）或 None
```

返回 `dict(demos, selected, cluster_order, num_clusters)`：demo 列表、被选中样本的全局下标、各 demo 来源簇编号、簇数 k。

## 输入约束

- `questions`/`rationales`/`pred_anses` 等长非空 list[str]；`cluster_assignment` 等长 list[int]（bool 不算数）且值域 [0, k)；`dist` 是 (n, k) 的 list[list[float]]（bool 不算数），k ≥ 1；
- `max_ra_len` ≥ 1 的整数（bool 不算数）；`arithmetic` 是 bool；`gold_anses` 给出时等长 list[str]。

## 提示

- 过滤 e 的 `r.split(".")[-2]` 在条件 c 之后才安全——c 保证 r 至少含一个句点，split 至少两段；
- 偏好堆空换堆、`random.shuffle` 采样（sampling≠center）与 log 文件解析（Q:/A:/pred_mode/GT : 行协议）不在判题域；
- 距离矩阵中**非所属簇的列是死数据**（官方只读 `dist[sentence_id][cluster_id]`），但读错列会改变簇内排序。
