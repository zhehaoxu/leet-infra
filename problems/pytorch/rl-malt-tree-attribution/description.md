# MALT：搜索树成败归因账（791）

**来源**：arXiv:2412.01928（COLM 2025 v3）§4.2 / A.6.1。**官方无开源实现**——本题口径 = 论文公式唯一确定的算术账，沙箱 oracle 为公式的独立 numpy 直译 + 手工锚点。

## 题目大意

多智能体 Generate→Verify→Refine 流水线的信用分配不该只看终局对错：把每题 n³ 条采样轨迹组织成三层搜索树，用值迭代（自下而上的兄弟期望）把稀疏叶奖励回传到每个中间节点，0.5 严格阈值二值化（对齐多数投票），再按同父上下文构造 DPO 偏好对。模型采样本身不出题——叶奖励与树结构全部显式入参。

## 求解什么

`solve(mode=..., ...)` 五入口（mode kwarg 分发，纯标准库）：

1. `mode="propagate"`（rewards: n×n×n 二元列表, n）→ (verifier_values n×n, generator_values n)
2. `mode="binarize"`（values[, theta=0.5]）→ 严格 > θ 的布尔标签
3. `mode="pairs"`（unit_values, parent_ids[, theta]）→ (正下标, 负下标) 对列表
4. `mode="sft"`（unit_values, parent_ids[, theta]）→ {positives, counts}
5. `mode="attribute"`（rewards, n[, theta]）→ 单题全流程（值/标签/三层配对/SFT 对账）

## 关键论文口径（详见 notes.md）

- **无折扣因子**：值迭代 = 两层样本均值（V(v)=mean of leaves、V(g)=mean of V(v)）
- **0.5 阈值严格大于**：V 恰等于 0.5 判 ×（n 偶数时兄弟均值可取到；n=3 取不到）
- **同父配对**：D_R 同 verifier 输出、D_V 同 generator 输出、D_G 同题；同父内正×负子集全组合
- **Generator 只 SFT**：D_G 有定义但论文明确 Generator 不做 DPO
- SFT 正样本分层口径：R 层用叶奖励 V(r)=1、V/G 层用二值 label ✓

## 计分

判题即账本比对：期望回传用独立 numpy 直译对拍 + 手工锚点（7/9、2/9 等），配对/标签全手工计数锚定。

## 判题环境

torch + numpy；无随机性（全确定性账，无模型调用）。
