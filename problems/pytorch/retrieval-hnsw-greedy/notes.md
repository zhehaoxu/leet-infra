# 详解：HNSW 贪心图层搜索

## 出处
Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*（arXiv:1603.09320 / TPAMI 2018）。论文的 `GREEDYSEARCH`（Algorithm 4 的前身、`SEARCHLAYER` 在 \(ef=1\) 时）即「沿边跳到更近邻居直到局部最优」。插入时逐层下降用贪心找 \(ef=1\) 入口，再在最底层用 \(ef\) 队列精搜——本题只做贪心子过程。

## 核心与易错点
1. **严格下降、一步一跳**：只有当「最近邻居」比当前点更近才移动；用 `d < cur_dist` 严格比较。距离单调下降 → 不会成环、必然终止。
2. **只看当前点的邻居**：这是纯贪心（\(ef=1\)），**不是**带候选队列的 `SEARCHLAYER(ef)`——后者会保留 \(ef\) 个待扩展点、能跳出更多局部最优。别把二者混为一谈。
3. **贪心会卡在局部最优**：图不连通或邻居方向不对时返回的未必是全局最近——这正是本题的「可核验陷阱」，测试专门放了一个卡局部最优的用例。
4. **邻居表用 `-1` 哨兵**：跳过无效槽；等距邻居取下标最小者。
5. 距离平方欧氏即可，`cd` 每轮必须基于新 `cur` 重算（缓存旧值会导致永不满足「更近」而反复横跳）。

## 对拍口径
参考解 while 循环：每轮 `cd=dist2(cur)`，扫描 `adj[cur]` 有效邻居取 `dist2` 最小者，若 `< cd` 则 `cur=argmin邻居`，否则返回 `cur`。判题 `_oracle` 用逐维纯 Python 累加平方差、邻居候选 `sort key=(dist,index)` 取首个（结构不同、数值等价）。另设「一维链走到全局最近」「某入口即局部最优立即返回」「图使贪心卡在次优点（≠全局最近）」断言。

## 复杂度
最坏访问 \(O(n)\) 个节点、每节点扫 \(O(\text{deg})\) 邻居、每距离 \(O(D)\)：\(O(n\cdot \text{deg}\cdot D)\)；实际在 NSW 图上近 \(O(\log n)\)。
