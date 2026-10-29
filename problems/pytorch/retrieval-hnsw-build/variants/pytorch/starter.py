import torch


def solve(points, levels, M=16, ef_construction=200):
    """HNSW 建图：按插入序逐点插入，显式层级 + Alg.4 启发式邻居选择。

    入参：
    - points: (N, D) 浮点张量，dtype 为 float32 或 float64；N >= 1、D >= 1。
      所有距离都用平方欧氏距离。
    - levels: (N,) int64 一维张量，levels[i] >= 0。它是节点 i 的最高层，
      节点 i 存在于第 0..levels[i] 层（第 0 层是最底层）。层级完全由该
      入参给出，不做随机 level 采样。
    - M: int，默认 16，>= 1。第 1 层及以上的出度上限为 M；第 0 层的
      出度上限为 M0 = 2 * M。
    - ef_construction: int，默认 200，>= 1。每层候选搜索的宽度；候选池
      实际大小不超过该层可达节点数。若 ef_construction < M，新点在该层
      最多只能拿到 ef_construction 个邻居，不补足。

    建图过程（节点按插入序 0, 1, ..., N-1 编号）：
    - 节点 0 直接作为初始入口点 entry，当前最高层 max_level = levels[0]。
    - 对每个新点 q（l = levels[q]）：
      1) 令 ep = entry。对 lc 从 max_level 递减到 l+1，在 lc 层做宽度
         ef=1 的候选搜索，把 ep 更新为搜索结果中离 q 最近的点（距离
         并列取节点号小者）。这一段是上层贪心下降。
      2) 对 lc 从 min(max_level, l) 递减到 0：
         a. 以 ep 为起点、宽度 ef_construction 做候选搜索，得到候选集 W；
         b. 对 W 做 Alg.4 启发式选择：候选按 (到 q 的平方距离, 节点号)
            升序排列，依次考察；若候选 c 到已选集合 R 中任一节点 r 的
            平方距离严格小于 c 到 q 的平方距离（dist(c,r) < dist(c,q)），
            则剪掉 c；否则选入 R。选满 M 个即停止（不足则全选）。
         c. 对 R 中每个 e 加双向边 q-e；不允许自环、不允许重复边。
         d. 对 R 中每个 e 检查出度：cap = 2*M（lc=0）或 M（lc>0）。若
            degree(e) > cap，反复删除 e 的邻居中「到 e 的平方距离最大」
            者；距离并列时删节点号大者。刚加入的 q 同等参与竞争，因此
            最终图允许不对称（q 的列表里可能仍有 e，而 e 的列表已删 q）。
         e. 把 ep 更新为 W 中离 q 最近的点，供下一层使用。
      3) 若 l > max_level，则更新 entry = q、max_level = l。
    - 候选搜索（search_layer）的确定性口径：已访问集合从 {ep} 开始；
      候选集按 (到 q 的平方距离, 节点号) 取最小；结果集 W 保留至多
      ef_construction 个按 (距离, 节点号) 最优的节点。每步取出候选集中
      最优者 c，若 dist(c,q) 严格大于 W 中当前最差距离则停止（相等仍
      继续扩展）；否则按节点号升序遍历 c 在该层的邻居，未访问者标记
      访问，若 |W| < ef 或 (dist(e,q), e) 优于 W 的最差 (距离, 节点号)，
      则同时加入候选集与 W，并在 |W| > ef 时弹出最差者。

    除 SEARCH-LAYER 的停止条件只看距离（相等继续扩展）外，所有排序、
    候选进入 W 的比较、argmin 与溢出删除都以 (平方欧氏距离, 节点号)
    为键，节点号小者优先；Alg.4 剪枝本身用严格的 dist(c,r) < dist(c,q)。
    全程无随机数。

    返回：
    - 三层嵌套的 Python list：out[lc][i] 是节点 i 在第 lc 层的邻居
      节点号列表。外层长度恰为 L = max(levels) + 1（第 0 层到最高层），
      中层长度恰为 N，内层按节点号升序排列、无重复、无自环，且
      len(out[lc][i]) <= 2*M（lc=0）或 <= M（lc>0）。节点 i 在
      lc > levels[i] 的层不存在，对应列表为空。节点 0 单独插入时所有
      层都是空列表。

    非法输入一律抛 ValueError：points 不是二维张量、dtype 不是
    float32/float64、N=0、D=0；levels 不是 int64 一维张量、长度不等于
    N、含负值；M 或 ef_construction 不是正整数（bool 也不接受）。
    """
    raise NotImplementedError
