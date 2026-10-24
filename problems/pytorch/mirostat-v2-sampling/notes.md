# 解题说明 · Mirostat v2

## 一句话原理

维护一个 surprisal 阈值 $\mu$：**截断**比 $\mu$ 更意外的尾部 token → **采样** → 用抽中 token
的观测困惑度 $\text{obs}$ **反馈**调节 $\mu$，让长期困惑度收敛到目标 $\tau$。

$$s_i=-\log_2 p_i,\quad \text{keep 降序前缀直到 } s_i>\mu,\quad \mu'=\mu-\eta(\text{obs}-\tau)$$

- v2 相对 v1 的简化：v1 要维护对数域的双重反馈，v2 只需**一个**标量 $\mu$、一套
  「截断→归一化→采样→更新」即可，工程上更受欢迎。

## 截断（关键）

C++ 里 `cur_p` 已按 $p$ 降序，`std::find_if(-log2f(p) > mu)` 返回**第一个越界**元素，
`size = distance(begin, it)` 就是保留的前缀长度：

- 越界发生在第 $k$ 个 → 保留 $k$ 个（$0..k-1$ 都不越界）。
- 首元素就越界（$k=0$）→ `size==0` 被夹成 1，永远保留最 probable 的那个。
- 全都不越界 → 保留全部。

本题对最后一维用 `torch.sort(descending=True)` 复现这个降序，再用
`size = clamp(argmax(surp>mu) if any else n, min=1)` 精确对齐。

## 归一化 / obs / 反馈

保留集合重归一化得到 `filtered`。注意 **obs 用归一化后的** $\text{filtered}[\text{chosen}]$
（C++ 里两次 softmax：先截断前的原始 p 定阈值，再对截断集重新 softmax，`obs` 取的是
重归一化后的 p）。`mu_new = mu - eta*(obs - tau)`：`obs` 越大（越意外）→ `mu_new` 越小 →
下次截断越狠，闭环把困惑度拉回 $\tau$。

## 与合集契约的一致性

`filtered` 按**原顺序**返回、`chosen` 用原顺序 CDF 逆变换得到（与合集内其它题统一）；
额外多返回一个 `mu_new` 作为下一步的 `mu` 入参。批量时对每行独立截断、每行各得一个 `mu_new`。
