# 解题说明 · ReST-KV 的赔率分与重建乘子

## 一句话

末窗注意力 → 赔率 `p/(1−p)` → 乘上「这个 value 离代表 value 有多远」→ 一维池化平滑 → top-k，四个阶段全是记账，没有任何训练量；考点在两处**顺序**（先聚合还是先算分）与两处**死开关**（`use_norm` 只在 `before`、`tau` 只在 `after + use_wo`）。

## 为什么是这套账

SnapKV 只按「末窗 query 对该 key 的平均注意力」排序，ReST-KV 认为这不够：一条 key 被关注得多（赔率大）**且**它的 value 与历史压缩出来的代表值差得远（重建误差大），才真的不可丢。`v̄` 是「假装整段历史只剩一个 token」时的输出，`‖W_o v̄ − W_o V_l‖₂` 就是丢掉它带来的输出差——把它写成对**逐层输出重建**的乘子，正是论文标题里的 layer-wise output reconstruction。

`before` 与 `after` 是同一句话的两种展开：`after` 先把 `W` 行平均成 `p̄` 再取赔率与差范数，`before` 逐 query 算完整分数再平均。`p/(1−p)` 是凸函数，Jensen 不等式保证两条路的分数不会成比例，排序可以不同——本题把两条都当合法输入，用同一个 oracle 判。

## 四个易错口径

- **`before` 的代表 value 用全量 cache**：`O = A @ value` 里的 `A` 是 `(B,H,W,S)` 整段、`value` 是全部 `S` 个位置，切 `[:-W]` 发生在分数算完之后；`after` 的 `v̄` 只由历史段 `value[:, :, :-W]` 加权，且权重和 `< 1`（不是凸组合）。
- **`after` 不读 `use_norm`，`before` 不读 `tau`**：上游 `metric_mode == "before"` 分支里根本没有 `self.tau`，`"after"` 分支里根本没有 `self.use_norm`；`use_wo=False` 时 `after` 的差范数也不带幂。这三个死开关是本题的判分信号，不是我们加的规矩。
- **EMA 只回吐最后一个状态**：`ema()` 沿 query 轴递推后 `squeeze(dim)`，返回的是 `(B,H,L)` 而不是整条轨迹；`alpha` 是**最新项**权重，`alpha=1` 塌成「只看最后一个 query」，`alpha=0` 塌成「只看第一个」。
- **池化两端把补零算进分母**：`F.avg_pool1d` 默认 `count_include_pad=True`，所以首末位的分母仍是 `kernel_size`；`L=1` 时窗口只有 1 个真实值，`avg` 会把它压成 `1/ks`，而 `max`（补 `-inf`）保持原值——两者在退化段上不等价。

## 本题自己的口径

- K/V 按 query head **已展开**后传入（上游调用点如此，`RestKVCluster` 内部不做 `repeat_kv`），`H` 恒为 query head 数。
- **分数段永远算**：`S < max_capacity_prompt` 时上游整步早退，本题仍回吐 `attn_weights / important_score / pooled_score`，只把选择段清空，好让容量门槛单独可判。
- top-k 钉成**稳定降序 + 同分取下标小的**：上游用 `topk(...).indices`，torch 不承诺同分次序；`maxpool` 会批量制造精确同分（相邻窗口的最大值来自同一个元素），所以必须钉。
- 掩码常量按注意力张量自己的 dtype 生成：上游 L608 造 `mask` 时没传 `dtype`，float64 输入会在 `torch.full` 处直接 `RuntimeError: value cannot be converted to type float without overflow`——上游只能跑 float32。本题两种 dtype 都放行，可观测结果（被掩位置的概率恰为 0）不变。
- `wo` 即使 `use_wo=False` 也照样校验形状：把它当「传了但不用」的入参，比允许 `None` 更容易判。

## 判分点

5 键。三个浮点键与输入同 dtype、同不就地改写；`selected_indices` 是历史段内下标（0 起即绝对下标）且按**选择序**；`kept_positions` 前 `k` 个跟选择序、末窗 `W` 个升序。`window_size >= S`、`max_capacity_prompt <= window_size`、`kernel_size` 为偶数都要报错——偶数核配上 `padding=k//2` 会让输出比输入多一格，上游那里是静默错位，本题直接拒。

## 复杂度

`before` 支的差范数张量是 `(B,H,W,S,C_out)`，主项 `O(B·H·W·S·C_out·D)`；`after` 支只到 `(B,H,L,C_out)`。判题用例控制在 `S <= 11`、`W <= 6`、`D <= 3`、`C_out <= 6`，随机对拍 12 组。

## 出处

- 仓库 `an-yongqi/rest-kv`；`restkv/restkv_utils.py`：`RestKVCluster` L521-699（打分主体 L617-656、池化 L658-680、top-k 与拼接 L683-698），`ema` L287-296，消融版 `AblationCluster.update_kv` L313-405（`ts`/`ss` 命名不同但同一套数学），`init_restkv` L1229-1256 的 `window_size=32 / max_capacity_prompt=4096` 兜底。
- README L66 与 L83-89 给出论文配置 `--use_wo --use_ema --alpha 0.3 --metric_mode after --pooling adaptive --scale 2000`，L98-106 逐个写明默认值（`tau` 只在 `after` 生效那句是 README 自述，与代码一致）。
- arXiv 2605.08840（2026-05-09）由 arxiv.org 摘要页的仓库链接亲见；ICLR 2026 proceedings 有同名 PDF。
- 对拍证据（本轮在 CPU 上跑）：把 `F.avg_pool1d` / `F.max_pool1d` 打桩抓出上游内部的平滑前/平滑后两张分数阵，15 seed × {before,after} × `use_wo` × `use_norm` × `use_ema` × {avg,max} 共 480 次 `torch.equal` **逐位相同**；再用返回的 cache 反解选择序，192 组里除 `maxpool` 的同分次序外全部一致。
- 剥除项：`pooling="adaptive"`（L662-674，硬编码 `attn_weights[0]`、核宽由注意力排序下标差算出）、`use_pyramid`（L591-602，与 `kv-pyramid-layer-budget` 重复）、`merge_kv`（L120-171）、`AblationCluster` 的 `indicator='attn'/'attn-v'/'random'` 三支（前两支就是 SnapKV、第三支随机）。

## 相关题目

`kv-snapkv-pooling`（同一个末窗注意力，但没有赔率与重建乘子）、`kv-rkv-redundancy-decay`（同样把注意力分与其他项相乘/相加后再池化 top-k）、`kv-pyramid-layer-budget`（被剥掉的逐层预算）、`kv-ott-outlier-token-trace`（另一条「按幅值挑少数 token 特殊保留」的路）。
