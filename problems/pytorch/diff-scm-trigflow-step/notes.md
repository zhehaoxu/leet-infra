# 解题笔记：SCM trigflow 一致性步

## 回源定位

- 源文件：diffusers `schedulers/scheduling_scm.py`（305 行，Sana-Sprint 团队 2025）。核心只有 `step()` 的一小段（L274-297）：trigflow 的 `pred_x0` 与「跳回再重加噪」的 `prev_sample`。
- `set_timesteps`（L130-188）三条构造路径：自定义表（长度 N+1）、两步特表 `[max_timesteps, intermediate_timesteps=1.3, 0]`、默认 `linspace(max_timesteps, 0, N+1)`（max 默认 1.57080，注释说明 sCM 论文原值是 arctan(80/0.5)≈1.56454，diffusers 特意选了不同的值）。
- `step_index` 状态机是 EulerDiscrete 的样板复制（`_init_step_index`/`index_for_timestep`），其中「首个 step 遇重复值取第二个索引」的注释解释了 image-to-image 起点的语义——判题域外置为显式入参，只保留「允许重复值」的可观测行为。

## 数学要点

1. **trigflow 的两层含义**。角度参数化 `θ = arctan(σ/σ_data)` 把噪声调度压进单位圆：`x(θ) = cos(θ)·x0 + sin(θ)·σ_data·ε`。模型输出的隐含定义是 `F(x;θ) = (cos(θ)·x − x0)/sin(θ)`，代入 x 得 `F = σ_data·cos(θ)·ε − sin(θ)·x0`——即模型同时预测「去噪方向」与「数据残差」。这就是为什么 roundtrip 测试里合成 mo 必须是 `σd·cos(s)·eps − sin(s)·x0`，而不是 EDM 式的 `σd·eps`（第一版测试写错即在此处）。
2. **一致性步的几何**。`pred_x0 = cos(s)·x − sin(s)·F` 是把点从圆上拉回 x0 的内积反演；`prev = cos(t)·x0 + sin(t)·σd·ε'` 是把点重新放上角度 t。多步推理就是圆上的折线跳跃，t=0 的终点步恒等于 x0。
3. **σ_data 的角色**。它同时是角度定义（arctan 分母）、噪声幅度（randn 乘子）与训练目标的一部分；推理期只影响噪声项的幅度缩放。
4. **浮点精确性**。`torch.cos(0.0) = 1.0`、`torch.sin(0.0) = 0.0` 精确成立，所以末步 `prev == pred_x0` 可以用 tol=0 的逐位断言；`cos(1.57080)` 是 ≈7.97e-7 的极小数，非零——第一步仍会留下微弱的噪声残留，这也是断言里不能把 `cos(s) ≈ 0` 简写成 0 的原因。

## 实现要点

- 校验顺序：张量类型 → 形状一致 → timesteps（张量/一维/长度≥2/浮点/单调不增）→ step_index（int 非 bool、范围 [0, len−2]）→ prediction_type → sigma_data（数值且 >0）→ noise（t≠0 必须给、类型、形状）。错误消息判题域固定，reject 用例逐条断言消息子串（防空抓）。
- `_as_scalar_angle` 兼容 0 维张量与单元素张量（`timesteps[i]` 在 torch 里是 0 维）。
- 单调不增用 `(diff > 0).any()` 拒绝，保留相邻相等（官方 index_for_timestep 的重复值语义）。
- noise=None 且 t=0 时 `noise_term` 返回全零张量（`torch.zeros_like`），保证返回字典形状稳定、调用方可无条件对拍。
- 返回 `s_angle/t_angle` 为 float 标量，方便测试断言角度取位（s 取 step_index、t 取 step_index+1 是本题最容易被写反的点，专门的 swap 变异验证）。

## 测试要点

- 两步 Sana-Sprint 默认表全循环对拍（oracle 用 `math.cos/sin` 独立实现），step 1 的 oracle 输入行必须取 `prev_sample` 的**实际两列值**——不能假设输出仍保持输入的 `[v, v·0.5−0.3]` 构造模式（第一版即在此处对拍错位）。
- `linspace(θ_max, 0, 4)` 三步循环：覆盖中间角度的噪声项。
- 单步 `[θ_max, 0]`：noise=None 合法、`prev == pred_x0` 逐位、`noise_term` 全零。
- **noise 给了但 t=0**：换不同 noise 两次调用 `torch.equal` 逐位相同——「乘零」语义的直接观测。
- trigflow roundtrip：合成一致性（x0 恢复，tol=1e-12）。
- σ_data 缩放：`prev(2.0) − prev(1.0) = sin(t)·noise`（逐列，`_t` 构造的两列噪声不同）。
- 相邻相等角度表 `[0.9, 0.9, 0]`：s=t=0.9 的合法步。
- 纯函数性：同参数两次调用逐位相同（官方状态机外置的可观测后果）。

## 变异清单（31 条，全部 CAUGHT）

数学主体 11 条：cos/sin 互换（mo 与 sample 两侧各一）、mo 符号翻转、s/t 互换、t 用当前角度、噪声系数 sin→cos、σ_data 丢失、prev 的 cos→sin、noise_term 符号、zeros_like→pred_x0、返回值 s_angle→t_angle。索引/边界 3 条：上界松 1、下界松 1、bool 混过。校验删除/松动 14 条：单调翻转（拒相邻相等）、单调删、长度删、一维改二维、浮点校验失效、mo/timesteps 类型删、mo 形状删、noise 形状删、noise 必给删、noise 必给松（t≥0 恒拒）、σ_data 边界松、σ_data bool 混过、prediction_type 松动、标量 numel 校验翻。消息类 2 条：step_index 范围文案、noise 必给文案。

经验：31 条全部 CAUGHT、无等价变异——本题无随机数、无循环状态，纯确定性公式，等价风险天然低；变异重点放在「s/t 取位」这类结构性错误上（swap-st 与 t-cur 各抓 8 个用例）。m-tab-tensor 首轮 SURVIVED 的教训：删除 timesteps 类型校验后 numpy 表会以 AttributeError（非 ValueError）炸出，而当时表_validation 里没有「timesteps 传 np」的用例（np 传在了 mo 位置）——**每个类型校验都要有自己的专属 reject 用例，不能靠相邻参数的用例搭车**。
