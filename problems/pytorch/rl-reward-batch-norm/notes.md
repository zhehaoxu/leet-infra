# 详解：奖励归一化的批统计管道

## 出处与背景
- OpenRLHF 在 rollout 之后调 `reward_norm(...)`，开关 `group_norm` 有 `prompt / group / running_state / batch / none` 五档——**"用哪个统计量"是一个调度选项，不是数学选项**；同时维护一个 `RunningMoments` 对象当 `running_state` 档的数据源。verl 侧对应的是 advantage/reward 归一化开关（GRPO 走组内统计，`norm_adv_type`/`use_kl_std` 之类决定除不排除 std），机制一样。
- TRL 的 PPO trainer 同样用 `RunningMoments` 同时归一化 KL 与奖励，默认折扣 `gamma=0.99`。
- 本题抽出的机制是**数据流**：按到达顺序排好的 $T$ 个批，每批"选统计源 → 归一化 → 记账本"，外加一个 `out_rms` 探针把"尺度失配"这种静默事故变成可见数字。全程都是给定向量的加减乘除，不含任何优势/损失推导。

## 四条必须钉住的口径
1. **更新顺序：先归一化、后写账本**。第 $t$ 批在 `stat_source=1` 时用 `running_mean[t] / running_var[t]`（本批尚未写入时的值），账本在第 4 步才更新。这与 BatchNorm 推理/训练一致，也是"第一批等于没归一化"（示例里 `normalized[0]` 就是 $x/\sqrt{1+eps}$）的原因。写成"先更新再用"会整体平移一格，对拍立刻挂。
2. **账本永远由批全局统计驱动**，与 `stat_source` 无关。`stat_source` 只回答"拿什么来除"，不回答"往账本里写什么"。示例里三种 `stat_source` 的 `running_mean/running_var` 完全相同，测试 `scale_mismatch_probe_fires` 就是拿这条当断言。
3. **动量口径是 PyTorch BatchNorm 的 $m$，不是 TRL 的 $\gamma$**：本题 $r\leftarrow(1-m)r+m\,x$；TRL/OpenRLHF 的 `RunningMoments` 是 $r\leftarrow\gamma r+(1-\gamma)x$，$\gamma=0.99$ 等价于本题 $m=0.01$。把别人的 $\gamma$ 直接抄进 $m$，账本收敛快 100 倍，早期批会被自己污染——这是本族最经典的事故。另外 TRL 版方差更新还额外补一项均值迁移修正（形如 $(1-\gamma)\big(v_{new}+\gamma(\bar x_{old}-\bar x_{new})^2\big)$），并用 `var(unbiased=True)`；本题**刻意不补、且统一用总体方差 ddof=0**，就是为了让"除 $n$ 还是除 $n-1$"这一处口径显式写在题面上而不是藏在库函数里。
4. **组统计是批内组**：`group_ids` 长 $B$，各批复用同一组号（组不是全局样本号）。组大小为 1 或组内全相等 ⇒ 分子与方差同时为 0 ⇒ 该项**恰为 0**（不是 $\pm1$，因为 $\sqrt{0+eps}$ 除的是 0）；组号稀疏（如 `[3,1,3,1]`）不影响结果，未出现的组号不会被任何条目引用。

## `out_rms` 探针为什么是题面输出
$\sqrt{\operatorname{mean}(z^2)}$ 在真正白化后 $\approx1$；`whiten_std=0` 时它退回原始尺度量级。它比"看 loss"更早暴露三类事故：源选错（`stat_source=1` 遇上奖励尺度跳变，题面示例里已经到 23 倍）、动量写反（`momentum=0` 账本冻死，$rms$ 一路走高）、eps 量级选错。真实训练器里这就是 `reward_norm` 后监控的 `std`/`1` 比值。

## 常见坑
- 把 `stat_source=1` 的"更新后再取"和"取完再更新"写反（口径 1）。
- 用样本方差（除 $B-1$）：$B=4$ 时尺度差 $\sqrt{3/4}$，示例里的 $\pm1.341635$ 会变成 $\pm1.161895$，一眼对不上。
- 组统计写成"全局按组号累积"：本题组是**批内**的，每批重新算，跨批不累计。
- `whiten_std=0` 时还去除 std（应只减均值），或 `eps` 加在 std 外面。
- 忘记 `running_*` 长度是 $T+1$（第 0 项是初值 $0.0/1.0$，最后 $T$ 项才是逐批更新后的状态）。
- 方差为 0 时先开方再除：`eps>0` 已经保证不除零，但分子为 0 时结果必须是**精确 0**，别用 `where(var>0, ..., 0)` 之外的分支把 $0/\sqrt{eps}$ 写成别的值。

## 与其他题族的分工
- `to-grpo-advantage`（order 213）推导"组内标准化成优势"这条**公式**：单批、单一来源、直接 $\frac{r-\mu_g}{\sigma_g}$。本题不出现任何优势定义，只问"统计量从哪来、什么时候更新、更新在归一化前还是后"，并且允许来源是批全局/账本/组三种。
- `to-gae`（order 211）是时序递推；`rl-trajectory-episode-assembly` 管奖励张量的拼装与 bootstrap 规则；本题只管拼装完之后的**归一化记账**。
- `to-kl-controller` / `to-reward-model-bradley-terry` 分别是 KL 系数自适应与奖励建模目标，同样属于公式侧。
- `batchnorm-forward`（order 16）/ `groupnorm`（order 19）/ `layernorm-forward` 归一化的**空间轴**是特征维或通道组，且训练态要现算批统计；本题的轴是**到达时间**（批与批之间），核心难点是 EMA 账本与"用旧统计归一化新数据"的顺序，这正是 RLHF 训练器里 reward 管道和推理期 BatchNorm 的本质差别。
- `rl-per-priority-replay` 处理奖励派生的优先级在采样概率与 IS 权重上的作用；本题只到"奖励数值归一化成单位尺度"为止。

## 复杂度
$O(T\cdot B)$：每批一遍求均值、一遍求方差、一遍写输出，组统计用 `index_add_` 聚合（$O(B)$）后按组号广播。内部用 float64 累加、输出 float32。oracle 是纯 Python 双重循环 + `math.fsum` + 按组号建字典，结构与参考解完全不同以便交叉验证。
