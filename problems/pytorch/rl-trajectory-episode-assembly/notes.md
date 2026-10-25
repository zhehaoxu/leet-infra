# 详解：多段 rollout 轨迹拼装与截断 bootstrap

## 出处

- **Gymnasium / gym≥0.26 的 Env API**：`step()` 返回 `(obs, reward, terminated, truncated, info)`，把「episode 结束」拆成两件互不相干的事。`terminated=True` = 真的走到了吸收态（goal / 失败 / EOS / 工具判定结束）；`truncated=True` = 环境本身还没结束，是 `TimeLimit`（`max_episode_steps`）或 LLM 侧的 `max_new_tokens`/`max_turns` 把这条 episode 从中间切了一刀。`gymnasium.wrappers.TimeLimit` 在超步数时置 `info["TimeLimit.truncated"] = True` 且 `terminated=False`。本题的 `terminated`/`truncated` 两个 0/1 向量就是这两个布尔量的逐步记录，**二者互斥**（题面保证：每段恰在末步带且只带一个 flag，即该步 XOR=1）。
- **吸收态价值为 0**：episodic return 分解的定义性约定（Sutton & Barto, *Reinforcement Learning: An Introduction*, 第 3 章 episodic tasks 一节）——终止之后不再有未来奖励，故 \(v(\text{terminal})\equiv 0\)。这就是「真终止绝不 bootstrap」的来源。
- **截断必须 bootstrap**：各家 PPO 实现的共识口径（漏掉就是系统性偏差）。把 timeout 当终止会破坏马氏性/引入偏差的正式讨论，见 Pardo 等（2018）关于把 timeout 增广为部分可观测的工作（**arXiv 编号待回源，不凭记忆写死**）。
- 本题**只交付** \((r, V, next\_value, done)\) 序列；\(\delta_t\) 与 \(\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}\) 的递推归 `to-gae`（Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438）。

## 完整推导与下标约定

先把三套下标钉死，本题一半的错都出在「下标空间串了」：

- \(t\in[0,T)\)：**出现序（arrival order）**数组下标，即 rollout buffer 里的物理位置；
- \(e\in[0,K)\)：**段号**，\(K\) = 极大连续段数；
- \(i\in[0,T)\)：**训练序**下标，即输出数组的物理位置。

1. **分段**：\(\text{run}_t=\sum_{s\le t}\big[s=0 \lor ep_s\ne ep_{s-1}\big]-1\in\{0,\dots,K-1\}\)（`cumsum` 打段号）。同一条 episode 的步在输入里**恰构成一段极大连续段**且 id 全局唯一不复现，故「段」等价于「episode」，`run` 就是 final_value 的索引空间。
2. **段末 / done**：\(\text{last}_t=(t=T-1)\lor(ep_{t+1}\ne ep_t)\)，\(done_t=\text{last}_t\)。截断段的末步同样 \(done=1\)。
3. **重排**：段按 \(ep\) 的 id **升序**得到训练段序，第 \(e\) 个训练槽对应出现段 \(\text{slot}^{-1}(e)\)；段内保持原时间顺序。于是 `lengths`（训练序每段步数）与 `starts`（exclusive 前缀和）确定，一个 \(T\) 长的 `gather` 下标就能把 5 个 \((T,)\) 数组同时搬到训练序——**注意 \(K\) 与「第几个训练槽」不是一回事**：出现序 `[7,7,7,3,3,8,8,8]` 里 id=3 是出现段 1，却是训练段 0。
4. **next\_value 三分规则**（实现上先在出现序算、最后统一 gather）：
   \[
   nv_t=\begin{cases} V_{t+1}, & \neg\,\text{last}_t\\[2pt] final\_value[\text{run}_t], & \text{last}_t\land truncated_t=1\\[2pt] 0, & \text{last}_t\land terminated_t=1\end{cases}
   \]
   向量化写法是「先整条 `roll(-1)`（`nv[:-1]=V[1:]`），再用 `nv[is_last] = where(truncated, final_value[run], 0)` **覆盖**所有段末位」。roll 在段末位带来的是**邻段首步的 value**（泄漏值），但它必被覆盖掉；数组最末一位 roll 越界落成 0，也必被覆盖（\(\text{last}_{T-1}\) 恒真）。这个「先污染后覆盖」的顺序不要反过来写。
5. **下游怎么用**（属于 `to-gae`，此处只给接缝）：拼装产出的每段 \((r_i, V_i, nv_i)\) 让 \(\delta_i=r_i+\gamma\,nv_i-V_i\) 对**所有** \(i\) 用同一条公式成立，无需判越界、无需区分段内/段末；优势侧 then 用 done-mask 递推 \(\hat A_i=\delta_i+\gamma\lambda(1-done_i)\hat A_{i+1}\)，它与「按 \([\text{starts}_e,\text{starts}_e+\text{lengths}_e)\) 逐段循环、段末把累加器归零」严格等价。

**实测数据（均由参考解 `solution.py` 跑出）**：

- 题面示例（ids 7/3/8，段 0 真终止、段 1、2 截断）训练序输出：
  `values=[0.25,0.375,0.5,0.625,0.75,0.125,0.25,0.3125]`、`dones=[0,1,0,0,1,0,0,1]`、
  `next_values=[0.375,0.875,0.625,0.75,0.0,0.25,0.3125,0.625]`、`lengths=[2,3,3]`、`starts=[0,2,5]`；
  \(\gamma=1\) 时 \(\delta=[0.625,2.5,1.125,0.125,-0.25,1.125,1.0625,0.3125]\)。
- 出现序天真 roll(-1) 得到 `[0.625,0.75,0.25,0.375,0.125,0.25,0.3125,0.0]`：位置 2（id=7 末步）拿到 **0.25**（id=3 段首步）、位置 4（id=3 末步）拿到 **0.125**（id=8 段首步）——两个都是邻段泄漏，正确值分别是 0.0 与 0.875。
- **多轮 LLM-RL 变体**（episode = 一次会话，步 = 一个 assistant 轮次，段末 flag = 该会话最后一次生成的收尾方式）：出现序 `episode_id=[12,12,12,5,5,5,5,9]`（id=12 三轮 EOS 真终止；id=5 四轮被 `max_new_tokens` 截断；id=9 是**单轮会话**且被截断），`terminated=[0,0,1,0,0,0,0,0]`、`truncated=[0,0,0,0,0,0,1,1]`、`values=[0.125,0.25,0.375,0.5,0.625,0.75,0.875,0.0625]`、`rewards=[0.0,0.0,1.0,0.5,0.5,0.5,0.5,2.0]`、`final_value=[0.5,0.9375,0.1875]`（按出现段）→ 训练序（id 升序 5、9、12）
  `actions=[11,12,13,14,7,101,202,303]`、`rewards=[0.5,0.5,0.5,0.5,2.0,0.0,0.0,1.0]`、`values=[0.5,0.625,0.75,0.875,0.0625,0.125,0.25,0.375]`、
  `dones=[0,0,0,1,1,0,0,1]`、`next_values=[0.625,0.75,0.875,0.9375,0.1875,0.25,0.375,0.0]`、`lengths=[4,1,3]`、`starts=[0,4,5]`。
  位置 4 就是那条**单轮被截断**的会话：首末同一步也必须 bootstrap `0.1875`，绝不能因为「这条只有 1 步」而跳过。轮次边界（谁是谁的后继）与 token 边界无关——拼装只在轮次粒度上做，token 级 mask 是 SFT 侧那几题的事。

## 核心与易错点

1. **flag 的归属步**：三分规则读的是**段末那一步**的 flag，不是「这一步自己把 episode 结束了」。题面保证 flag 只出现在段末且 XOR=1；一旦数据把 truncated 标在段中间（或段末不带任何 flag），「正确输出」在契约下就不存在了。用例里的 ⓪ 号输入自检专门卡这条——两个 flag 向量曾经把截断标错位置，判题就变成了「逼参考解违反契约」。
2. **两种混法各自单调**（数值实测，\(V_{end}\) 就是被漏/多算的量）：
   - 截断当终止（**漏 bootstrap**）：末步 \(\delta\) 少 \(\gamma V_{end}\)。上例 id=5 段末步正确 \(\delta=0.5+0.1875-0.875=-0.1875\)；置 0 变 \(-0.375\)（少 0.1875）；天真 roll 变 \(-0.3125\)（少的是邻段首步 value 0.0625）。critic 侧表现为「接近生成上限」的状态价值被系统性压低，策略侧截断样本整条被打上负优势。
   - 真终止补 bootstrap（**多算**）：末步 \(\delta\) 多 \(\gamma V_{end}\)。题面 id=7 段（真终止）末步正确 \(\delta=0.5+0.0-0.75=-0.25\)，若误用 `final_value[0]=0.9375` 就变 \(+0.6875\)。吸收态凭空拿到一份未来价值，「失败/被切断」与「值钱的续写」不再可分——这正是「只要拖到上限就算好结局」那一类 reward hacking 的入口。
   两个错法方向相反，所以「结果看起来还行」不能当自检；只有按 flag 分支才可能对。
3. **done ≠ 真结束**：done 是**结构位**，语义是「禁止把拼接数组里的下条记录当后继」，它同时封掉两件事：非段末位的 roll 不许跨段、段末位的优势递推不许跨段。截断位 \(done=1\) 且 \(nv=V_{end}\)，真终止位 \(done=1\) 且 \(nv=0\)。把 done 读成 terminated，就会在所有 done 位统一置 0，恰好退化成错法 2 的第一条。
4. **不许用 \(nv=0\) 反推「真终止」**：截断段的 \(V(s_{end})\) 本身就可能是 \(0.0\)（终局观测零价值、或 reward 归一化后为 0）。实测 `episode_id=[4,4,2]`、`truncated=[0,1,1]`、`final_value=[0.0,0.75]` → 训练序 `next_values=[0.75,0.5,0.0]`：位置 2 明明是截断 bootstrap，数值却是 0。终止/截断只能回到两个 flag（或 done 加 flag 的二维判定），任何「靠数值猜边界」的实现都会在含 0 值的 batch 上静默错。
5. **done-mask 不能省**：把整条拼接数组直接反向递推、不 reset，题面示例实测 \(\hat A\) 为 `[5.788627,5.435397,3.089891,2.068307,2.045586,2.416406,1.359375,0.3125]`，位置 4（id=7 段的真终止末步）本该是 \(-0.25\)，结果被下一条 episode（id=8）的优势灌成 \(2.045586\)。正确做法要么逐段循环（用 `starts/lengths`），要么乘 \((1-done_i)\)；这两种严格等价，实测 done-masked 得 `[3.0,2.5,1.018125,-0.1125,-0.25,2.416406,1.359375,0.3125]`。
6. **final_value 的索引空间**：按**出现段序号**（`run_idx`）取，不是排序后的训练槽、更不是 id 的字典序 rank；真终止段的该项存在但不被使用（本题示例里 `final_value[0]=0.9375` 全程不出现）。把 `boot = final_value[run_idx]` 写成 `final_value[slot]` 是纯粹的置换 off-by-permutation，且只在乱序 batch 上错——升序 batch 恰好全对，是最容易漏测的一类 bug（`check_sorted_ids_identity` 就是为此存在的）。
7. **退化与边界约定**：极大连续段的定义保证**段长 \(\ge 1\)**，故 `lengths` 不含 0 项、\(K=\) distinct id 数、`starts[K-1]+lengths[K-1]=T\)；「空 episode」在契约里不存在——轮次数为 0 的会话必须在进入 buffer 前就被丢掉，否则它会占掉一个 `final_value` 槽却没有归属步。\(T=0\)（此时 \(K=0\)、`final_value` 亦为空）走空约定：返回七个 0 元素张量，dtype 依次 `int64/float32/float32/int64/float32/int64/int64`，不抛异常。id 已升序时 `gather` 退化为恒等，规则一字不改。契约外若 id 复现（如 `[3,5,3]`）不承诺判题，参考解的真实行为是「按出现段切 3 段、`lengths=[1,1,1]`」——即把同 id 的两条段当成两条 episode。

## 与邻近题的区别

- **`to-gae`（接缝必须钉死）**：`to-gae` 从**已经拼好的**单条 \((r_0..r_{T-1}, V_0..V_{T-1})\) 序列推 \(\hat A\) 与 \(R\)，其 \(\delta_t=r_t+\gamma V_{t+1}-V_t\) 里 \(V_{t+1}\) 是**数组内右移**得来的，越界处写死 \(V_T=0\)——这等价于「这条数组的末步是真终止、且只有一条 episode」的特例。本题**不做**任何 \(\delta\)、不求和、不递推，只交付 \(r, V, next\_value, done, lengths, starts\)。接缝恰在**末步的 \(V_{t+1}\) 从哪来**：`to-gae` 假设它是 0，本题按 flag 决定它到底是 0 还是 \(V(s_{end})\)，并把「一条数组其实是 K 条 episode」这件事用 done/offsets 显式化。实测（题面示例，\(\gamma=1,\lambda=0.95\)）：把训练序逐段喂 `to-gae`，真终止段 id=7 得 `advantages=[1.018125,-0.1125,-0.25]`，与 done-masked 逐位一致（此时 \(nv_{end}=0\) 与它的 \(V_T=0\) 假设吻合）；截断段 id=3 得 `[2.168750,1.625]`，而 done-masked（用 \(nv_{end}=0.875\)）得 `[3.0,2.5]`，差 \(=V_{end}=0.875\) 与 \(\gamma\lambda V_{end}=0.831250\)——这段差值**完全由本题的拼装层负责**，`to-gae` 看不到任何能推出它的信息（它连 flag 都没有入参）。另一侧：`to-gae` 是单段 \((T,)\to\) 二元组，无 id、无 flag、无乱序、无 gather。
- **`data-collator` / `sequence-pack-pad`**：SFT 侧的批打包——pad/collate、产出 token 级 `input_ids/attention_mask/position_ids/labels` 与 cu_seqlens 式边界。形似（都是「把变长记录拼成训练批次」），神不似：那里**没有 \(V\)、没有 done、更没有 bootstrap**，边界只影响 mask 而不影响任何数值分支。本题不生成任何 token 级张量，`actions` 只是原样透传的 int64 列；边界影响的是**每个位置的后续价值**。
- **`rl-rollout-shard-repack`**：跨 rank 的重分片与 padding-free 装箱，考负载均衡、搬运条数与 padding 代价。它搬的是**已经拼好的样本**，本题重建的是**样本本身**（段边界与价值对齐）。两者都在「多段混合」的数据流上工作，但本题全程单进程、与 rank 无关。
- **`rl-reward-batch-norm`**：批统计侧（running mean/std、组内 vs 全局归一化），发生在 reward 数值分布上，对段边界与 flag 语义不敏感；本题不产生任何统计量、不改写 reward 数值。
- **`rl-async-rollout-staleness`**：生成队列/陈旧度（partial rollout、off-policyness 的时间账）；本题假设 rollout 已全部落地为一组显式张量，无任何时间/步序语义，乱序只发生在 **buffer 里段的排列**这一层。

## 对拍口径

参考解走「`cumsum` 打段号 → `bincount` 数段长 → `argsort` 定段序 → 一次 `gather` 收拢 5 个 \((T,)\) 数组，next_value 先整条 roll 再在段末位 `where` 覆盖」的向量化路径；判题 `_oracle` 走「显式逐段扫描成 `(id, 段号, 步下标)` → python `sorted()` 排段 → 段内朴素 for 循环三分规则 → 纯 list 拼接成张量」，两条路径不共享任何中间量。除题面示例（`check_example`）外，还有 4 条规则探针（跨段泄漏、单步段、全截断、全真终止）、\(T=0\) 空约定、升序 id 恒等 gather；再加 11 条无随机电池（乱序/升序/单步段/T=1/全截断/全真终止/多段混合）逐位比对七元组，以及一条**不依赖 oracle** 的结构性不变量探针：输入 flag 合法性、\(\sum lengths=T\)、`starts` 前缀和账、`sum(dones)=K`、训练序第 \(e\) 段 values 恰等于第 \(e\) 小 id 段的出现序切片（注意这里必须用出现序 values 配出现序下标、训练序 values 配训练序下标，混用两个下标空间是这条探针原先的错）、done 位 bootstrap 计数与 flag 计数相等、非 done 位 \(nv_i=V_{i+1}\)。全程无随机，所有 episode 数据都是显式输入。

## 复杂度

\(O(T + K\log K)\)（段 id 的 `argsort`），无 Python 循环、无分支同步；\(T\le 4096\) 下 gather 一次完成，显存 \(O(T)\)。
