# notes — la-rwkv-channel-decay（order 658）

## 回源

- 官方实现：`BlinkDL/RWKV-LM` 仓库 `RWKV-v5/src/model.py` 的 `RWKV_Tmix_x060`（`--my_testing x060` 即 v6.0），核心算子 `WKV_6`（`cuda/wkv6_cuda.cu`）。`BlinkDL/ChatRWKV` 的 `RWKV_in_150_lines.py` 是 v4 口径（`time_decay` 为常量、加在 pp 上）——**勿混淆**；`BlinkDL/RWKV-LM/RWKV-v6/` 目录只剩 README，v6 训练代码实际挂在 RWKV-v5 目录以 x060 开关启用。
- 上游模型侧：`w = time_decay + tanh(xw @ time_decay_w1) @ time_decay_w2`（data-dependent，负值），`u = time_faaaa`（逐通道常量参数）；头内 GroupNorm、token-shift lerp、门控 silu 都在 `RWKV_Tmix_x060` 里，本题**不收**——只收 WKV 核的递推账。
- 官方 CUDA kernel 逐通道循环即上文两条分支；kernel 内部 float32、单序列（无 batch 维）；初始 `aa=bb=0`、`pp` 用大负数代理 -inf。本题把 -inf 代理常量钉死为 `-1e38`（float32 可表示）。

## 判题面

- solve 入参：r/k/v/w `[T,C]`、u `[C]`、state dict（a/b/p 各 `[C]`，可 None）。返回 `wkv`/`y`/`final_state`/`p_trace`（嵌套 list，逐 token 的指数上界，让衰减算术可观察）。
- oracle：`test_cases._py` 独立纯 Python 实现（math.exp 逐通道），对拍 y/wkv/终态/p_trace，容差 1e-5。
- 锚点用例三处手算：① 零初始态 t=0 的 `wkv=v`、`p_trace[0]=k0`（-1e38 的数值吸收）；② 初始 `(a,b,p)=(1,1,-2)`、w=-1、k=-3：输出分支 `p_out=max(-2,0)=0`、`wkv=(e^-2+2)/(e^-2+1)`，更新分支 `ww=-3=p`、终态 a=3、b=2，t1 再验 `e^-1` 衰减；③ bonus 分支 `u=0.7 > pp=0` 的解析式。
- 因果性用例：整段跑 vs 分两段跑（第二段喂终态）结果一致（同序逐位一致）。
- 通道独立性用例：只扰动通道 1，通道 0/2 输出逐位相等。
- 非法输入：四张量形状不匹配、u 通道不齐、空 T/C、state 缺键/长度不齐 → ValueError。

## 变异记录（10 条）

9 条有效变异全抓：bonus-to-zero（u 归零）、decay-on-new-p（`ww = p_new + w`，v6 最经典复现错误）、no-decay（`ww = p`）、no-exp-numer / no-exp-denom（更新分支丢 e1）、out-no-e2 / out-no-denom-e2（输出分支丢 e2）、no-gate（y 不乘 r）、state-reset（终态清零）、p-init-zero（初始 p=0，靠零状态用例 t≥1 抓）。

**等价变异 1 条（已验证，非测试缺口）**：`bonus-to-k`——输出分支的 `p_out = max(pp, u)` 改成 `max(pp, k)`。原因：`wkv = (e1·aa + e2·v)/(e1·bb + e2)` 的 e1/e2 共享同一 p_out，对分子分母同乘 `exp(p_out)` 后 `wkv = (exp(pp)·aa + exp(u)·v)/(exp(pp)·bb + exp(u))`，**与 p_out 取值无关**——max 平移纯属数值稳定化，其比较对象是行为自由的。但 `e2 = exp(u - p_out)` 中的 u 本身必须保留（bonus-to-zero 即可抓到丢 u）。

## 边界与陷阱

- `pp + w` 在 pp=-1e38 时数值吸收（float32/float64 均如此），零状态下衰减第一帧不可见——测试锚点必须给有限初始 p 才能观察 w。
- y 在循环里已按 `rt * wkv_t` 算过，返回处再 stack 一次等价；两处必须一致。
- 红线：模型侧 token-shift lerp / LoRA 生成 w / GroupNorm / 头结构不出；batch 维不出（kernel 单序列口径）。
