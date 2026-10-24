# GQA 因果注意力详解

## 从 MHA 到 MQA/GQA

标准多头注意力（MHA）里每个 query head 都配一套独立的 key/value head，KV cache 随层数×头数膨胀，长上下文推理时内存与带宽都吃不住。

- **MQA**（Shazeer 2019）：所有 query head **共享一组** kv head（$H_k=1$）。省内存最多，但质量略降。
- **GQA**（Ainslie 2023）：折中——把 $H_q$ 个 query head 分成 $H_k$ 组，**每组内共享**一套 kv。LLaMA-2-70B、Qwen、Mistral、DeepSeek 都用它。

本题用权重形状天然表达分组：`q_weight` 行数是 $H_q d$，`k/v_weight` 行数是 $H_k d$，$H_k \mid H_q$。当 $H_k=1$ 就退化成 MQA（测试里专门有一例）。

## repeat_kv 的对应关系

HF 的 `repeat_kv` 把 $k,v$ 从 $H_k$ 头复制扩到 $H_q$ 头，靠的是 `x[:,:,None].expand(B,H_k,n_rep,S,d).reshape(B,H_k*n_rep,S,d)`——**相邻 $n_\text{rep}=H_q/H_k$ 个 query 头共享同一个 kv 头**。写成 `repeat_interleave` 才是对的；用 `repeat`（整体平铺）会错配头。测试用 `nh=4,nkv=2` 与 `nkv=1` 两种配置对拍 oracle 抓这一点。

## 因果掩码：-inf vs finfo.min

因果性 = 位置 $i$ 只能看 $j\le i$。做法是上三角置 $-\infty$ 再 softmax，未来 token 权重归零。

- 本解用 `torch.full(...,float("-inf")).triu(1)`；HF 用 `finfo.min`（一个极大有限负数）。
- softmax 里两者结果一致（$\exp(-\infty)=0$；$\exp(\text{finfo.min})\approx 0$），差异仅在浮点末位，落在 $10^{-5}$ 容差内。scratch 对拍 maxdiff 约 $3\times10^{-7}$，可接受。
- 首 token 只attend自己 → 输出等于它的 $v$（单位投影时等于输入本身），`check_first_row_only_self` 就是验证这条。

## 其余语义坑

- **q/k/v 带 bias、o_proj 不带**：Qwen2 如此（Llama 同款）。少加 bias 会对不上 oracle。
- **缩放是 $\sqrt{d}$**（head_dim），不是 hidden。`scaling = head_dim**-0.5`。
- **softmax 在 float32 算再转回**：与 HF `eager_attention_forward` 一致，避免半精度下溢。
- **RoPE 只作用于 q/k、不作用于 v**，且在 head 轴用 `cos.unsqueeze(1)` 广播——与 `llm-rope-apply` 完全同规则。

## 关联

- 上游：`llm-rope-apply`（这里的旋转就是它；本题 solve 内部自带 `_rotate_half` 以便独立评测）。
- 下游：`llm-decoder-layer`（把本注意力 + RMSNorm + SwiGLU MLP 串成 pre-norm 双残差的一层）。
- 对拍：`transformers` `Qwen2Attention`（**必须 `attn_implementation="eager"`**，默认的 sdpa 走融合核，数值路径与手写 eager 有微差）、同权重同 cos/sin 同因果掩码，`allclose` $10^{-5}$。
