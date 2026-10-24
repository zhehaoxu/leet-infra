# SwiGLU MLP 详解

## 从 FFN 到 SwiGLU

经典 FFN 是 $\text{ReLU}(xW_1+b_1)W_2$——一个升维、非线性、再降维。Shazeer (2020) 指出：在**相同参数量/算力**下，
给激活加一条**乘性门控**（GLU 家族）明显更优。LLaMA/Qwen 采用的 **SwiGLU** 用 SiLU(swish) 作门：

$$ \text{FFN}(x) = \text{SiLU}(xW_{gate}) \odot (xW_{up})\,W_{down} $$

三条矩阵 $W_{gate},W_{up}$ 把 hidden 升到 inter，$W_{down}$ 降回。**都没有 bias**。

## 为什么是三、而不是两个矩阵

GLU 需要「一条算门、一条算值」两路，再加输出投影 = 三个矩阵。为保持与旧 FFN 相当的参数量，
inter 维度通常取 $\approx \tfrac{2}{3}\cdot 4\,hidden = \tfrac{8}{3}hidden$（LLaMA 用 $2\cdot\text{hidden}/3$ 向上取整到 64 倍数）。

## 实现坑

- **是逐元素乘、不是相加**：`silu(gate)*up`。门控若写成加，容量与语义全变。测试里 `up=0 → 输出 0` 正是抓这个。
- **权重布局 [out, in]**：`F.linear(x, W)=x@Wᵀ`。和 HF `nn.Linear.weight` 一致，别转置反。
- **SiLU = z·sigmoid(z)**：与 `swish`/`F.silu` 同，注意不是 `x·sigmoid(βx)` 的带参版本。

## 关联

- 上游：`llm-rmsnorm`、`llm-rope-apply`。
- 下游：`llm-decoder-layer` 后半段 `post_attention_layernorm → mlp → 残差` 用这里。
- DeepSeek-V3 的 MoE 把「一个 MLP」换成「N 个专家 MLP + 路由」，每个专家内部仍是 SwiGLU（`llm-moe-experts`，待做）。
- 对拍：`transformers` `Qwen2MLP`（同权重、`hidden_act="silu"`），`allclose` $10^{-5}$。
