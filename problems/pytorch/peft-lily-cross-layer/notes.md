# 详解：Lily 跨层互联专家组合与路由

## 出处

论文 Low-Rank Interconnected Adaptation across Layers（arXiv:2407.09946，ACL 2025 Findings，Yibo Zhong 等，四川大学）。理论出发点：LoRA 本质是**梯度压缩器**——固定投影矩阵 \(A_0\) 把梯度投影进低秩子空间，\(\Delta W\) 被钉死为低秩。Lily 用层级框架解耦：LP（下投影）留在层组局部，HP（上投影）剥离为模型级共享 expert 池（MoE 式多 expert），每个 LP 连全部 HP；层间互联 + router 选择性混合让有效秩突破单层低秩。官方仓按任务分子目录（commonsense-reasoning / natural-language-understanding / subject-driven-generation / visual-adaptation-benchmark），NLU 版 `lily.py` 是最干净的实现（基于 fourierft 实验脚手架改）。

## 官方实现的逐行翻译

```python
class lily_adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, ne, lp, hps, mlp=False):
        self.router = nn.Linear(hidden_dim, ne, bias=False)
        self.non_linear = nn.ReLU() if mlp else nn.Identity()
    def forward(self, x):
        hidden = self.non_linear(self.lp(x))
        router_logits = self.router(hidden)                    # [B, N, ne]
        router_probability = F.softmax(router_logits, dim=-1)
        expert_probabilities = router_probability.mean(dim=(0, 1))   # (ne,)
        combined_hp = torch.einsum("e,eio->io", expert_probabilities, self.hps)
        return torch.matmul(hidden, combined_hp)
```

三个关键口径：

1. **router 是 bias-free Linear + 全局 softmax 平均**：没有 top-k、没有负载均衡噪声、没有可学习 gate 之外的稀疏化——`expert_probabilities` 是 batch 与 token **双轴**的全局平均，整个 batch 共享一个 expert 混合比（数据依赖但逐 batch 确定性）。
2. **monoscale 变体**（`lily_adapter_monoscale`）：`combined_hp = sum(hps, 0) * (1/ne)`——均匀平均，连 router 都不要，是论文的 ablation 基线。
3. **HP 零初始化**：`set_lily` 里 `nn.Parameter(torch.zeros(ne, dim, 768))`——训练起点所有 expert 全零 ⇒ \(\Delta W=0\)，与 LoRA 的「B 零 A 随机」同哲学但更彻底。

**set_lily 的 LP 放置账**：

```python
stride = 12 // ne
if idx % stride == 0: new_lp = nn.Linear(768, dim, bias=False)
layer.lily_adapter = lily_adapter(..., previous_lp, hps)
```

层号从 0 计，每 stride 层换新 LP，层组内共享同一 LP 实例；n=12/ne=4 → stride=3 → 4 个 LP。n 不整除时（如 5/2 → stride=2）层 4 落入组 2——尾组收纳。

**attention 侧的 scale**：`value_layer = ... + self.lily_v(hidden_states) * self.s`——delta 乘全局 scale \(s\)（默认 1.0），本题不含（乘法在 adapter 外）。

## 易错点

1. **双轴平均**：`mean(dim=(0, 1))` 是 batch+token 全局平均；只平均 batch 或只平均 token 都是错位变异（m-mean-axis/m-mean-axis2 双钉）。这也是 Lily 与标准 MoE 的差异点——混合比是「本 batch 的层画像」而非逐 token 路由。
2. **softmax 在 expert 轴**：`dim=-1`（ne 维）；对 batch 轴归一化完全破坏语义（m-softmax-axis 钉）。
3. **softmax 平移不变**：logits 的数值稳定项（减 max）可替换为任意常数平移而不改变输出——变异 `logits+1.0` 是真等价（实测 SURVIVED 剔除记录），不要试图钉它。
4. **expert_mix 的 einsum 下标**："e,eio->io"：e 求和、io 自由；写反成 "->oi" 形状虽同（方阵 hps 时）但数值转置（m-einsum-trans 钉）。
5. **mlp 开关作用在 hidden 上**（LP 之后、router 之前），不是输出侧；ReLU 泄漏到 Identity 分支或丢掉都数值可钉（m-relu-leak/m-relu-drop）。
6. **monoscale 的 1/ne**：均匀平均不是求和（忘乘 1/ne 就是放大 ne 倍，m-mono-scale 钉）；也不是只取第一个 expert（m-mono-sum 钉）。
7. **LP 放置是 `idx % stride == 0`**：等价于组 id `i // stride`；写成 `i % ne` 破坏层组连续性（m-lp-mod 钉）。

## 对拍口径

锚点由沙箱 dump（numpy float64）：F1 用 in=8/hidden=4/out=6/ne=3，router 与 MLP 两分支、monoscale 两分支全锚。互证：`forward = Σ_e p̄_e·(hidden@hps_e)`（expert 维显式展开）。LP 放置账四组配置（含尾组收纳 n5/ne2 与 stride=1 的 n6/ne4）。测试另钉 softmax 行质（概率和 1、token 间分布非退化）与 mlp 分支可观测差异。

## 复杂度

前向 \(O(BN(in\cdot h + h\cdot n_e + h\cdot out))\)，与单 LoRA 同阶（expert 混合只加 \(O(n_e\cdot h\cdot out)\) 的组合项，且每 batch 一次）；LP 组共享把 LP 参数从 n_layers 份压到 \(\lceil n/st\rceil\) 份。
