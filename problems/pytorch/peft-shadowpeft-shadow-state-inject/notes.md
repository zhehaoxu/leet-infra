# Notes

- 来源：ShadowPEFT (arXiv:2604.19254)，官方 ShadowLLM/shadow-peft `src/shadow_peft/modules.py`（commit 65a43a7，已入 HF PEFT）。
- 判题域：ShadowInjectionModel 与 ShadowUpdateModel 的纯函数核心 + 参数量账 + 初始化 RNG 回放 + 门控代数恒等式；shadow backbone 与 KV cache 不在域内（dropout 按 eval 语义 p=0）。
- 模式（8）：inject / inject_zero_init / inject_sequence / update / gate / gated_identity / param_count / init_injection。
- 权重约定：gate/transform 权重按 nn.Linear (out, in)（F.linear）；injection 的 W_down (d, k)、W_up (k, d) 按 einsum 下标直用（与官方 Parameter 形状一致）——两套约定不同，锚值手算注意。
- 锚值 float64、容差 1e-9；downs 是单次 normal_(std=0.02) 整块抽取（(L, d, k) 一个张量），不是逐层抽。
- 与论文出入：T（transform）无 sigmoid、sigmoid 只在门 G；dropout 只在 transform 中段；LN 共享 eps=1e-5。
- mut 17：差值符号/alpha 方向/up 下标/down 转置/零初始化翻转/序列跳层/LN eps/门 relu/门去 SiLU/transform SiLU→ReLU/门控公式/门向翻转/恒等翻转/两处参数量账/init std（solve 层）/ups 非零，全部 CAUGHT。solve 层显式默认会遮蔽函数签名默认的变异（同 Arrow 教训）。
