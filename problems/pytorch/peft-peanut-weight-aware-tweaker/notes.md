# Notes

- 来源：PEANuT (arXiv:2410.01870)，已合入 HF PEFT `src/peft/tuners/peanut/`（`PeanutConfig`：r=32、depth=0、act_fn='relu'、scaling=1.0、init_weights=True）。
- 判题域：`_compute_delta_weight` 的纯函数核心 + 参数量账 + 初始化 RNG 回放 + 残差配对断言；官方 layer 类的 device/dtype 搬运与 merge 缓存不在域内。
- 模式（6）：delta_weight / forward / param_count / init_factors / zero_init_check / residual_order。
- 锚值 float64、容差 1e-9；kaiming_uniform_(a=√5) 等价 `uniform(±1/√fan_in)`，直接用 `uniform_(generator=)` 回放规避 `nn.init` 无 generator 的版本差异。
- 残差反序用 identity 激活 + 对角 `i+1` 的 encoder/decoder 权重做端到端区分：顺序不同结果必不同。
- mut 17：残差反序/去残差/scaling/transpose/B 转置/三处激活/enc-dec 跳首/lora 账/depth 账/RNG 顺序/bound/零初始化/forward 方向/零检查翻转，全部 CAUGHT。
