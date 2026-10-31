# notes — int-activation-steering（order 660）

## 回源

- 官方实现：`nrimsky/steering-vectors`（SteeringVector 类，`apply` 把 `vector * coefficient` 加到指定层的残差流输出上；支持 `normalize=True` 先归一化）。上游作用于真实模型 forward，本题按判题红线把隐态全显式入参，只收注入与方向账。
- 论文语境：representation engineering / activation steering（ITI、ActAdd 同族）；与 653 `int-tuned-lens-affine`（探针读出）划界：本题是**写入侧**。

## 判题面

- solve 入参：hiddens `[L,T,D]`、s `[D]`、alphas（K 个实数）、layer_indices（K 个 int，可重复）、normalize bool。返回 steered/s_norm/layer_delta/cos_before/cos_after/proj_coef。
- oracle：`test_cases._py` 独立纯 Python（math.sqrt），对拍全部字段，容差 1e-9。
- 锚点：① h=[3,0]、s=[0,4]、α=0.5 正交注入（cos_before=0、cos_after=2/√13、layer_delta=2）；② normalize 前后对比（|s|=4 时注入长度差 4 倍）；③ 同层两次注入 α=1,2 ≡ 单次 α=3（累加语义）；④ 跨层注入只动目标行、cos_before 全部基于原始值；⑤ α=−proj_coef 使 cos_after=0（正交化语义自洽）；⑥ float64 输入不被原位修改（`.double()` 对 float64 是恒等映射，原位污染可观察）。
- 非法输入：维度/形状、零 s、K=0、长度不齐、层号越界/浮点、alpha nan/inf/bool → ValueError。

## 变异记录（9 条全抓）

no-inject（α 归零）、idx-rotate（层号循环错位）、overwrite（同层覆盖不累加）、normalize-drop、normalize-sq（除 |s|²）、cos-no-norm（余弦不除范数）、cos-after-num（cos_after 分子用原始 h）、proj-denom（分母 |s| 而非 |s|²）、in-place（`clone()` 改引用——float32 输入下 `.double()` 恒拷贝故等价，靠 float64 输入用例抓）。

## 边界与陷阱

- `.double()` 只在输入为 float32 时保证拷贝；「不得修改入参」的契约要靠 float64 用例钉住。
- cos/proj 全部 float64 语义；layer_delta 是「累计注入向量的范数」，同向注入下 = |Σα|·|s_eff|，带符号 α 先求和再取绝对值。
- 红线：真实模型 forward、生成行为对比、ITI 注意力头侧机制不出。
