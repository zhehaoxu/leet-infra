# 详解：Checkpoint 键重映射与 strict=False 账目

## 出处

`torch.nn.Module.load_state_dict`（PyTorch 官方文档）：`strict=False` 时返回 `_IncompatibleKeys(missing_keys, unexpected_keys)` 命名元组；源码里两个列表由 `set` 差集再 `sorted` 生成（不同版本对 `embed.weight` 这类根级 `nn.Parameter` 与模块属性的键名规则不同：`nn.Parameter` 直接挂在根模块上时 state_dict 键就是属性名本身，如 `embed`，而不是 `embed.weight`——所以本题把它包进子模块）。HF 的 `convert_gpt2_checkpoint.py` / `convert_llama_weights_to_hf.py` 等脚本正是「改前缀 + Conv1D `w`↔Linear `weight` 转置 + 增删层」这类重映射的工业化版本。

## 核心与易错点

1. **missing/unexpected 的口径**：missing = 新模型有而（映射后）checkpoint 没有 → `blocks.{0,1}.gate`；unexpected = checkpoint 有而新模型没有 → `lm_head.weight`。判断必须发生在重映射**之后**，否则 7 个改名键会全部同时出现在两份清单里。
2. **返回顺序**：判题环境把 `PYTHONHASHSEED=0` 固定，但 `sorted` 仍是唯一可靠的对齐方式；`_IncompatibleKeys` 的原始列表不承诺顺序，题面要求自行排序。
3. **值变换与键变换是两件事**：`embed.w → embed.weight` 除了改名还要转置；(3,2) 的 \([[25,26],[27,28],[29,30]]\) 转置成 (2,3) 后行主序展平是 25,27,29,26,28,30。只改名不转置，`load_state_dict` 会因 shape 不匹配直接 `RuntimeError`（strict=False 也救不了形状错误）。
4. **未加载的参数保持初始化值**：两个 `gate` 是新加的层，strict=False 加载后仍是零初始化——这就是「missing 不是错误而是增量初始化」的语义。
5. **shape 相同但语义不同的键不会被发现**：strict=False 只查名字与形状，不查语义；真正的转换脚本必须自己保证映射正确（本题用数值账核对）。

## 对拍口径

solution 走真 `load_state_dict`；判题 `_oracle` 完全不碰 nn.Module：按题面数列公式手推 8 段 checkpoint 值，用硬编码 old→new 对照表 + 集合差手写 missing/unexpected，embed 的转置按下标公式 \(M'[i,j]=M[j,i]\) 手工展开成 6 个数，`loaded` 逐参数与手工拼装表精确相等（小整数 float32 无损，`torch.equal` 级）。另设三条退化断言：gate 保持零、embed 非原样拷贝、两清单已排序。

## 复杂度

全流程 \(O(\text{参数总量})\)（约 40 个元素），瞬时完成；难点只在 API 语义正确性。
