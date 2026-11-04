# 解题笔记：FasterCache CFG-Cache 频域逼近步

## 回源定位

- 源文件：diffusers `hooks/faster_cache.py`（654 行，FasterCache 论文 arXiv 2410.19355）。与 768 同文件分立两题：768 取注意力块钩子，本题取 `FasterCacheDenoiserHook` L261-361 + `_split_low_high_freq` L638-654 + `apply_faster_cache` 权重缺省 L524-562。
- 官方门控：`should_skip_uncond = iteration > 0 and iteration % N != 0 and within`——`>0` 被 0%N==0 完全子sumed（死代码，与 768 同款），判题域直写三条件与。另官方 identifiers 字符串前置检查不在本题域。
- tensor_format 分支（BCFHW/BFCHW/BCHW/tuple 输出）判题域固定 BCFHW；BFCHW 只是 permute 换序，机制同构。
- 官方权重是 callback 返回值；`apply_faster_cache` 的缺省规则是「alpha_low/high_frequency 在各自 `*_weight_update_timestep_range` 内生效」，判题域按此内联为 `_freq_weight`。

## 数学要点

1. **CFG-Cache 的频域假设**：uncond 与 cond 输出之差 `Δ = uncond − cond` 在相邻步间低频主导、缓变。整批步把 Δ 拆成低频/高频两组复数谱存入 state；skip 步用 `uncond_freq ≈ (Δ_low·w_l + low_cond) + (Δ_high·w_h + high_cond)` 重建——低频差近恒定（缓变），高频差被 alpha 权重衰减/放大。
2. **delta 缩放是状态累积**：官方 `delta <- delta * weight` 写回 state，连续 k 次 skip 在区间内 → delta 乘 alpha^k；这是 Equation 11 的实现而非单步缩放。整批步 `delta = low_uncond − low_cond` 重算覆盖。测试用 ×1.1^4 / ×1.1³ 的连乘断言抓「单步缩放/不缩放」变异。
3. **圆形低频掩码**：fftshift 把零频移到中心 (W//2, H//2)（偶数尺寸）；半径 `min(H,W)//5`——取 min 因 fft2 只作用在最后两维，H/W 是空间分辨率。掩码含边界 `<= r²`：r=1 时 (cx±1, cy)、(cx, cy±1) 四点在圈内，`<` 变异可观测。
4. **往返一致性**：ifftshift(ifftshift 域) = 原谱；`real(ifft2(完整谱))` 是可逆的——alpha=0 时 delta 清零，uncond_freq == cond_freq，重建 uncond 与 cond 逐元素相等（浮点误差 1e-4 内）。这是免 oracle 的闭式断言。
5. **B*F 展平**：BCFHW → permute(0,2,1,3,4) → (B,F,C,H,W) → flatten(0,1) → (B*F,C,H,W)；fft2 只在 (H,W) 上做，帧维与 batch 维合并处理。还原时 unflatten(0,(B,-1)) 后 permute(0,2,1,3,4) 反向。numpy oracle 侧的陷阱：C=1 时 `reshape(b, *shape[1:])` 已得 (B,C,F,H,W) 排布，直接 transpose 会把 F/C 换反。

## 实现要点

- 校验顺序：张量 → 5 维 → current_timestep 整数（bool 排除）→ skip_range ≥1（bool 排除）→ 三个区间（二元数值组，bool 元素排除）→ alpha 有限数值 → distilled bool → state 四键（未知键先于缺键）→ skip 前置（batch_size 已记录 / 输入半批 / delta 已建立）→ 整批一致性（batch 与记录一致、偶数）。
- `_validate_range` 的 str/bytes 前置检查被 list/tuple 检查完全子sumed（任何 str 都不是 list/tuple），实现中删除（与 766 m-powerexp 同款冗余）。
- `iteration % N` 对 N=5：整批步只能出现在 iteration ≡ 0 (mod 5)，skip 序列中手工 state 的 iteration 要与目标分支对齐。
- 纯函数式：新 dict + delta `detach().clone()`；skip 步存的是缩放后的 delta（克隆自乘积），整批步存重算结果。

## 测试要点

- **oracle 三件套**（numpy float64 独立复算）：`_np_split`（fft2+fftshift+圆形掩码）、`_np_delta`（整批 delta）、`_np_uncond_approx`（skip 重建）——复杂张量按 real/imag 堆叠后 allclose，rel/atol 1e-3~1e-4（实现 float32 vs oracle float64）。
- 完整序列 `[F,T,T,T,T,F]`（N=5）+ iteration 终值断言。
- 权重区间：t=400（low 内 high 外）、t=200（双内，连乘 1.1^4/1.1³）、t=99 / t=301（开区间端点，×1.0）——端点用例是 m-w-open-lo/hi 的唯一猎手。
- alpha=0 闭式往返：delta 清零 → 重建 uncond == cond。
- skip 前置三查（batch_size 未记录 / delta 未建立 / 整批输入）、蒸馏序列（delta 恒 None）、purity、校验 battery（5 维、bool timestep、skip_range=0/True、区间 len/类型/bool 元素、alpha inf/True、state 四键结构）。
- 数值可观测性：`_lat` 生成的确定性纹波张量保证 delta 逐元素非平凡；区间/掩码变异产生的差异远超 1e-3 容差。

## 变异清单（50 条：49 CAUGHT + 1 等价剔除）

门控 5：开区间左右端 <=、取模反转/错位、丢蒸馏豁免。skip 前置 3：batch_size 检查删、半批复核反转、delta 检查删。权重 6：区间左右端 <=、反转、低/高缩放丢失、权重互换。频域拆分 5：半径 //4、中心偏移、掩码严格化、fftshift 丢弃、高低掩码同体。skip 重建 7：ifftshift 丢弃、丢 .real（等价剔除）、delta 符号反、丢高频项、cat 顺序反、state 存未缩放 delta。flat 转换 2：两处 permute 丢弃。整批分支 8：chunk 交换、delta 符号反、高频只用 uncond、batch_size 记录 +1、偶数校验反转、蒸馏也算 delta、整批输出截半、iteration 不增（skip/整批两处各 1）。校验 14：张量、维度、timestep、skip_range 下界、区间长度/类型/bool 元素、alpha 有限、iteration 负值、batch_size 偶数、delta 类型、state 缺键/未知键、消息。

等价剔除 1 条：`m-real-drop`（`ifft2(...).real` 去掉 `.real`）——末尾 `.to(model_output.dtype)` 把复数转实数时同样精确丢弃虚部（`torch.equal(c.to(float32), c.real)` 实证 True），`.real` 被子sumed。

经验：① **oracle 的 shape 排布陷阱**：numpy 侧 B*F 展平的还原必须「先 reshape(b, f_cnt, -1, …) 再 transpose(0,2,1,3,4)」，凭直觉先 transpose 会在 C=1 时把 F/C 换反；② **验证型守卫的子sumed 检查要删**（str/bytes 检查被 list/tuple 检查完全覆盖），否则对应变异永 SURVIVED；③ **区间端点变异需要端点用例**——默认参数的 t=400/200 都在区间内部，m-w-open-lo/hi 只有 t=99/301 才能分化；④ 复数张量的校验消息断言用键名子串（`state['…']` 引号边界易错）。
