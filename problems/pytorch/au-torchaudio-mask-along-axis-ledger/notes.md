# 865 出题笔记：torchaudio SpecAugment 掩码账

## 1. 立项与查重

order 865，合集 `audio` 16→17，slug `au-torchaudio-mask-along-axis-ledger`。取 `functional.py` 的 `_get_mask_param`(806-810) / `mask_along_axis_iid`(813-882) / `mask_along_axis`(885-958)。**重要回源纠错**：此前把 `mask_along_axis` 记作已被 693 覆盖，实际 693 是 FunASR/espnet 的显式段列表账，与 torchaudio 这三个函数完全无关；todos.md 已改正该口径。全库 meta / description / notes 查重确认与 855、857-864 的函数集合不重叠。

## 2. 官方源码与逐字环境

`official865.py` 用 `ast.get_source_segment` 抽取三个顶层定义并 `exec` 进共享 NS，官方函数体一字未改；`torch` 是转发代理，只把 `rand` 换成返回显式入参张量的记录器，`Tensor` 仍是真 `torch.Tensor` 以保留 `isinstance(mask_value, Tensor)` 分支。`_interval` / `_mask_rows` / `_apply_ledger` 从记录到的 rand 调用重新推导掩码，`_check_ledger` 断言它与逐字官方输出一致，因此区间账不是另写的第二套算法，而是对官方输出的独立复核。

## 3. 机制账

`p==1.0` 直通 / `min(mask_param, int(axis_length*p))` 截断；维度 guard（shared ≥2、iid ≥3）与逐字错误消息；轴集合 `[dim-2, dim-1]`；`p ∈ [0,1]` 闭区间；`mask_param<1` 返回原对象；shared 的 `reshape([-1]+shape[-2:])` 打包与解包、`torch.rand(1)` 不带 dtype 因而 float32、`min_value.long()+value.long()` 分离取整、freq 轴 `unsqueeze(-1)`、死代码宽度 guard；iid 的前导维 rand 形状、两次 transpose、`where` + `mask_value.repeat` Tensor 分支、无宽度 guard。纯 Python 侧用 `struct` 逐标量步量化 float32、用 `math.trunc` 复刻 `.long()`。

## 4. 三层验证与变异

12 段探针 `380/380 ALL GREEN`。三层对拍 `22054 checks / 74 configs / 0 fails`：官方走真实逐字环境，参考解走纯 Python 递归 + `struct` 量化 + `math.trunc`，独立 oracle 走扁平索引算术 + 集合成员判定 + `Decimal ROUND_DOWN` + `ctypes.c_float`，三条路径无共享算法。判题 driver `140/140 ALL ACCEPTED`。

首轮 169 项 / 154 CAUGHT / 14 SURVIVED / 1 ANCHOR-ERROR。锚点错误来自 shared guard 消息在源码里被拆成两行，修正为单行锚点。补 13 条硬化用例后只剩 3 项，逐个证明为真等价并写入 `mut865.py::EQUIVALENT`：`h-pack-float`（`_pack` 的 float 转换被 `_nest` 两个 base case 的二次转换覆盖，且叶子校验发生在转换之前）、`i-shared-first`（shared 的 rand_shape 恒为 `[1]`，draw-size guard 保证 vdraw/sdraw 恰 1 个元素，`[0] is [-1]`）、`e-cases-dimshared`（`dim_shared` 用例是 1-D 谱，dim guard 的消息只报维度数，3 或 4 个元素结果相同）。`equiv865.py` 对每项在 1901 配置 × 10 mode = 19010 次比较上零差异，含异常类型与消息。最终 **169 项剔除 3 个文档化真等价 / 166 全 CAUGHT / 0 SURVIVED / 0 ERROR**。

## 5. 剥除口径与工程教训

判题域不含真实音频 IO、mel/STFT 数值、模型前向；谱、rand 抽样、`mask_value` 全部显式入参，纯 Python、无 numpy/torch/scipy、无全局 RNG。最大的坑是**默认前导维太小**：默认谱 `(2,3,4)` 只有 2 个 iid 抽样，`0.31+0.17k` 与 `0.13+0.29k` 都不越过 1.0，导致 `% 1.0` 的取模变异存活；`SPEC4=(2,2,3,4)` 的 n=4 仍不够，必须补 `(2,3,3,4)` 的 n=6 前导维。第二个坑是 `_map2` 的切片步长在 `(2,2)` 上 `prod(shape[1:]) == prod(shape[:-1])`，只有不对称前导维 `(2,3)` 才暴露。第三个坑是 `mask_value=0.0` 的默认配置让「freq 分支写死 0.0」的变异存活，必须补 `mask_value=-1.0` 的频率轴用例；整数填充值还需断言输出叶子类型是 `float`，因为数值对拍不区分 int 与 float。

## 6. 官方 quirk 与查重结论

与 `description.md` 第 5 节一致：`int()` 向零截断而 docstring 写 floor；shared 的 `torch.rand(1)` 不带 dtype 使区间走 float32，iid 跟随谱 dtype；`end` 是分离取整之和；shared 的宽度 guard 在合法抽样下不可达、iid 完全没有；`mask_param<1` 返回原对象且不消耗 draw；Tensor `mask_value` 的 `repeat` 在转置之后。补充：`variant` 除字面量 `"iid"` 外全部落到 shared；负 axis 被轴集合拒绝；`p` 闭区间两端合法。全库查重确认 693 与本题无关，函数集合与 855、857-864 完全不重叠。

## 7. 注册与护栏

`register865.py` 最小侵入注册 `audio` 16→17、中英 description 各追加 865 口径账长段；复核 `843 registered = 843 disk / max_order=865 / 无 dangling / 无 unregistered`。`todos.md` 更新为 843 题 / audio 17 / 下一批从 866 起，并把 865 从「已立项」改为「已落地」。双语全库校验预期 `0 error / 15 warning`（15 条全为历史既有）；短护栏用独立 `--basetemp`，全量护栏后台运行后回填结果。按约定改完先不 git commit，等用户确认。
