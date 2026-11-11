# torchaudio `spectral_centroid` 谱质心账：`linspace(0, sample_rate//2, 1+n_fft//2)` 频率栅、`(-1,1)` 列向量广播、分母对原始谱按 `dim=-2` 求和、`0/0 → nan` 与精确 spectrogram kwargs

## 1. 背景

`torchaudio.functional.spectral_centroid` 用幅度谱与线性频率栅算逐帧谱质心。它**不含 RNG、不含音频 IO、不含 warning**，只依赖 `spectrogram` 取得幅度谱、一次 `linspace` 与两个 `sum(dim=-2)`。本题用纯 Python 复刻成八个可独立判题的 mode（`centroid` / `freqs` / `call_kwargs` / `ledger` / `zero_frame` / `shapes` / `errors` / `census`），把全部数值用官方常量快照钉死。
全库查重：`spectral_centroid` 在 `problems` 的全部 `meta.json` / `description*.md` / `notes*.md` 中仅 2 次命中，且都是 859 / 860 notes 里「零命中避让」的排除说明，**并非覆盖**；历史 notes 明确把它列为后续候选。855 抽取的是 `spectrogram` 本体加后处理四函数，本题**不重判** `spectrogram`：把它作为显式入参 spy 注入，只判它被调用时的精确 kwargs 与质心自身语义，边界清晰。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  spectral_centroid(waveform, sample_rate, pad, window, n_fft,
                    hop_length, win_length)   (lines 1257-1299)

  specgram = spectrogram(waveform, pad, window, n_fft, hop_length,
                         win_length, power=1.0, normalized=False)
  freqs = torch.linspace(0, sample_rate // 2, 1 + n_fft // 2,
                         device=specgram.device).reshape((-1, 1))
  freq_dim = -2
  return (freqs * specgram).sum(dim=freq_dim) / specgram.sum(dim=freq_dim)
```

`official868.py` 用 `ast.get_source_segment` 抽出顶层 `spectral_centroid` 并 `exec` 进同一共享 NS，官方函数体**一字未改**；注入对象只有真实的 `torch` / `math` / `typing` / `warnings` 与一个 `spectrogram` spy。spy 返回显式提供的幅度谱并记录精确 kwargs，手法同 855（stft spy）与 864（STFT / ISTFT / phase_vocoder 注入）。全局默认 dtype 固定为 `float64`，因此 torch 的 `linspace` / 广播 / 除法与纯 Python 参考解可逐位对拍。

## 2. 口径约定

### 八个 mode

`solve(mode, **kw)` 分发八路：`centroid` 返回逐帧质心，`freqs` 返回 `(F,1)` 列向量频率栅，`call_kwargs` 返回 spy 记录的 12 个 kwargs，`ledger` 返回含频率栅 / 加权矩阵 / 分子 / 分母 / 输出形状 / 调用参数的完整账本，`zero_frame` 判全零帧的 `nan`，`shapes` 判 2D / 3D / 4D 输入形状，`errors` 判六类边界，`census` 判源码级事实。除 `unknown mode` 抛 `ValueError` 外，所有异常收敛成 `{type, msg}` 字典返回，消息逐字对齐官方。

### 输入域

本题约定 `n_fft >= 2` 且 `1 + n_fft // 2` 等于 `specgram` 的频率维长度 `F`。**`n_fft <= 1` 明确排除在输入域外**：此时 `torch.linspace(..., steps=1)` 产生单 bin 频率栅，会触发 PyTorch 的 `(1,1)` 广播（质心恒为 0），而 `steps=0` 产生空张量并走另一条广播错误路径；这些都是 torch 张量实现细节，不是质心算法本身。`freqs` mode 仍单独给出 `n_fft=0` 的官方常量（该 mode 不经过广播）。

## 3. 五个官方 quirk

① `power=1.0`（幅度谱）与 `normalized=False` 是**唯一显式传参**，`center` / `pad_mode` / `onesided` / `return_complex` 全部走 spectrogram 默认值（分别 `True` / `"reflect"` / `True` / `None`），传参与缺省必须分开判。② `sample_rate // 2` 是整除截断：16000 与 16001 都给出 8000，负 `sample_rate` 也按 Python 向下取整（`-8000 // 2 = -4000`）。
③ 频率栅用 `1 + n_fft // 2` 个 bin（不是 `n_fft // 2`），且 `linspace` 的后半段由 `end` 反向计算，纯 Python 参考解必须复刻这个对称公式才能逐位一致。④ **分母对原始 `specgram` 求和、不广播**：`specgram.sum(dim=-2)` 要求至少 2 维，所以 1D 谱即使分子能广播也会抛 `IndexError ... got -2`。
⑤ 全零帧是 `0/0`，官方输出 `nan`（判题统一为字符串 `"nan"`）；若分母为 0 而加权分子非零，则按 IEEE 得 `+inf` / `-inf`。输出形状是 `shape[:-2] + shape[-1:]`，即丢掉频率维、保留前导维与时间维。

## 4. 判题账本

`fixtures868.py` 从逐字官方环境导出 33 个常量，`cases868_checks.py` 提供 50 个 `check_*(solve)`：频率栅的整除截断与对称公式、12 个 spy kwargs、完整 ledger 的中间量、2D / 3D / 4D 形状、六类错误字典、`census` 的 14 个源码事实、`unknown mode` 的完整消息，以及三条硬化检查——`census` 每次返回全新 dict、`0/0` 与 `±inf` 分支、`shapes(n_fft=2)` 三个 layout 都返回真实错误 dict（而非被吞掉）。

## 5. 验证流水线

```text
official868.py smoke        8 modes, EXIT=0
probe868.py                 117/117 ALL GREEN (12 segments)
probe_solution868.py        11769 checks / 68 configs / 0 bad
verify868.py                43488 checks / 84 configs / 99 props / 0 fails
fixtures868.py              33 constants
driver868.py                50/50 ALL ACCEPTED
mut868.py                   155 mutants / 4 proven equivalents / 151 CAUGHT
equiv868.py                 4 mutants x 1692 comparisons = 6768, 0 diffs
```

三层对拍的第三层与参考解**不共享任何算法**：用 `Fraction` 精确有理数重算频率栅与加权和，再用与公式无关的代数性质校验——单频脉冲谱的质心必等于该 bin 频率、质心必落在 `[f_min, f_max]`、对某一频率列整体缩放 `s` 后分子分母同比例变化而质心不变。50 个判题用例全部由官方常量驱动。

## 6. 变异与等价留证

首轮 155 项变异有 8 个 SURVIVED 与 5 个锚点错误。5 个锚点错误是因为 `c["sample_rate"] // 2` 与 `1 + c["n_fft"] // 2` 在 `_freq_vec` 与 `_ledger` 各出现一次，改用显式出现序号定位。8 个存活项中，4 个是**判题盲区**并已补硬化用例捕获：`census` 返回同一对象、分母为 0 时的 `±inf` 分支、`shapes` 的异常吞噬分支。
另外 4 个是**语义真等价**，由 `equiv868.py` 在约定输入域的 188 组配置 × 9 个 mode 上逐项比较完整 `solve()` 输出（含异常类型与消息）证明，共 6768 次比较零差异：`e-linspace-msg` / `l-steps-neg` 命中的 `steps<=0` 分支在 `n_fft>=2` 时不可达；`l-step-num` 因 `start` 恒为 0 使 `end-start == end+start`；`er-odd-sr` 因 `16001//2 == 16000//2` 输出相同。**不删除任何变异**，最终 151 项有效变异全 CAUGHT / 0 SURVIVED / 0 ERROR。

## 7. 注册与护栏

`register868.py` 最小侵入注册 `audio` 19→20、中英 description 各追加 868 口径账长段，复核 `846 registered = 846 disk / max_order=868 / 无 dangling / 无 unregistered`。`todos.md` 更新为 846 题 / audio 20 / 下一批从 869 起，并把 868 从「已立项」改为「已落地」。落地后跑 selftest `submit` / `run` 双模式、全库双语校验、短护栏与独立 `--basetemp` 全量护栏。按约定改完**先不 git commit**，等用户确认。
