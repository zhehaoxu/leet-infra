# torchaudio Mel / 线性滤波器组刻度账：HTK 与 Slaney 双刻度、张量布尔掩码、`n_mels+2` 点网格与三角斜率式、`norm` 与 `mel_scale` 两个独立开关、全零列 warning

## 1. 背景

`torchaudio.functional` 的 `_hz_to_mel` / `_mel_to_hz` / `_create_triangular_filterbank` / `melscale_fbanks` / `linear_fbanks` 是 mel 与线性滤波器组的构造链。它们只做刻度换算、`linspace` 网格、三角权重与面积归一，不含 STFT、音频 IO 或模型前向；五个函数自身**没有任何 RNG**。本题把它们用纯 Python 复刻成十个可独立判题的 mode，把全部数值用官方常量快照钉死。
全库查重：五个函数名在 `problems` 的全部 `meta.json` / `description*.md` / `notes*.md` 中零命中。342 是老式 mel 滤波器组题，把 Slaney 刻度与面积归一**绑定成一套口径**；852 是 Kaldi 口径（`1127*ln(1+f/700)`）加 VTLN / framing / mfcc 兼容层。866 直接判五个官方函数的刻度中间量与三角矩阵，覆盖 `linear_fbanks` 的 Hz 域等距、零列 warning 与 `norm` / `mel_scale` 的正交性，与两者边界清晰。

### 官方源码位置

```text
pytorch/audio main::src/torchaudio/functional/functional.py
  _hz_to_mel                    (lines 425-456)
  _mel_to_hz                    (lines 459-489)
  _create_triangular_filterbank (lines 492-515)
  melscale_fbanks               (lines 518-587)
  linear_fbanks                 (lines 590-633)

  _hz_to_mel / htk:
    return 2595.0 * math.log10(1.0 + (freq / 700.0))
  _hz_to_mel / slaney:
    f_sp = 200.0 / 3 ; min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    mels = (freq - f_min) / f_sp
    if freq >= min_log_hz:  # scalar if, NOT a tensor mask
      mels = min_log_mel + math.log(freq / min_log_hz) / logstep

  _mel_to_hz / slaney:
    freqs = f_min + f_sp * mels
    log_t = mels >= min_log_mel        # TENSOR boolean mask
    freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

  _create_triangular_filterbank:
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes   = slopes[:, 2:] / f_diff[1:]
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

  melscale_fbanks:
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)
    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    if (fb.max(dim=0).values == 0.0).any(): warnings.warn(...)

  linear_fbanks:
    f_pts = torch.linspace(f_min, f_max, n_filter + 2)
    # no norm, no mel_scale, no warning
```

`official866.py` 用 `ast.get_source_segment` 抽出五个顶层函数并 `exec` 进同一共享 NS，官方函数体**一字未改**；注入对象只有真实的 `torch` / `math` / `warnings` 与类型别名，全局默认 dtype 固定为 `float64`，因此 `linspace` / `pow` / `exp` 链可与纯 Python 参考解对拍。ledger 层（all_freqs / m_pts / f_pts / f_diff / enorm / zero_columns / column_max / warnings）由 `official866.py` 用真实 torch 复算，是判题常量的唯一权威来源。

## 2. 口径约定

### 十个 mode

`solve(mode, **kw)` 分发十路：`hz_to_mel` / `mel_to_hz` 为单值刻度，`mel_grid` / `hz_grid` 为多值刻度，`triangular` 直接判斜率式矩阵，`mel_fbanks` / `linear_fbanks` 判完整矩阵，`fbank_ledger` 判全部中间量，`errors` 判七种配置是否抛错，`census` 判源码级事实。除 `unknown mode` 抛 `ValueError` 外，所有异常都收敛成 `{type, msg}` 字典返回，消息逐字对齐官方。

### 刻度与网格

HTK 是 `2595*log10(1+f/700)`，逆向 `700*(10**(m/2595)-1)`。Slaney 在 1000 Hz 分段：低频 `f/(200/3)`，高频 `min_log_mel + log(f/1000)/logstep`，其中 `logstep=log(6.4)/27`，6400 Hz 恰好比断点高 27 mel；逆向用 `log_t = mels >= min_log_mel` 的**张量布尔掩码**加原地赋值，不是标量 if。`min_log_mel = 1000/(200/3) = 14.999999999999998`（不是恰好 15），所以 15.0 落在对数分支。
mel 网格是 `n_mels+2` 个 mel 点经 `linspace` 再逆变换成 `f_pts`；`all_freqs = linspace(0, sample_rate // 2, n_freqs)` 用整除截断，16001 与 16000 给出同一端点。三角矩阵用**移位分母**：下降沿除以 `f_diff[:-1]`、上升沿除以 `f_diff[1:]`，再 `max(0, min(down, up))` 夹到非负。

## 3. 五个官方 quirk

① `norm` 与 `mel_scale` 是**两个独立开关**：`norm="slaney"` 只把列乘以 `enorm=2/(f_pts[j+2]-f_pts[j])`，完全不改变刻度网格；四种组合必须两两不同。② `linear_fbanks` 在 Hz 域等距取点，函数签名里没有 `norm` / `mel_scale`，也**从不 warn**，即使很多列全零；传入非法 `mel_scale` 会被静默忽略。
③ 全零列 warning 在 `fb *= enorm` **之后**判定，但 `enorm` 为正数，零列不会复活；文本含 `n_mels` 与 `n_freqs` 两个插值，必须逐字保留。④ `n_mels=0` 退化为每行空列表，无异常、无 warning；`f_min=-1` 不被校验，HTK `log10` 仍合法而给出负 mel。⑤ `norm` 只接受 `None` / `"slaney"`，`mel_scale` 只接受 `"htk"` / `"slaney"`，大小写敏感。

## 4. 判题账本

`fixtures866.py` 从逐字官方环境导出 121 个常量，`cases866_checks.py` 提供 123 个 `check_*(solve)`：既有数值容差比较，也有针对 `torch.linspace(float64)` 的**逐位精确相等**断言（参考解用 PyTorch 的对称公式 `i<steps//2 -> start+i*step`，否则 `end-(steps-1-i)*step` 复刻，实测与 torch 零差异）。另有异常类型、warning 文本、叶子必须为 Python `float`、四种 `norm`×`mel_scale` 组合两两不同、端到端用 `triangular` 重算完整矩阵等硬化检查。

## 5. 验证流水线

```text
official866.py smoke          10 modes, EXIT=0
probe866.py                   821/821 ALL GREEN (12 segments)
precision866.py               symmetric linspace diff = 0
probe_solution866.py          6344 good / 0 bad
verify866.py                  37578 checks / 69 configs / 0 fails
fixtures866.py                121 constants
driver866.py                  123/123 ALL ACCEPTED
equiv866.py                   8 mutants x 2700 comparisons / 0 diffs
mut866.py                     168 mutants - 8 equivalents = 160 CAUGHT
```

三层对拍的第三层与参考解**不共享任何算法**：用 `fractions.Fraction` 做精确有理数 `linspace`，用 `log1p` / `expm1` / `pow(6.4, x)` 变换重写两条刻度，用「左半上升、右半下降」的几何分段定义重写三角矩阵，用集合成员判定列是否全零。八个被判为真等价的变异（不可达的 `steps<=0` 分支、两个断点处数值完全相同的阈值比较、未被消费的第三个返回值、只含单元素列表的 `[0]` 与 `[-1]`、`errors` 中不改变结果的两个探针配置）均由 `equiv866.py` 在 270 组配置 × 10 mode 上逐项证明，绝不以删除变异的方式让测试变绿。

## 6. 工程教训与边界

第一个坑是 `torch.linspace` 的**对称后半段**：朴素的 `start+i*step` 在 8 点网格上有约 9e-13 偏差，容差 1e-8 完全抓不到，必须用精确 `==` 断言才能杀掉把对称公式改成单向公式的变异。第二个坑是 `mel_to_hz` 的张量掩码阈值 `>=` 与 `>` 在 `mel == min_log_mel` 处给出**同一个 double**（`1000*exp(0)` 与 `f_sp*min_log_mel` 都是 1000.0），同理 `hz_to_mel` 的 1000 Hz 断点两分支也同为 14.999999999999998，因此这两个变异是真等价而非判题盲区。
第三个坑是 `n_freqs=0`：官方 `melscale_fbanks` 的 `fb.max(dim=0)` 会抛 `IndexError`，而 `linear_fbanks` 的异常来自参考环境的 ledger 辅助层，属于 torch 实现细节而非题目语义，故明确排除在输入域外（`n_freqs>=1`、`n_mels>=0`）。`linear_fbanks` 传入非法 `mel_scale` 仍返回正常矩阵、以及全零列依然不 warn，则被补成硬化用例。

## 7. 注册与护栏

`register866.py` 最小侵入注册 `audio` 17→18、中英 description 各追加 866 口径账长段，复核 `844 registered = 844 disk / max_order=866 / 无 dangling / 无 unregistered`。`todos.md` 更新为 844 题 / audio 18 / 下一批从 867 起，并把 866 从「已立项」改为「已落地」。落地后跑 selftest `submit` / `run` 双模式、全库双语校验、短护栏与独立 `--basetemp` 全量护栏。按约定改完**先不 git commit**，等用户确认。
