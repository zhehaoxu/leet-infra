# 详解：把 LSTM 反向拆成「门」与「时间」两条账

## 出处

「手写单层 LSTM 的前向与 BPTT」是语音/时序方向与训练框架方向的经典白板题，也是检验候选人是否真的推过公式的利刃——因为 LSTM 反向有**两处几乎所有人第一次都会写错**的地方：`tanh(c)` 前的 \(o\) 因子，以及门切分顺序。

## 门顺序是 (i, f, g, o)，不是教科书常见的 (i, g, f, o)

论文里习惯按 `input, cell-candidate, forget, output` 叙述，而 PyTorch 的 `weight_ih_l0` 行块实际是 `i | f | g | o`。本题用 `nn.LSTM` 作 oracle 正是为了把这条钉死：门切错时前向 \(h\) 就会与内核不一致，对拍立刻失败。安全做法是同时打印 `c_t` 与内核输出比对，或穷举四种排列取逐位相同者。

## dc_t 里那个 o 因子

由 \(h_t=o_t\tanh(c_t)\) 得 \(\partial h_t/\partial c_t=o_t(1-\tanh^2c_t)\)，因此

$$dc_t = dc_{t+1} + dh_t\odot o_t\odot(1-\tanh^2c_t)$$

只写 \(dh(1-\tanh^2c)\) 是最常见的漏因子，它在 \(T=1\) 就错，藏不住。同理 \(do=dh\tanh(c)\) 而不是 \(dh\cdot c\)。折进 \(dgates\) 时，sigmoid 门的导数是 \(y(1-y)\)（漏掉前面的 \(y\) 会让 \(di,dg,df,do\) 的量级全变），tanh 门是 \(1-y^2\)。

## 时间账与偏置账

- \(dx_t=dgates_tW_{ih}\)、\(dh_{t-1}=dgates_tW_{hh}\)、\(dc_{t-1}=dc_t\odot f_t\)：三个量都靠同一份 \(dgates_t\) 往回传，`dh_next/dc_next` 要在循环里持续更新而不是每步重置。
- \(dW_{ih}=\sum_t dgates_t^T x_t\)、\(dW_{hh}=\sum_t dgates_t^T h_{t-1}\)：**跨时间累加**，且 \(h_{-1}=h_0\) 也要参与第二项。
- \(db_{ih}=db_{hh}=\sum_t dgates_t\)：因为 `gates` 里两个偏置都是全长相加，二者必然相同（测试用 `check_bias_pairing` 断言 `got[3] == got[4]`）。
- 截断 BPTT：末状态梯度可以整份为 0，但本步产生的分量仍要继续往回传。

## 与邻近题的区别

`ag-activation-backward` 只处理逐元素导数；本库的 `lstm-cell`、`gru_cell` 只做单步前向、不涉及时间累加。本题的独有考点是「门顺序 + o 因子」这两个纯记忆无法绕过的事实。

## 对拍口径与复杂度

oracle 用 `nn.LSTM(E, Hid).to(xs.dtype)`，把四个参数 `copy_` 进去、`h_0/c_0` 传 3-D `(1, N, Hid)`，以 `loss=(hn*dhT).sum()+(cn*dcT).sum()` 取梯度后按 `(dxs, dW_ih, dW_hh, db_ih, db_hh, dh0, dc0)` 重排比对，float64 atol=1e-8。示例是 \(T=N=Hid=1\) 的最小情形，七个量都能手算复核。复杂度 \(O(T\cdot N\cdot E\cdot Hid)\)。
