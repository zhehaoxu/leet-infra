# 量化：PolarQuant 的极坐标成对量化

## 题目描述

Key 过了 RoPE 之后，$(d,\; d + D/2)$ 这一对通道其实是同一个二维向量的两个分量：旋转只改这个对的角度，不改它的模长。PolarQuant 就顺着这个结构把 key 折成极坐标 $(\phi, r)$——角度用一套低比特码、半径用另一套低比特码，两个码再拼进同一个整数 `indices = (rho << tbits) + theta`；反量化时按码字查回角度和半径，取 $\cos/\sin$ 就把 key 还原出来。角度和半径各自都在 token 维做 min-max 量化，所以同一组里的 token 共享一对 $(\text{min},\text{scale})$。

上游的解码是 GPU kernel，本题按作者在 `kernel4group.py` 的 `__main__` 里贴出的「用 PyTorch 检查我们的 kernel」参考实现出题，逐条对齐它的取整方向、分母和 bin 中心。

## 规则

记 `keys` 为 $(B,H,L,D)$，$D$ 为偶数、$D_h = D/2$；把 token 轴切成 $L/G$ 组，每组 $G=$ `group_size` 个 token。内部一律 float64，返回值按输入 dtype。

1. **配对**：第 $d$ 对是 $\big(\text{keys}[\ldots, d],\ \text{keys}[\ldots, D_h + d]\big)$，即**前后两半**配对（RoPE `rotate_half` 的那一对），不是奇偶交错。记前半 $x_{lo}$、后半 $x_{hi}$。
2. **极坐标**：$\phi=\operatorname{atan2}(x_{hi}, x_{lo})\in(-\pi,\pi]$，负值加 $2\pi$ 折回 $[0,2\pi)$；$r=\sqrt{x_{lo}^2+x_{hi}^2}$。
3. **分组 min-max**：对每个 $(B,H,\text{组},d)$，只在**该组的 $G$ 个 token** 上取 min/max（归约的是 token 轴，不是通道轴）：$\text{scale}=(\max-\min)/2^{\text{bits}}$——分母是 $2^{\text{bits}}$，不是 $2^{\text{bits}}-1$。角度用 `tbits`、半径用 `rbits`。
4. **码字**：$\text{code}=\operatorname{clamp}\big(\lfloor (x-\min)/\text{scale}\rfloor,\ 0,\ 2^{\text{bits}}-1\big)$，**向下取整**而不是四舍五入；$\text{scale}=0$（组内恒定）时码字取 $0$。角度码记 $\theta$、半径码记 $\rho$。
5. **重构取 bin 中心**：$\hat{x}=(\text{code}+0.5)\cdot\text{scale}+\min$，于是 $\hat\phi=(\theta+0.5)t_{\text{scale}}+t_{\min}$、$\hat r=(\rho+0.5)r_{\text{scale}}+r_{\min}$，再
   $\text{recon}[\ldots,d]=\hat r\cos\hat\phi$、$\text{recon}[\ldots,D_h+d]=\hat r\sin\hat\phi$。
6. **打包**：$\text{indices}=(\rho \ll \text{tbits}) + \theta$，落在 $[0,\,2^{\text{rbits}+\text{tbits}})$。

返回 6 键字典：`phi`、`radii`、`theta`、`rho`、`indices` 的形状都是 $(B,H,L/G,G,D_h)$，前两个跟随输入 dtype、后三个是 int64；`recon` 是 $(B,H,L,D)$ 的反量化重建，跟随输入 dtype。

## 任务

实现 `solve(keys, group_size=128, rbits=4, tbits=4)`：

- `keys`：四维 float32 或 float64 张量 $(B,H,L,D)$，只读；$D$ 必须是偶数，$L$ 必须是 `group_size` 的正整数倍；
- `group_size`、`rbits`、`tbits`：正整数（`bool`、`4.0` 这类都算非法）；$r_{\text{bits}}$ 定半径、$t_{\text{bits}}$ 定角度。

非法输入抛 `ValueError`。全程不得修改入参。

## 示例

$B=H=1$、$L=4$、$D=4$、`group_size=4`、`rbits=tbits=2`。四个 token 把 $d=0$ 这一对摆成角度 $0.1/0.9/1.6/2.0$、半径 $1/2/3/4$，$d=1$ 那一整组恒为 $(1,1)$。于是 $t_{\text{scale}}=(2.0-0.1)/4=0.475$、$r_{\text{scale}}=(4-1)/4=0.75$：$1.6$ 与 $2.0$ 都落进最后一个 bin，$\theta=[0,1,3,3]$；半径正好是四个 bin 的 centre 候选，$\rho=[0,1,2,3]$。

```
keys    = [[[0.9950041652780258, 1.0, 0.09983341664682815, 1.0],
            [1.2432199365413290, 1.0, 1.5666538192549670, 1.0],
            [-0.0875985669038664, 1.0, 2.9987208091245150, 1.0],
            [-1.6645873461885700, 1.0, 3.6371897073027270, 1.0]]]

phi     = [[[[[0.0999999999999999, 0.7853981633974483],
              [0.9000000000000001, 0.7853981633974483],
              [1.6000000000000000, 0.7853981633974483],
              [2.0000000000000000, 0.7853981633974483]]]]]
radii   = [[[[[1.0, 1.4142135623730951], [2.0, 1.4142135623730951],
              [3.0, 1.4142135623730951], [4.0, 1.4142135623730951]]]]]
theta   = [[[[[0, 0], [1, 0], [3, 0], [3, 0]]]]]          # int64
rho     = [[[[[0, 0], [1, 0], [2, 0], [3, 0]]]]]          # int64
indices = [[[[[0, 0], [5, 0], [11, 0], [15, 0]]]]]        # (rho << 2) + theta

recon   = [[[1.2974299748897400, 1.0, 0.4553026029550115, 1.0],
            [1.4613318197185730, 1.0, 1.5427683924290140, 1.0],
            [-0.5477784488661208, 1.0, 2.8223330368611400, 1.0],
            [-0.6906771746572827, 1.0, 3.5585938290857850, 1.0]]]
```

第 3 个 token 的重构角是 $0.1+3.5\times0.475=1.7625$、半径 $1.0+2.5\times0.75=2.875$，两者都严格落在组跨度内部；$d=1$ 的常数组码字全 $0$，重构精确回到 $(1,1)$。

## 约束

- **两半配对**：第 $d$ 对是 $(d,\ D_h+d)$。按奇偶交错去读，$(2,5)$ 这一对会被拆成 $(1,2)$ 与 $(0,5)$，$\operatorname{atan2}(5,2)$ 变成 $\operatorname{atan2}(2,1)$，$\phi$、$\theta$、`recon` 一起错。
- **分母是 $2^{\text{bits}}$**：最大元 $\lfloor(\max-\min)/\text{scale}\rfloor$ 本来是 $2^{\text{bits}}$，靠夹回才落在 $2^{\text{bits}}-1$；分母若写成 $2^{\text{bits}}-1$，最大码字就停在 $2^{\text{bits}}-2$，整组分箱都偏一格。
- **折负角不是装饰**：$-0.1$ 折成 $2\pi-0.1$ 之后组内最小值换成另一个角，全组码字跟着平移；$\operatorname{atan2}(0,3)=0$ 本来非负，不会被折成 $2\pi$。
- **重构取 bin 中心**：$(\text{code}+0.5)\cdot\text{scale}+\min$。写成 $\text{code}\cdot\text{scale}+\min$ 会把每个 bin 压到下边界，最大元还会掉到 $\max$。
- **归约轴是 token**：min/max 沿组内 $G$ 个 token 取；沿 $D_h$ 取会让同一 token 的两半混进同一组。
- **常数组给码字 $0$**：上游那一步是 $0/0$ 的 NaN，本题把 $\text{scale}=0$ 的组直接判成码字 $0$——此时 $(0+0.5)\cdot 0+\min=\min$，重构精确等于输入。别改成 `clamp_min(eps)`。
- **打包宽度跟着 `tbits` 走**：`indices` 是 $\rho$ 左移 `tbits` 位再加 $\theta$，改 `tbits` 就改打包，改 `rbits` 只改 $\rho$ 的上界。
