# Solution Notes · QuaRot Hadamard Rotation

## Change the coordinate system and "outliers" disappear

SmoothQuant/AWQ/LLM.int8()/SpQR all handle outliers **in place**: either migrate the scale, exempt them, or rescale. QuaRot takes another path — if one coordinate is unusually large, **rotate the coordinate system** so that the energy is spread evenly over all coordinates, leaving no component of the rotated vector standing out, and the quantization range is no longer held hostage by a single extreme value.

The key is to use an **orthogonal** transform: $HH^\top=I$, so $X W = (XH)(H^\top W)$ is mathematically completely equivalent, and at inference time the rotation only needs to be absorbed into the adjacent linear layers, leaving the network function unchanged.

## Why Hadamard

Every entry of a Hadamard matrix is $\pm1$; after normalizing by $\frac{1}{\sqrt n}$ each entry is $\pm\frac{1}{\sqrt n}$. A single matrix multiply "mixes" any coordinate into all output coordinates with weight $\pm\frac{1}{\sqrt n}$ — effectively an optimal "spreader" that markedly lowers kurtosis, which is exactly the core statistic governing low-bit quantization error. The Sylvester recursion

$$H_{2m}=\begin{bmatrix}H_m&H_m\\ H_m&-H_m\end{bmatrix}$$

starting from $H_1=[1]$ gives a symmetric orthogonal matrix for $n=2^k$, hence $H^{-1}=H^\top=H$ — the inverse rotation uses the same matrix, which keeps the implementation minimal.

## The three-step pipeline in this problem

1. **Rotate**: $\widetilde W=W\,H$ ($C$ must be a power of two).
2. **Quantize**: per-tensor symmetric RTN in the rotated domain (scale $=\max|\widetilde W|$).
3. **Rotate back**: $\hat W=\widetilde W_q\,H$ to restore.

At 8 bits it is nearly reversible; `check_outlier_flattened` uses $[100,1,1,1]$ to show that even at 2 bits, after spreading, the maximum error of the inverse rotation stays very small, whereas without rotation the single 100 would monopolize the whole range.

## Orthogonality to other quantization techniques

Rotation is a **preprocessing** step and composes with any quantizer: in the QuaRot paper the rotated weights are still quantized to INT4/FP8, only the data distribution before quantization is friendlier. It addresses the same pain point as AWQ (scale protection) and SpQR (sparse exemption), but the idea shifts from "protect the outliers" to "eliminate the outliers".
