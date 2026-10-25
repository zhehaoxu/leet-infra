# ELU Explained

## Definition

$$ \text{elu}(x)=\begin{cases} x, & x>0\\ \alpha(e^{x}-1), & x\le 0\end{cases}\qquad(\alpha \text{ defaults to }1) $$

## Design Motivation

- The negative region saturates exponentially to a finite asymptote $-\alpha$ (rather than being unbounded), which brings the **mean closer to 0** and accelerates convergence; at the same time the negative gradient is nonzero, alleviating ReLU's "dead neurons."
- It is continuous and differentiable at $x=0$ (left derivative $=\alpha e^0=\alpha$, right derivative $=1$; for $\alpha=1$ it is $C^1$ smooth).

## Implementation & Numerical Notes

- `torch.where(x>0, x, alpha*(torch.exp(x)-1))`, or compute `exp` first and then select by sign.
- When $x$ is very negative, $e^x\to0$ and the output $\to-\alpha$, without underflow problems; but for $e^x-1$ with $x$ close to 0, `torch.expm1(x)` is more accurate (avoiding catastrophic cancellation).
- Backward: for $x\le0$ the gradient is $=\alpha e^x$ (it approaches 0 but is not identically 0).

## Lineage

Leaky-ReLU is the linear version, "multiply the negative region by a small slope"; ELU is the smooth version, "exponential saturation in the negative region"; SELU = an ELU variant with fixed $\lambda,\alpha$ scaling used in self-normalizing networks.
