# SELU Self-Normalizing Activation in Detail

## The Curve: a Scaled ELU

SELU is just ELU multiplied by a $\lambda$: the positive half-axis slope becomes $\lambda\approx1.05$, while the negative half-axis keeps the saturating lower bound $-\lambda\alpha$ ($\approx-1.758$). It is continuous at $x=0$ (both sides tend to 0) and differentiable, giving smoother gradients in the negative region than ReLU and an overall scaling that ELU lacks.

$$ \operatorname{SELU}(x)=\lambda\begin{cases}x,&x>0\\ \alpha(e^x-1),&x\le0\end{cases} $$

## Where "Self-Normalization" Comes From

Klambauer et al. 2017 proved that if the inputs are approximately Gaussian and the weights use the **SELU-specific initialization** ($\text{lecun_normal}$, a normal with variance $1/n$), then this pair $(\lambda,\alpha)$ is exactly a "fixed point": each layer's output still has mean 0 and variance 1. Deep networks therefore keep a stable activation distribution without BN/LN, and gradients neither explode nor vanish.

## Preconditions and Pitfalls

Self-normalization is **a whole recipe**; swapping in the activation alone is not enough:
- The weights must use `nn.init.lecun_normal_` (not the default kaiming/xavier);
- The inputs must first be standardized to mean 0 / var 1;
- **Do not apply dropout arbitrarily** (it breaks the fixed point); use **AlphaDropout** instead (a dedicated variant that preserves mean and variance).

Break any one of these and the self-normalizing advantage shrinks — which is also why SELU never became as universal as ReLU.

## Links to Other Problems

- It differs from `elu` by a single $\lambda$; compare the two to see how scaling changes the distribution's fixed point.
- It is an alternative way of thinking to normalization operators (layernorm/batchnorm): one relies on an external norm layer, the other internalizes normalization into the activation + initialization.
