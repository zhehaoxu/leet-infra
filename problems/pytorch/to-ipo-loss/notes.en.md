# Solution Notes · IPO

## From "preferences are logistic" to "regress the margin directly"

DPO starts from Bradley–Terry: $P(y_w\succ y_l)=\sigma(r_w-r_l)$, then sets the implicit reward $r=\beta\log\frac{\pi_\theta}{\pi_\text{ref}}$. Azar et al. take a more general view: the preference probability is **any** monotone function $\Psi$ of the reward difference, and DPO is just the $\Psi=\sigma$ special case. Repeating the derivation for a general $\Psi$ yields an optimality condition; when $\Psi$ is the identity (the dual of the squared loss) that condition reduces to $\Delta=1/(2\beta)$. The loss $(\Delta-\frac{1}{2\beta})^2$ is then the regression form that makes the stationarity equation hold at the optimum.

## Why a squared loss cures overfitting

$-\log\sigma(\beta\Delta)$ has a nonzero gradient for every $\Delta$ (merely shrinking), so "already-correct" pairs keep being pushed to larger $\Delta$ and the implicit reward drifts further from the reference model. The squared loss has a gradient **exactly zero** at $\Delta=1/(2\beta)$ and pulls $\Delta$ back from either side: too small gets raised, too large gets lowered. That is the essential behavioral difference from DPO, and why the tests explicitly check "loss is exactly 0 when $\Delta$ hits the target".

## β changes the target, not the scale

$\beta$ appears in the target margin $1/(2\beta)$: a larger $\beta$ demands a smaller margin (more sensitive to deviation from the reference), a smaller $\beta$ demands a larger one. It is **not** multiplied over the whole loss — that would merely rescale the loss by $\beta$ (and the gradient likewise), equivalent to changing the learning rate, not IPO.

## Two invariances of the reference values

$\Delta$ depends only on the **differences** of two logprob pairs. Hence shifting both refs by the same $c$ changes nothing (a constant shift of the reference model cannot affect preferences); swapping the chosen/rejected sides together with their refs leaves $\Delta$ unchanged (no sign flip, no reordering), so the loss is unchanged. Each invariance has a test case, ruling out implementations that mix absolute ref values into the loss or swap only one side.
