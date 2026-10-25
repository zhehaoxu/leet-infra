# Assembling an MLP Block in Detail

## What This Problem Practices

Use `nn.Module` to assemble the most typical FFN block, "fully connected → ReLU → fully connected", and understand three conventions of the PyTorch module system.

## Structural Convention

The hidden dimension is $\text{hidden}=\text{in\_features}\times\text{hidden\_multiplier}$:

$$ \text{fc1}:\ \text{in}\to\text{hidden},\quad \text{ReLU},\quad \text{fc2}:\ \text{hidden}\to\text{out} $$

`forward` is simply `self.fc2(self.act(self.fc1(x)))`. The `Linear(d, 4d)→act→Linear(4d, d)` inside a Transformer has exactly this shape (multiplier=4).

## Three Iron Rules of nn.Module

1. **Call `super().__init__()` first**: this must be the first statement in `__init__`; otherwise the module registry does not exist yet when you later assign `self.fc1=...`, and the parameters will not be registered.
2. **Assign submodules directly to `self.xxx`**: assigning `nn.Linear`/`nn.ReLU` instances as attributes registers them automatically, so `.parameters()`, `.to(device)` and `.state_dict()` pick them up. Putting them in a plain Python list means they are not tracked.
3. **Write only `forward`, and call `__call__`**: externally use `module(x)` (which triggers `__call__`→hooks→`forward`) rather than calling `module.forward(x)` directly.

## Parameters and Initialization

`nn.Linear` uses Kaiming-uniform initialization by default and includes a bias. The judge only cares that the structure and the forward output's shape/values match; it does not compare random weights, so **do not override `_reset_parameters` yourself** unless the problem statement asks for it.

## Validation Convention

When `hidden_multiplier < 1`, `solve` raises `ValueError` (the hidden dimension is at least equal to the input dimension); this is the negative-path test point.

## Related

linear-layer is the math of a single `nn.Linear`; nn-module-block composes it into a trainable subnetwork; optimizer-training-loop demonstrates how those parameters get updated.
