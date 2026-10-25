# Reward Modeling: Bradley–Terry Pairwise Ranking Loss

## Problem

The first step of RLHF is to train a **reward model** $r_\phi(x,y)$: given a prompt and a response, it outputs a scalar score such that the response humans prefer scores higher. The usual **Bradley–Terry** model turns a "pairwise comparison" into a probability: $P(y_w \succ y_l) = \sigma(r_w - r_l)$, which is then trained by maximum likelihood (= log loss):

$$\mathcal{L}_\text{RM} = -\,\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\Big]$$

The reward model itself is normally a backbone network plus a **value head** that maps a hidden state to a scalar (a linear $D\to1$). HF `RewardTrainer` does the following: it takes the hidden state of the **last valid token** of each sequence, pushes it through the value head, and obtains the reward for the whole response (sequence pooling). This problem gives you the per-token hidden states and attention masks of a chosen and a rejected batch, and asks you to complete the chain "pool the last valid token → score with the value head → Bradley–Terry loss".

## Rules

Given (batch size $N$, sequence length $L$, hidden dim $D$): `chosen_hidden`, `rejected_hidden`, each $(N, L, D)$; `chosen_mask`, `rejected_mask`, each $(N, L)$ (1 = valid token, 0 = padding); `weight` $(D,)$ and the scalar `bias` (the value head parameters):

1. **Pooling**: take the hidden state of the **last valid token** of each sequence — i.e. the largest index $j^\ast$ with mask$=1$, and take $h = $ hidden$[b, j^\ast, :]$ ($(D,)$)
2. **Scoring**: $r = h \cdot $`weight` $+$`bias` (a dot product down to a scalar), yielding `r_chosen` and `r_rejected` (each $(N,)$)
3. **Loss**: per pair $-\log\sigma(r_\text{chosen} - r_\text{rejected})$, using a stable log-sigmoid, then take the **mean** over the batch

Return a 0-dim `torch.Tensor` = the mean reward modeling loss.

## Example

```
N=2, L=3, D=2, weight=[1.0,0.5], bias=0.1
chosen_hidden   = [[[0.1,0.3],[0.2,-0.1],[0.5,0.2]],
                   [[0.3,0.0],[0.9,0.4],[0.0,0.0]]]
rejected_hidden = [[[0.4,0.1],[0.2,0.5],[0.1,-0.2]],
                   [[0.6,0.2],[0.7,-0.3],[0.0,0.0]]]
chosen_mask     = [[1,1,1],[1,1,0]]      # last valid token of the second sequence is at idx1
rejected_mask   = [[1,1,0],[1,1,1]]

pool: hc=[chosen[0,2]=[0.5,0.2], chosen[1,1]=[0.9,0.4]]
      hr=[rejected[0,1]=[0.2,0.5], rejected[1,2]=[0.0,0.0]]
score: r_c=[0.5+0.5*0.2+0.1=0.7, 0.9+0.2+0.1=1.2]
      r_r=[0.2+0.25+0.1=0.55, 0+0+0.1=0.1]
diff =[0.15, 1.1]
loss  = -mean(logsigmoid([0.15,1.1])) = 0.454146
```

## Constraints

- CPU float32; every sequence has at least one valid token.
- "Last valid token" means the position with mask$=1$ at the largest index (padding is usually on the right, but do not assume the mask is contiguous).
- log-sigmoid must be numerically stable; return a 0-dim scalar tensor.
