# Alignment Training: GRPO Group-Normalized Advantage

## Problem

**GRPO** (Group Relative Policy Optimization, Shao et al., arXiv:2402.03300) drops PPO's value network: for the same prompt it samples a group of \(G\) responses, each getting a reward, and uses the **within-group** statistics as the baseline to standardize the rewards into advantages:

\[ A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)} \]

Each group is normalized independently (group mean / group std), so the advantages are naturally zero-mean and unit-variance, with no critic needed. The standardized advantages are then fed into the PPO-clip objective.

## Rules

Inputs are a flat `rewards` (\((N,)\), where \(N\) is an integer multiple of \(G\), split into groups of \(G\) consecutive entries) and the integer `group_size` (\(G\)). For each group compute \((r-\text{mean})/\text{std}\) (the standard deviation uses the population formula, ddof=0), and return `advantages` as float32 with the same shape \((N,)\). Do not use an off-the-shelf RL library.

## Example

`rewards=[1.0,2.0,3.0,0.0,0.0,6.0]`, `group_size=3` (two groups):

```
Group 0 [1,2,3]: mean=2, std=sqrt((1+0+1)/3)=0.816497 → [-1.224745, 0, 1.224745]
Group 1 [0,0,6]: mean=2, std=sqrt((4+4+16)/3)=2.828427 → [-0.707107, -0.707107, 1.414214]
advantages = [-1.224745, 0.0, 1.224745, -0.707107, -0.707107, 1.414214]
```

## Constraints

- CPU float32; `N % G == 0`; grouping is **consecutive** (reshape \((N/G, G)\)).
- The standard deviation uses the population formula (divide by \(G\), not \(G-1\)).
- Each group's mean and variance are computed only within that group, never across groups.
