# Solution Notes · LLM.int8()

## Emergent Feature Dimensions: Why Not All-INT8

The authors found that Transformer activations contain **"emergent dimensions" that are extremely few but enormous in magnitude** (about 0.1%), appearing nearly in every layer and on every token, with values reaching 100 times those of normal channels. An integer quantizer must widen its range to accommodate these extremes, which "drowns" the normal channels into very few quantization levels; and once these outliers are clipped, the error is amplified by their own large weights, collapsing the entire output.

## Mixed-precision Decomposition (vector-wise quantization)

Split $XW$ along the input channels into two blocks, "outlier columns + normal columns":

$$XW = X_{\text{out}}W_{\text{out}} \;(\text{FP16}) \;+\; X_{\text{normal}}W_{\text{normal}}\;(\text{INT8})$$

Outlier columns are very few (<1%), so per-column FP16 products cost little; the normal columns make up the vast majority and run through the fast INT8 path. The sum of the two is nearly equivalent to full precision.

## This Problem Simplifies to the Weight Side

The judge does not introduce the activation $X$ (its outlier columns are determined dynamically only at inference time, which hurts determinism); instead it fixes this idea **onto the weights**: whether a column is kept verbatim (treated as an outlier, at FP precision) or quantized is decided by "does the column absmax exceed the threshold $\tau$". Returning the outlier columns verbatim is equivalent to "the FP16 branch makes no changes". This preserves the **discrete-versus-continuous dichotomy** at the core of LLM.int8() while allowing exact comparison on pure CPU.

## The SpQR / AWQ Lineage

- **LLM.int8()**: keeps entire outlier **columns** in FP; the granularity is the column.
- **SpQR**: keeps individual outlier **weights** (not whole columns) sparsely at high precision + row/column importance grouping; finer granularity.
- **AWQ**: keeps no FP at all; instead it **scales** the important columns and quantizes everything together.

All three answer the same question — "how to preserve a few important elements under low bit-width" — with the technique evolving step by step from "whole-column exemption → single-point exemption → scaling protection".

## Boundary Self-checks

`check_all_outlier_passthrough`: a very low threshold makes everything an outlier → the output equals the input (no quantization). `check_no_outlier_all_quant`: a very high threshold yields no outliers → everything goes through RTN. The two ends clamp the "column dichotomy" logic.
