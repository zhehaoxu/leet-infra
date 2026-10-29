# Solution Notes · MiniLM

## Why distill the attention distributions

The teacher's final logits only reflect "what to predict", not "how the model organizes information internally". The self-attention distribution is per-head and per-position: how much weight head $h$ gives key $j$ when processing query $i$. MiniLM argues this is precisely the structural knowledge most easily lost in downsizing, so it puts a KL directly on each head's attention distribution. Because the distributions are normalized, the KL is 0 exactly when the attention patterns coincide, and it is not perturbed by an overall scale the way a logits MSE would be.

## The dual role of the mask

The same `mask` marks both valid keys and valid queries. As a key: masked positions must have probability exactly 0 (fill with $-\infty$ so the softmax yields 0), otherwise padding steals attention mass. As a query: masked rows are entirely invalid and enter neither numerator nor denominator. Masking only the key side and forgetting the row side makes the loss depend on the padding ratio in a variable-length batch — the most common implementation bug here.

## Why divide by "valid queries × H"

The KL is computed per (query, head), and there is no reason to weight heads differently (the paper treats all heads equally). Summing over valid queries and dividing by "valid queries × heads" equals "average over queries within each head, then average over heads". Writing the denominator as $L\cdot H$ counts padding rows as zero and dilutes the loss.

## Relation representations and the MSE

Besides attention, the paper distills token relations: it forms scaled dot products of query/key/value vectors into several $L\times L$ relation matrices and aligns them with an MSE (compared with aligning hidden vectors element-wise, relation matrices are insensitive to dimension ordering). This problem abstracts that step into three $(B,3,D)$ relation vectors — the `3` standing for three relation types (e.g. CLS·CLS, CLS·token, token·token) — aligned by a plain element-wise MSE averaged over all $B\times3\times D$ elements. It is **added** to the attention term rather than weighted: in the original implementation the two are of comparable magnitude, so a direct sum suffices.
