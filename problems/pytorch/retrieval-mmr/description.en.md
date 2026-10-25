# Retrieval: Maximal Marginal Relevance (MMR) Reranking

## Problem

Ranking purely by relevance returns a pile of nearly duplicate documents (the "redundancy" problem in retrieval and summarization). **Maximal Marginal Relevance** (MMR, Carbonell & Goldstein, SIGIR 1998) greedily picks, at each step, the document that is "both relevant and as different as possible from the already-selected set":

\[ \text{MMR}=\arg\max_{d\in R\setminus S}\Big[\lambda\,\cdot\,\text{sim}(d, q)\;-\;(1-\lambda)\,\cdot\,\max_{s\in S}\text{sim}(d, s)\Big] \]

where \(S\) is the selected set and \(q\) the query. It underlies LangChain's `MMRRetrieval` and all kinds of "diversity reranking / result deduplication". This problem implements MMR's greedy selection sequence.

## Rules

Implement `solve(query, docs, lam, k)`. The input `query` (float32, \((D,)\)), `docs` (float32, \((n, D)\)), `lam` (\(\lambda\in[0,1]\)), `k` (number selected, \(k\le n\)). \(\text{sim}\) is **cosine similarity** (the similarity between a zero-norm vector and any vector is 0). Select \(k\) documents one by one:
- First: pick the \(d\) with the largest \(\text{sim}(d,q)\) (i.e. pure relevance);
- Each subsequent one: over the unselected set, maximize \(\lambda\cdot\text{sim}(d,q)-(1-\lambda)\cdot\max_{s\in S}\text{sim}(d,s)\).

Return an int64 tensor of shape \((k,)\) giving **the selected document indices in selection order**. Ties go to the smallest document index.

## Examples

\(q=[1,1,0]\), \(docs=[[1,0,0],[0.9,0.1,0],[0,1,0]]\), \(\lambda=0.5, k=3\) (cosine: doc0=.707, doc1=.781, doc2=.707):

```
pick 1: highest relevance -> doc1
pick 2: sim(doc2, doc1)=.110 is far smaller than sim(doc0, doc1)=.994
     -> doc2 score 0.5*0.707-0.5*0.110=0.298 > doc0's -0.144 -> pick doc2 (more marginal)
pick 3: only doc0 left
out = [1, 2, 0]
contrast: with λ=1 (pure relevance) the output is [1, 0, 2] — doc0 is pushed behind doc2 because it is redundant with doc1
```

## Constraints

- CPU float32; \(\lambda\in[0,1]\), \(1\le k\le n\).
- \(\lambda=1\) degenerates to pure-relevance top-k; \(\lambda=0\) pursues diversity alone.
- Similarity is always cosine; the similarity of a zero-norm vector is 0.
- Ties go to the smallest document index, ensuring determinism.
