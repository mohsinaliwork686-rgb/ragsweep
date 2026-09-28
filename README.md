# ragsweep

**Stop guessing your chunk size.** Point it at your documents and a small set of labelled
questions, and it tells you which retrieval settings actually find the right passage.

> Week 2 of a 10-week build. Currently in progress — see
> [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the design.

---

## The problem

Every RAG system makes four choices before it retrieves anything: chunk size, overlap, embedding
model, and whether to use vectors, keywords, or both.

Almost nobody measures them. `chunk_size=512, overlap=50` gets copied from a tutorial and never
revisited. Then retrieval is mediocre, and the blame lands on the generation model, the prompt,
or "hallucination" — when the real problem is that the correct passage never reached the context
window at all.

**If the right passage is not retrieved, no amount of prompt engineering fixes the answer.**
Retrieval is the ceiling on everything downstream, and it is the part nobody checks.

## What it does

```console
$ ragsweep run

  retriever  strategy   chunk  overlap  recall@5    MRR   nDCG@5   time
  hybrid     sentence     512       64      0.93   0.86     0.89    12s   <- best
  dense      sentence     512       64      0.89   0.81     0.85    11s
  dense      fixed        512       64      0.84   0.74     0.79    11s
  bm25       sentence     512       64      0.77   0.71     0.74     1s
  dense      fixed       1024      128      0.71   0.60     0.66     9s
  dense      fixed        256        0      0.64   0.55     0.60    14s

Best: hybrid / sentence / 512 / 64
Written to results.json
```

An answer, measured, instead of a guess.

## Why there is a keyword baseline

`bm25` is classic keyword ranking. No model, no embeddings, instant. It is in the sweep on
purpose: on short factual documents it often beats dense retrieval, and hybrid usually beats
both. A tool that only measured embeddings would hide that and quietly push people toward the
slower, more expensive option.

## Status

| Day | |
|-----|-|
| Mon | Plan and README ✅ |
| Tue | Corpus loading, chunking strategies, metrics |
| Wed | Embeddings with cache, the three retrievers, the sweep |
| Thu | Results schema, reporting, CLI, tests, CI |
| Fri | Example corpus and labels, real numbers, publish |

## Non-goals

- **It does not measure answer quality.** Retrieval only. No generation, no LLM-as-judge.
- **It needs no paid API.** Everything runs locally and free.
- **It is not a RAG framework.** It answers one offline question: which settings should I use?

## License

MIT — see [LICENSE](LICENSE).
