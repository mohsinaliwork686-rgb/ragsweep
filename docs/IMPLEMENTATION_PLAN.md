# Implementation plan — ragsweep

> Written Monday 28 September 2026, before any code.
> Week 2 of the 10-week programme.

## 1. Problem

Every RAG system makes four choices before it retrieves anything:

- How big is a chunk?
- How much do chunks overlap?
- Which embedding model?
- Dense vectors, keyword search, or both?

Almost nobody measures these. People copy `chunk_size=512, overlap=50` from a tutorial and never
revisit it. Then retrieval quality is mediocre and the blame lands on the generation model, the
prompt, or "hallucination" — when the real problem is that the right passage never made it into
the context in the first place.

If the correct passage is not retrieved, no prompt engineering can save the answer. Retrieval is
the ceiling on everything downstream, and it is the part nobody measures.

`ragsweep` turns those four guesses into a measurement: point it at documents plus a labelled
question set, and it reports which combination actually retrieves the right passage.

## 2. Non-goals

- **It does not measure answer quality.** Only retrieval. No generation, no LLM-as-judge.
- **It does not need a paid API.** Everything runs locally and free.
- **It is not a RAG framework.** It does not serve queries in production. It answers one question
  offline: which settings should I use?
- **No web interface.** Output is a terminal table, a JSON file and a chart.

## 3. Surface

Designed first, because if this is awkward the rest does not matter.

```bash
ragsweep init                      # write a starter sweep.toml and a sample labels file
ragsweep run                       # run the sweep defined in sweep.toml
ragsweep run --corpus ./docs --labels ./q.jsonl --out results.json
ragsweep report results.json       # re-print the table from a saved run
ragsweep report results.json --chart chart.png --format csv
```

`sweep.toml`:

```toml
corpus = "./corpus"
labels = "./labels.jsonl"

retrievers  = ["bm25", "dense", "hybrid"]
strategies  = ["fixed", "sentence", "recursive"]
chunk_sizes = [256, 512, 1024]
overlaps    = [0, 64, 128]
models      = ["sentence-transformers/all-MiniLM-L6-v2"]
k           = [1, 3, 5, 10]

cache = ".cache/ragsweep"
```

**Changed from the Monday draft, which said `sweep.yaml`.** YAML would mean adding PyYAML
as a runtime dependency, and `tomllib` has been in the standard library since 3.11, which
is already the floor here. For a flat config of lists the two formats are equivalent, so
the dependency buys nothing.

Everything has a default, so `ragsweep run` with no flags works on a folder of Markdown.

### Why a keyword baseline is included

`bm25` is classic keyword ranking — no embeddings, no model, instant. It is in the sweep on
purpose. On short factual documents, BM25 frequently beats dense retrieval, and hybrid usually
beats both. A tool that only measures embeddings would hide that and quietly push people toward
the slower, more expensive option. Including the baseline is what makes the results trustworthy.

## 4. Architecture

Each stage is a plain function that takes data and returns data. No stage knows about the CLI.

```
  corpus/  ─────> load ─────> chunk ─────> index ──┐
                            (size,                  │
                             overlap,               ├──> search ──> score ──> report
                             strategy)              │       ▲
  labels.jsonl ──────────────────────────────────────┘       │
                                                     labels ─┘
```

| Module | Responsibility |
|--------|----------------|
| `corpus.py` | Walk a folder, read `.md` / `.txt` / `.html`, return `Document(id, path, text)` |
| `chunking.py` | Three strategies, all pure: `fixed`, `sentence`, `recursive`. In: text + size + overlap. Out: `list[Chunk]` |
| `labels.py` | Read and validate the ground truth file, with line-numbered errors |
| `embedding.py` | Text to vectors, with a disk cache. The only module that loads a model |
| `retrievers/` | `dense.py`, `bm25.py` (written here, no dependency), `hybrid.py` (reciprocal rank fusion) behind one `Retriever` interface |
| `metrics.py` | `recall_at_k`, `mrr`, `ndcg_at_k`. Pure maths, no dependencies |
| `sweep.py` | Build the grid, run each combination, collect results, show progress |
| `results.py` | The results schema, plus read and write |
| `report.py` | Table, CSV, chart |
| `cli.py` | Argument parsing only |

### Key decision: the sweep is not a naive nested loop

A naive implementation re-chunks and re-embeds for every combination. With 3 strategies × 3 sizes
× 3 overlaps × 3 retrievers × 4 values of k that is 324 runs, and it would take an hour.

The work is shared in layers, because each layer only depends on the ones above it:

```
strategy + size + overlap   ->  chunk once      (27 chunkings)
  + model                   ->  embed once      (27 embeddings, cached)
    + retriever             ->  index once      (81 indexes)
      + k                   ->  score is free   (retrieve top max(k) once, slice for each k)
```

That last line matters: retrieve the top 10 once, then compute recall@1, @3, @5 and @10 by
slicing the same result list. k costs nothing.

### Key decision: the embedding cache

Cache key is `sha256(model_name + chunk_text)`, stored as one `.npy` per model in `.cache/`.
The second run of a sweep should take seconds, not minutes. Without this, iterating on the report
formatting means waiting for the whole sweep every time, and I will stop iterating.

## 5. Data contracts

### Labels file — `labels.jsonl`

One JSON object per line. Kept deliberately small so it is realistic to hand-write.

```jsonl
{"id": "q1", "question": "How long does a refund take?", "relevant": ["handbook/refunds.md"], "tags": ["policy"]}
{"id": "q2", "question": "Who signs off refunds over $500?", "relevant": ["handbook/refunds.md", "handbook/approvals.md"], "tags": ["policy", "multi-hop"]}
```

**Relevance is recorded at document level, not chunk level.** This is the important call. Chunk
ids change every time chunk size changes, so chunk-level labels would have to be rewritten for
every configuration, which defeats the entire purpose. A retrieved chunk counts as correct when
it came from a document listed in `relevant`.

### Results file — `results.json`, schema version 1

**Week 6 (`evalboard`) reads this file**, so it is versioned from day one and the per-case detail
is included, not just the aggregates.

```json
{
  "schema": 1,
  "tool": "ragsweep",
  "tool_version": "0.1.0",
  "created_at": "2026-09-29T10:14:22Z",
  "corpus": { "path": "./corpus", "documents": 15, "chunks_hash": "3f9a..." },
  "labels": { "path": "./labels.jsonl", "questions": 30 },
  "runs": [
    {
      "id": "dense-fixed-512-64-MiniLM",
      "config": {
        "retriever": "dense", "strategy": "fixed", "chunk_size": 512,
        "overlap": 64, "model": "all-MiniLM-L6-v2"
      },
      "metrics": {
        "recall@1": 0.63, "recall@3": 0.80, "recall@5": 0.89, "recall@10": 0.93,
        "mrr": 0.81, "ndcg@5": 0.85
      },
      "timing": { "chunk_s": 0.4, "embed_s": 8.1, "search_s": 0.2, "chunks": 412 },
      "cases": [
        { "id": "q1", "tags": ["policy"], "expected": ["handbook/refunds.md"],
          "retrieved": ["handbook/refunds.md", "handbook/shipping.md"],
          "first_correct_rank": 1, "score": 1.0 }
      ]
    }
  ]
}
```

`cases` is what makes per-question comparison possible in week 6. Without it, `evalboard` could
only diff averages, which is exactly the problem `evalboard` exists to solve.

## 6. The example corpus

A tool like this is worthless without data anyone can run it on. Three options were considered:

| Option | Problem |
|--------|---------|
| Wikipedia articles | CC BY-SA. Share-alike is awkward to redistribute inside an MIT repo |
| Public documentation (Python, Kubernetes) | Licensing is workable but the text is long and uneven, and writing good ground truth over it is slow |
| **Write a small fictional company handbook** | **Chosen.** Fully owned, MIT, and I control the difficulty |

**Decision: write a fictional company handbook** (refunds, approvals, shipping, leave policy,
expenses, security, onboarding, and so on), plus hand-written questions. It came out at 17
documents and 31 questions. This mirrors
the most common real RAG use case — internal documents — and it lets me deliberately include the
cases that break naive retrieval:

- **Near-duplicate distractors.** "refunds over $500" and "refunds under $50" in different documents.
- **Multi-hop.** A question whose answer needs two documents.
- **Negation.** "Which expenses are *not* reimbursable?"
- **Vocabulary mismatch.** The question says "money back", the document says "refund". This is where
  BM25 fails and embeddings win, and the results should show that clearly.

**The README must say the corpus is synthetic**, and that the point of the tool is running it on
your own documents. Passing off a tuned demo set as a general benchmark would be dishonest and any
reviewer would spot it.

## 7. Build order

| Day | Work |
|-----|------|
| **Mon** | This plan. README pitch. Commit both before writing code |
| **Tue** | `corpus.py`, `chunking.py` with all three strategies, `metrics.py`. All pure, all tested as I go |
| **Wed** | `embedding.py` with the cache, the three retrievers, `sweep.py` with the shared-work layering |
| **Thu** | `results.py`, `report.py`, `cli.py`. Then the full test suite. Then CI |
| **Fri** | Write the corpus and the 30 labels. Run it for real. README with the actual output table and chart. Publish |

Writing the corpus is deliberately on Friday: it is content work, not code work, and it is the
part most likely to expand without limit. Putting it last caps it.

## 8. Test plan

The rule: **no test may download a model or touch the network.** A fake embedder that returns
deterministic vectors covers everything except the real model wrapper, which gets one test marked
slow and excluded from CI.

| Area | Must work | Must not break |
|------|-----------|----------------|
| `metrics` | recall@k, MRR, nDCG match values worked out by hand | empty results, no correct answer found, k larger than the result list |
| `chunking` | each strategy splits as documented; overlap repeats the right text; every chunk's text equals its own document slice; the chunks cover every character | empty document; document shorter than one chunk; overlap ≥ chunk size; unicode preserved; **no chunk sits wholly inside the previous one** |
| `corpus` | reads md/txt/html, stable document ids | unreadable file skipped not crashed on; empty folder gives a clear error |
| `embedding` | cache miss computes and stores; cache hit skips compute | cache key changes with the model name; corrupt cache file is rebuilt, not fatal |
| `retrievers` | dense, bm25 and hybrid each return top-k in ranked order | all three satisfy the same interface test |
| `sweep` | one config produces one run; N configs produce N runs | chunking is not repeated across retrievers that share it |
| `results` | written file validates against schema 1 and round-trips | reading a schema 2 file gives a clear version error |
| `cli` | exit codes; `run` then `report` reproduces the same table | missing corpus or labels file gives a useful message, not a traceback |

The metrics tests come first on Tuesday. If `recall@5` is wrong, every number the tool prints is
wrong, and nobody would be able to tell.

## 9. Risks

| Risk | Mitigation |
|------|------------|
| `sentence-transformers` pulls in torch, making install slow and heavy | Core install stays light (numpy and rich). Only the *embedding model* lives behind `pip install ragsweep[dense]`. BM25 is implemented here in ~60 lines and needs no extras, so the tool is usable immediately |
| The sweep is too slow to be pleasant | The layered sharing in section 4, plus the disk cache. Target: a 27-config sweep over 15 documents in under 60 seconds on a second run |
| The synthetic corpus looks self-serving | Say plainly in the README that it is synthetic and exists as a worked example, not a benchmark |
| Hand-written labels are subtly wrong | Keep it to 30 questions so each one can be checked properly. Quality over quantity |
| Scope creep into generation quality | It is in the non-goals. Hold the line |
| PyTorch has no Python 3.14 wheels, so `[dense]` cannot install there | Found on the day. The base install is unaffected, and the README says so plainly rather than letting someone discover it mid-install |

## 10. Definition of done

- [ ] Public repo with description and topics
- [ ] This plan, committed before the code
- [ ] README: the problem, a real output table, install, usage, config, non-goals, and an honest
      note about the synthetic corpus
- [ ] Example corpus and labelled questions in the repo
- [ ] Test suite covering the table in section 8, no network access
- [ ] CI green: tests and ruff
- [ ] `results.json` schema documented in the README, since week 6 depends on it
- [ ] At least 10 commits, one logical change each
- [ ] A second sweep run finishes in under a minute
