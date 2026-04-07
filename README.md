# HockeyRAG

A production-grade RAG system over high-velocity structured and unstructured NHL data streams. Live game scores, player stats, news articles, and Reddit discussion are continuously ingested, embedded, and stored in a vector database. Queries are answered by retrieving live context and generating grounded responses via LLM — not from stale training data.

**[Architecture](#architecture)** · **[Retrieval pipeline](#retrieval-pipeline)** · **[Results](#results)**

---

## What this demonstrates

This is not an NHL chatbot. The hockey domain is an implementation detail. What this project demonstrates:

- **Hybrid retrieval** — dense semantic search + BM25 sparse search, fused via Reciprocal Rank Fusion, re-ranked by a cross-encoder. Each layer is ablated and measured.
- **Eval-driven development** — a RAGAS evaluation suite with a golden test set runs in CI on every push. Retrieval quality is a hard gate, not a manual check.
- **DPO retriever fine-tuning** — preference pairs collected from production query logs are used to fine-tune the bi-encoder via Direct Preference Optimization. Before/after eval results are documented.
- **Production patterns** — async ingestion pipeline, Redis Streams message queue, Langfuse observability, Docker Compose → Railway deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Sources                          │
│   NHL Stats API        Reddit (r/hockey)       NewsAPI       │
│   (no auth, live)      (public JSON)           (100/day)     │
└──────────┬─────────────────┬──────────────────┬─────────────┘
           │                 │                  │
           ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Ingestion Worker                          │
│   APScheduler → fetch → deduplicate → chunk → embed         │
│   SHA-256 dedup     512 tok / 64 overlap    bge-large-en     │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌─────────────────┐       ┌─────────────────────┐
   │   Qdrant        │       │   Redis              │
   │   Vector DB     │       │   Streams queue      │
   │   HNSW index    │       │   dedup hash set     │
   │   1024-dim      │       │                      │
   └────────┬────────┘       └──────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Retrieval Pipeline  [Phase 3]             │
│                                                              │
│   Query → embed (bge-large)                                  │
│         → dense search top-50 (Qdrant)                       │
│         → BM25 search top-50 (rank_bm25)                     │
│         → RRF fusion                                         │
│         → cross-encoder re-rank top-20 → top-5               │
│         → prompt construction                                │
│         → LLM generation (Ollama / Groq fallback)            │
│         → trace log (Langfuse)                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   FastAPI              │
              │   /query   /health     │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Streamlit            │
              │   Chat interface       │
              │   Thumbs up/down       │
              └────────────────────────┘
```

### Services

| Service | Role | Always on |
|---|---|---|
| `qdrant` | Vector storage, HNSW index, payload filtering | Yes |
| `redis` | Message queue (Redis Streams), dedup cache | Yes |
| `ingestion` | Scheduled polling, chunking, embedding, Qdrant upsert | Yes |
| `fastapi` | Retrieval pipeline, LLM generation, query API | Yes |
| `streamlit` | Chat UI, feedback collection | On demand (`--profile ui`) |

Streamlit is stateless and intentionally decoupled — stop and start it independently without affecting the pipeline.

---

## Tech stack

| Layer | Tool | Why this over alternatives |
|---|---|---|
| Dense embeddings | `BAAI/bge-large-en-v1.5` | Best CPU-viable dense model. Outperforms MiniLM on retrieval benchmarks while remaining feasible on Intel CPU. |
| Sparse embeddings | `rank_bm25` | Pure Python, zero infrastructure overhead. Handles exact player name and stat queries where dense search fails due to vocabulary mismatch. |
| Vector DB | Qdrant | Native hybrid search, payload filtering at query time, excellent Python client, open source. Milvus adds operational complexity without benefit at this scale. |
| Hybrid fusion | RRF (Reciprocal Rank Fusion) | Rank-based combination avoids the score normalisation problem — dense and sparse scores are not on the same scale. RRF fuses ranked lists directly. |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder attends over query+document jointly, significantly more accurate than bi-encoder scoring. Too slow for retrieval (O(n)), correct for re-ranking top-20. |
| LLM | Ollama (llama3 8B) + Groq fallback | Completely free. Ollama runs locally on CPU, Groq free tier as fallback when local inference is too slow. |
| Message queue | Redis Streams | Async decoupling between ingestion and processing. Simpler than Kafka with sufficient guarantees at this scale. |
| Eval | RAGAS | Purpose-built for RAG evaluation. Faithfulness, answer relevancy, and context recall cover the failure modes that matter. |
| Observability | Langfuse (self-hosted) | Per-query traces with per-step latency. Query logs become DPO training data — observability and training data collection are the same system. |
| CI/CD | GitHub Actions | Eval suite runs on every push. RAGAS score gate prevents silent retrieval regression. |
| Fine-tuning | HuggingFace TRL | DPO training on preference pairs collected from Langfuse logs. Colab free T4 for compute. |

---

## Retrieval pipeline

> ⚙️ *Planned for Phase 3. Architecture documented below.*

The pipeline is built in layers. Each layer is independently ablated — the numbers are in the [results section](#results).

### 1. Dense semantic search

Query is embedded with `bge-large-en-v1.5` (1024 dimensions, cosine similarity). Top-50 candidates retrieved from Qdrant via HNSW approximate nearest neighbour search.

Handles semantic queries well. Fails on exact player names and statistics due to vocabulary mismatch — "McDavid" and "Connor" may not be close in embedding space to "McDavid scored".

### 2. BM25 sparse search

BM25 index built over all ingested chunk texts. Scores based on term frequency and inverse document frequency. Top-50 candidates retrieved in parallel with dense search.

Handles exact keyword queries well. Fails on semantic queries — "who has been playing well lately" has no keyword match to retrieve against.

### 3. RRF fusion

Reciprocal Rank Fusion combines the two ranked lists:

```
score(d) = Σ 1 / (k + rank_i(d))    k = 60
```

Rank-based combination avoids the score normalisation problem. Dense scores (cosine similarity) and BM25 scores are not on the same scale — combining them directly requires arbitrary normalisation decisions. RRF uses only the rank position, which is scale-invariant.

### 4. Cross-encoder re-ranking

Top-20 fused candidates are scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike the bi-encoder used at retrieval time, the cross-encoder attends over the query and document jointly — it sees both at once and can model fine-grained interactions.

Too slow to run at retrieval time (requires a forward pass per candidate). Applied only to the top-20 shortlist, returning top-5.

### 5. Generation

Top-5 chunks are formatted into a prompt with the user question. Response generated by Ollama (llama3 8B local) with Groq API as fallback. Full trace logged to Langfuse including per-step latencies, retrieved chunks, and token counts.

---

## Data ingestion

### Sources and schedule

| Source | Endpoint | Schedule | Data |
|---|---|---|---|
| NHL Stats API | `api-web.nhle.com/v1` | Every 5 min | Scores, player stats, standings, play-by-play |
| Reddit | Public JSON endpoints (`r/hockey`, `r/nhl`) | Every 15 min | Posts, comments, trade discussion |
| NewsAPI | `newsapi.org` | Every 60 min | Hockey news articles (100 req/day budget) |

### Processing

Each raw document goes through:

1. **Deduplication** — SHA-256 hash of content + source checked against Redis set. Identical documents skipped before embedding.
2. **Chunking** — fixed-size chunks of 512 tokens with 64-token overlap. Metadata preserved on each chunk: source, date, entity tags (player names, team names).
3. **Embedding** — `bge-large-en-v1.5` via sentence-transformers. Batched at 16 chunks per batch for CPU efficiency. Embedding latency logged per batch.
4. **Upsert** — vectors stored in Qdrant with full metadata payload. Duplicate vector IDs overwritten.

---

## Evaluation

> ⚙️ *Planned for Phase 4.*

### Golden test set

50 Q&A pairs curated from actual ingested data. Each entry contains: question, ground truth answer, and the chunk IDs that contain the answer. Stored as JSON, version-controlled.

### RAGAS metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Does the answer contain only claims supported by the retrieved chunks? |
| Answer relevancy | Does the answer actually address the question that was asked? |
| Context recall | Did the retrieval step surface the chunks that contain the answer? |

### Ablation results

> ⚙️ *Results will be populated after Phase 4 is complete.*

| Pipeline config | Faithfulness | Answer relevancy | Context recall |
|---|---|---|---|
| Dense only (baseline) | — | — | — |
| Hybrid (dense + BM25 + RRF) | — | — | — |
| Hybrid + cross-encoder re-ranker | — | — | — |

### DPO results

> ⚙️ *Results will be populated after Phase 7 is complete.*

| | Faithfulness | Answer relevancy | Context recall |
|---|---|---|---|
| Pre-DPO | — | — | — |
| Post-DPO | — | — | — |
| Delta | — | — | — |

---

## Observability

> ⚙️ *Planned for Phase 5.*

Every query through FastAPI will be traced in Langfuse with:

- Per-step latency (embedding, Qdrant search, BM25 search, RRF, re-ranking, LLM)
- Top-5 retrieved chunks with source, score, and rank
- Prompt tokens, completion tokens, model used (Ollama vs Groq)
- User feedback score (thumbs up/down from Streamlit UI)

The feedback signal is the preference data source for DPO fine-tuning — observability and training data collection are the same system.

---

## DPO fine-tuning

> ⚙️ *Planned for Phase 7. Methodology documented below.*

### Motivation

Standard retrieval training optimises for relevance to a query in isolation. DPO fine-tuning uses preference pairs — (query, good chunk, bad chunk) triples collected from production query logs — to teach the bi-encoder what *this system's users* consider a good retrieval result.

### Data collection

Preference pairs exported from Langfuse: for each query where a user provided thumbs up/down feedback, extract the (question, positively-ranked chunk, negatively-ranked chunk) triple. Minimum 200 pairs required before training.

### Training

HuggingFace TRL `DPOTrainer` fine-tunes the `bge-large` bi-encoder. Contrastive objective: the score of (query, positive chunk) should exceed the score of (query, negative chunk) by a margin. Training runs on Google Colab free T4.

After training, all existing Qdrant documents are re-embedded with the fine-tuned weights (full re-index). Eval suite run before and after to measure improvement.

---

## CI/CD

> ⚙️ *Planned for Phase 6.*

### Pipeline

```
push to any branch
  └── unit tests (pytest tests/)

pull request
  └── unit tests
  └── integration tests (test containers: qdrant + redis)
  └── eval gate (10-question subset, RAGAS scores vs stored baseline)
         └── fail if any metric drops > 5%

merge to main
  └── all above
  └── Docker image build + push to ghcr.io (tagged with git SHA + latest)
```

### Pre-commit hooks

`black` · `isort` · `flake8` · `mypy` — run on every commit. Broken code cannot be committed.

---

## Performance

> ⚙️ *Benchmarks will be populated after Phase 8 is complete.*

| Metric | p50 | p95 |
|---|---|---|
| End-to-end query latency | — | — |
| Embedding throughput | — chunks/sec | — |
| Qdrant search latency | — | — |
| Re-ranker latency (top-20) | — | — |

---

## Engineering decisions

### Why Qdrant over Milvus or Pinecone

Milvus adds significant operational complexity (multiple internal services) without benefit at this scale. Pinecone is a managed service which contradicts the goal of demonstrating infrastructure ownership. Qdrant runs as a single Docker container, has a clean Python client, and natively supports hybrid search with payload filtering — the exact combination this pipeline needs.

### Why RRF over score normalisation for fusion

Dense similarity scores (cosine) and BM25 scores live on different scales with different distributions. Combining them directly requires normalising both to a common range, which introduces arbitrary decisions about normalisation method and weighting. RRF uses only rank positions, which are scale-invariant. A document ranked 3rd by dense and 7th by BM25 gets the same fusion treatment regardless of the raw scores.

### Why a cross-encoder re-ranker instead of retrieving top-5 directly

Bi-encoders embed query and document independently — they cannot model fine-grained query-document interactions. The cross-encoder sees both together, which is significantly more accurate but requires a full forward pass per candidate. Running it over all documents is infeasible. The hybrid retrieval pipeline narrows to top-20 candidates cheaply, then the cross-encoder re-ranks those 20 accurately. The combination gives near-cross-encoder quality at near-bi-encoder cost.

### Why Redis Streams over Kafka

Kafka adds substantial operational overhead — Zookeeper (or KRaft), broker configuration, topic management. Redis Streams provides the same consumer group semantics, at-least-once delivery guarantees, and backpressure handling within a single Redis container already in the stack. Correct choice at this scale.

### Why DPO over continued pretraining for retriever fine-tuning

Continued pretraining on domain text would improve the model's general hockey knowledge but would not teach it what constitutes a *good retrieval result for this specific pipeline*. DPO uses explicit preference signals — (query, good chunk, bad chunk) triples from real user feedback — to directly optimise the retrieval objective. It is also more stable than PPO-based RLHF: no reward model required, no policy rollouts, direct gradient signal from the preference pairs.

---

## What failed

> ⚙️ *This section will be populated as the project progresses. Documented failures are a feature, not a weakness — they show empirical iteration.*

---

## Running locally

### Requirements

- Docker Desktop (allocate at least 8GB RAM in settings)
- Ollama installed locally (`brew install ollama`)
- Python 3.11+

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/hockeyrag
cd hockeyrag

# Copy environment variables
cp .env.example .env
# Fill in: NEWS_API_KEY, GROQ_API_KEY

# Start core services (qdrant, redis, ingestion, fastapi)
make up

# Start with Streamlit UI
docker compose --profile ui up --build -d
```

### Services

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |
| Streamlit chat UI | http://localhost:8501 |

### Common commands

```bash
make up              # Start all containers (core + streamlit)
make down            # Stop all containers
make logs            # Tail all container logs
make logs-fastapi    # Tail FastAPI logs only
make logs-ingestion  # Tail ingestion worker logs
make shell-fastapi   # Shell into the FastAPI container
make shell-ingestion # Shell into the ingestion worker
make shell-redis     # Shell into Redis

# Run tests
make test
pytest tests/ -v
```

---

## Repository structure

```
hockeyrag/
├── services/
│   ├── ingestion/          # Data clients, chunking, embedding, Qdrant upsert
│   │   ├── clients/        # NHL API, Reddit, NewsAPI clients
│   │   ├── chunker.py      # Fixed-size chunking with overlap
│   │   ├── dedup.py        # SHA-256 deduplication via Redis
│   │   ├── embedder.py     # bge-large wrapper, batch embedding
│   │   ├── qdrant_store.py # Qdrant collection management + upsert
│   │   ├── worker.py       # APScheduler jobs, Redis Stream consumer
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── fastapi/            # Query API (retrieval pipeline — Phase 3)
│   │   ├── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── streamlit/          # Chat UI, feedback collection
│       ├── app.py
│       ├── Dockerfile
│       └── requirements.txt
├── tests/
│   ├── test_chunker.py
│   ├── test_dedup.py
│   ├── test_integration_ingestion.py
│   ├── test_container_networking.py
│   └── test_networking.py
├── docker-compose.yml
├── Makefile
└── .env.example
```

---

## Roadmap

- [x] Project specification and architecture
- [x] Phase 1 — Infrastructure skeleton (all 5 Docker services)
- [x] Phase 2 — Data ingestion pipeline (scheduler, clients, dedup, chunker, embedder, Qdrant upsert)
- [ ] Phase 3 — Retrieval pipeline (dense + BM25 + RRF + cross-encoder)
- [ ] Phase 4 — Eval suite + ablation table
- [ ] Phase 5 — Langfuse observability
- [ ] Phase 6 — CI/CD with eval gate
- [ ] Phase 7 — DPO fine-tuning
- [ ] Phase 8 — Railway deployment + documentation

---

## Licence

MIT
