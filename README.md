# PrecisionMemBench

Every major benchmark for LLM memory systems measures whether a model answered correctly, not whether the memory system retrieved correctly. These are not the same question. A system that returns its entire belief store achieves recall of 1.0 and, at the corpus sizes current benchmarks use, a capable generative model can find the right answer in the noise and score well on F1 or LLM-as-a-Judge. Mean retrieval precision of 0.05 passes. The failure is invisible.

PrecisionMemBench is the first benchmark that measures retrieval precision independently of the generative model downstream. Cases carry `mustExclude` assertions and `shouldOnlyInclude` constraints. Noise is a hard failure, not an invisible inference cost.

**89 cases** covering: alias resolution · scope disambiguation · supersession chain exclusion · fuzzy matching · cross-user isolation · budget eviction · ranking stability · session-level noise isolation under multi-turn topic drift

Paper: [arXiv](https://arxiv.org/abs/2605.11325) — Dataset: [HuggingFace](https://huggingface.co/datasets/tenurehq/precisionmembench) — Leaderboard: [HuggingFace Spaces](https://huggingface.co/spaces/tenurehq/precisionmembench)

## Results

| System         | Active passes | Total passes | Mean precision | Mean recall | Retrieval p50 (ms) | Ingestion (s) |
| -------------- | ------------- | ------------ | -------------- | ----------- | ------------------ | ------------- |
| Tenure         | 48/48         | 77/77        | 1.00           | 1.00        | 8.82               | 0.98          |
| Vector (mxbai) | 0/48          | 10/77        | 0.09           | 1.00        | 64.18              | —             |
| Mem0           | 0/48          | 9/77         | 0.08           | 1.00        | 61.94              | 103.8         |
| Zep            | 0/48          | 8/77         | 0.06           | 0.81        | 99.10              | 452.3         |
| Hindsight      | 0/48          | 8/77         | 0.04           | 0.27        | 516.21             | 102.9         |

**Active passes** require a `retrievalPrecision` assertion to be satisfied — the only pass type that demonstrates verified retrieval capability. All comparison system passes are trivially empty or budget-forced. Zero comparison systems achieve a single active retrieval pass.

The "recall 1.0" pattern for Vector, Mem0, and Hindsight is not a strength: it means those systems return the full belief corpus on every query. At 12 beliefs, a capable model can sort the noise. At thousands of beliefs, this architecture fails structurally.

The live leaderboard is maintained on [HuggingFace Spaces](https://huggingface.co/spaces/tenurehq/precisionmembench).

## Pass taxonomy

Understanding the three pass types is required to interpret any results table.

**Active retrieval pass** — the case carries a `retrievalPrecision` assertion and it is satisfied. This is the only pass type that demonstrates verified retrieval capability. A system cannot accumulate active passes by returning everything or nothing.

**Structural pass** — the case asserts scope isolation, supersession exclusion, or type routing without a precision assertion, and the structural property holds.

**Trivially empty pass** — the expected `relevantBeliefs` tier is empty by case design (empty query, `maxBeliefs: 0`, budget set to exact pinned count). Any system returning an empty set passes by construction. `retrievalPrecision` is null for these cases.

Aggregate pass counts without this breakdown are misleading. Every comparison system's passes are either structural or trivially empty.

## Baseline reports

Pre-run reports for all reference systems are committed at `test-results/baseline/`:

```
test-results/baseline/
  retrieval-report.json
  retrieval-report-vector.json
  retrieval-report-mem0.json
  retrieval-report-zep.json
  retrieval-report-hindsight.json
```

Each report contains per-case results including `passed`, `failures`, `retrievalPrecision`, `retrievalRecall`, and `retrievalLatencyMs`, plus aggregate `p50`/`p95` latency and mean precision/recall at the top level.

When you run against your own provider, compare your output in `test-results/` directly against these files.

## Running the benchmark

### Prerequisites

- Node.js 20+
- Docker (for the vector baseline and provider stacks)
- An Ollama instance for the vector baseline only

### 1. Install dependencies

```bash
npm install
```

### 2. Run against a comparison provider

Start the provider's stack, then:

```bash
MEMORY_PROVIDER=mem0 npx ava retrieval.external.eval.test.ts
MEMORY_PROVIDER=mem0 npx ava session-retrieval.external.eval.test.ts
```

Reports land in `test-results/`. Valid values: `mem0`, `zep`, `hindsight`

### 3. Run the vector baseline

The vector eval manages its own MongoDB Atlas Local container. Docker must be running but you do not set anything up manually.

```bash
# One-time: generate embeddings and commit the result
OLLAMA_URL=http://localhost:11434 npx tsx embed-seed.ts

# Run the eval
npx ava retrieval.vector.eval.test.ts
npx ava session-retrieval.vector.eval.test.ts
```

The Atlas Local container starts and stops automatically per run. Ports `27019` (single-turn) and `27021` (session) are used.

### 4. Export results to HuggingFace format

```bash
python export_to_hf.py
# Output: hf_export/leaderboard.json + hf_export/README.md
```

## Adding your provider

### 1. Write a wrapper

Expose a FastAPI service with three endpoints. See `wrappers/mem0_service.py` for the full contract.

**`POST /add`**

```json
{
  "text": "redis_cache Redis",
  "user_id": "test-user",
  "metadata": { "beliefId": "b-redis-code" }
}
```

**`POST /search`**

```json
{ "query": "Redis eviction policy", "user_id": "test-user", "limit": 20 }
```

Returns: `{ "results": [ { "id": "...", "memory": "...", "metadata": { "beliefId": "..." } } ] }`

**`DELETE /reset`**
Clears all memories for all users. Called once before seeding.

The `beliefId` in metadata is how the harness maps provider results back to the benchmark's belief schema. If your provider cannot round-trip arbitrary metadata, implement a custom `resolveBeliefId` in the adapter.

### 2. Register the provider

Add one entry to `providers.config.json`:

```json
"myprovider": {
  "envVar": "MYPROVIDER_URL",
  "defaultUrl": "http://localhost:8082",
  "seedDelayMs": 1000,
  "beliefToText": "canonical_name_aliases"
}
```

### 3. Run

```bash
MEMORY_PROVIDER=myprovider npx ava retrieval.external.eval.test.ts
MEMORY_PROVIDER=myprovider npx ava session-retrieval.external.eval.test.ts
```

The eval files themselves never need to change.

## Submitting results to the leaderboard

1. Fork this repo.
2. Run the full eval suite against your provider (both `retrieval.external.eval.test.ts` and `session-retrieval.external.eval.test.ts`).
3. Commit your report files from `test-results/` to `test-results/baseline/` using the naming convention `retrieval-report-{provider}.json`.
4. Open a PR. Include the provider name, Docker image digest (if applicable), and any relevant configuration notes in the description.

Results from merged PRs are reflected on the [live leaderboard](https://huggingface.co/spaces/tenurehq/precisionmembench).

## Provider wrappers

Each comparison provider is wrapped with a thin FastAPI service that normalises the `/add` / `/search` / `/reset` contract. Wrappers are in `wrappers/`.

### Mem0

```bash
cd wrappers && docker compose up
```

Requires `MEM0_URL`, an Ollama instance for embeddings, and a running Qdrant container (included in `docker-compose.yml`).

### Hindsight

```bash
cd wrappers
HINDSIGHT_URL=http://localhost:8888 python hindsight_wrapper.py
```

### Zep

```bash
cd wrappers && docker compose up
```

## Citation

```bibtex
@article{flynt2026precisionmembench,
  title   = {Structured Belief State and the First Precision-Aware Benchmark
             for LLM Memory Retrieval},
  author  = {Flynt, Jeffrey},
  year    = {2026}
}
```
