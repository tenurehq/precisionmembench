# PrecisionMemBench

PrecisionMemBench is a multi-dimensional retrieval benchmark for LLM memory systems. It measures four orthogonal properties that single-turn answer-quality benchmarks cannot detect:

- **Retrieval precision** - does the right belief surface, and only that belief, against a fixed seed corpus of 35 beliefs spanning two domain scopes, a supersession chain, and a secondary-user fixture
- **Noise isolation** - do beliefs introduced during off-topic drift turns contaminate retrieval on subsequent unrelated turns across a 10-turn session
- **Session-turn latency** - does retrieval latency degrade under session load relative to single-turn baselines
- **Belief mutability** - do beliefs updated mid-session surface immediately within the same session via the alias enrichment flywheel

These properties are independent. A system can pass on precision and fail on drift. A system can have clean single-turn latency and degrade 4x under session load. A system with no write-time mutation primitive cannot be scored on the fourth property at all, it is an architectural absence, not a performance difference.

Every case specifies not just what the memory system must return, but what it must not. Noise is a hard failure, not an invisible inference cost.

**89 cases** covering: alias resolution · scope disambiguation · supersession chain exclusion · fuzzy matching · cross-user isolation · budget eviction · ranking stability · session-level noise isolation under multi-turn topic drift

Paper: [arXiv](https://arxiv.org/abs/2605.11325) — Dataset: [HuggingFace](https://huggingface.co/datasets/tenurehq/precisionmembench) — Leaderboard: [HuggingFace Spaces](https://huggingface.co/spaces/tenurehq/precisionmembench)

## Results

### Retrieval Precision

| Provider       | Active passes | Total passes | Mean precision | Mean recall | Retrieval p50 (ms) | Ingestion total (s) |
| -------------- | ------------- | ------------ | -------------- | ----------- | ------------------ | ------------------- |
| `tenure`       | 43/43         | 77/77        | 1.00           | 1.00        | 9.77               | 1.00                |
| `supermemory`  | 17/17         | 44/77        | 0.43           | 0.55        | 819.48             | 0.00                |
| `agentmemory`  | 0/0           | 7/77         | 0.17           | 0.97        | 82.28              | 1.10                |
| `yourmemory`   | 0/0           | 21/77        | 0.17           | 0.88        | 313.39             | 16.40               |
| `atomicmemory` | 0/0           | 9/77         | 0.15           | 0.95        | 71.01              | 658.90              |
| `zep`          | 0/0           | 9/77         | 0.09           | 0.95        | 124.36             | 897.00              |
| `vector`       | 0/0           | 11/77        | 0.09           | 1.00        | 71.87              | —                   |
| `hindsight`    | 0/0           | 9/77         | 0.06           | 1.00        | 589.86             | 173.30              |
| `mem0`         | 0/0           | 9/77         | 0.06           | 0.99        | 64.94              | 111.30              |
| `a-mem`        | 0/0           | 9/77         | 0.06           | 0.99        | 13.80              | 178.80              |

**Active passes** are the only column that answers whether the memory system itself retrieved correctly. A system cannot accumulate active passes by returning everything or nothing.

Recall of 1.0 does not imply precision. Every comparison system returns the correct belief alongside many incorrect ones and scores perfectly on recall as a result. Mean precision of 0.05 to 0.09 means roughly 10 to 18 irrelevant beliefs are returned alongside each correct one.

### Pass type breakdown

Total pass counts require this breakdown to be interpreted correctly. All counts are over the 77 non-session cases.

| Provider       | Active retrieval | Structural | Trivially empty |
| -------------- | ---------------- | ---------- | --------------- |
| `tenure`       | 43               | 25         | 9               |
| `supermemory`  | 17               | 18         | 9               |
| `a-mem`        | 0                | 6          | 3               |
| `agentmemory`  | 0                | 5          | 2               |
| `atomicmemory` | 0                | 6          | 3               |
| `hindsight`    | 0                | 6          | 3               |
| `mem0`         | 0                | 6          | 3               |
| `vector`       | 0                | 8          | 3               |
| `yourmemory`   | 0                | 15         | 6               |
| `zep`          | 0                | 6          | 3               |

- **Active retrieval pass** - the case carries a `retrievalPrecision` assertion and it is satisfied. This is the only pass type that demonstrates verified retrieval capability.
- **Structural pass** - the case asserts scope isolation, supersession exclusion, or type routing without a precision assertion, and the structural property holds.
- **Trivially empty pass** - the expected `relevantBeliefs` tier is empty by case design (empty query, `maxBeliefs: 0`, budget set to exact pinned count). Any system returning an empty set passes by construction.

### Embedding model invariance

| Model                    | Precision | Recall | Passes | Mean (ms) | p95 (ms) |
| ------------------------ | --------- | ------ | ------ | --------- | -------- |
| nomic-embed-text (768)   | 0.09      | 1.0    | 11/77  | 43.36     | 85.21    |
| mxbai-embed-large (1024) | 0.09      | 1.0    | 11/77  | 96.48     | 257.24   |
| qwen3-8b (4096)          | 0.09      | 1.0    | 11/77  | 1130.95   | 2604.84  |

All 11 passes in every configuration are structural or trivially empty. Active retrieval passes are 0 across all three models.

### Session eval — noise isolation under multi-turn drift

The 12 session cases test three orthogonal properties: whether beliefs introduced during off-topic drift turns contaminate retrieval on subsequent unrelated turns, whether latency degrades under session load, and whether beliefs introduced mid-session surface within the same session window via the alias enrichment flywheel.

The drift score is the fraction of retrieved non-pinned beliefs originating from drift-turn topics; 0 is perfect isolation.

| Provider       | Turns passed | Pass rate | Mean drift | Noise isolation | Mean precision | Session p50 (ms) |
| -------------- | ------------ | --------- | ---------- | --------------- | -------------- | ---------------- |
| `tenure`       | 12/12        | 1.00      | 0.0000     | 1.00            | 1.0000         | 47.79            |
| `supermemory`  | 2/12         | 0.17      | 0.1667     | 0.17            | 0.6000         | 867.83           |
| `yourmemory`   | 1/12         | 0.08      | 0.7365     | 0.08            | 0.1965         | 430.49           |
| `agentmemory`  | 0/12         | 0.00      | 0.8087     | 0.00            | 0.1913         | 98.49            |
| `atomicmemory` | 0/12         | 0.00      | 0.8449     | 0.00            | 0.1551         | 355.08           |
| `zep`          | 0/12         | 0.00      | 0.8888     | 0.00            | 0.1112         | 418.13           |
| `vector`       | 0/12         | 0.00      | 0.9142     | 0.00            | 0.0858         | 256.75           |
| `a-mem`        | 0/12         | 0.00      | 0.9259     | 0.00            | 0.0741         | 25.66            |
| `hindsight`    | 0/12         | 0.00      | 0.9285     | 0.00            | 0.0715         | 1880.60          |
| `mem0`         | 0/12         | 0.00      | 0.9398     | 0.00            | 0.0602         | 377.93           |

‡ SuperMemory & yourmemory returned no results for these session cases. A drift score of 0.0 is recorded by construction; no beliefs were returned, so none could originate from drift topics. The correct belief also failed to surface, making this an empty-result failure rather than a genuine isolation pass.

## Pass taxonomy

Understanding the three pass types is required to interpret any results table.

**Active retrieval pass** — the case carries a `retrievalPrecision` assertion and it is satisfied. This is the only pass type that demonstrates verified retrieval capability. A system cannot accumulate active passes by returning everything or nothing.

**Structural pass** — the case asserts scope isolation, supersession exclusion, or type routing without a precision assertion, and the structural property holds.

**Trivially empty pass** — the expected `relevantBeliefs` tier is empty by case design (empty query, `maxBeliefs: 0`, budget set to exact pinned count). Any system returning an empty set passes by construction. `retrievalPrecision` is null for these cases.

Without this breakdown, aggregate pass counts do not distinguish verified retrieval from structural or empty-set passes.

## Case categories

The 89 cases cover the following categories. Session cases extend the corpus dynamically — beliefs are created and alias sets updated mid-session — validating that retrieval reflects the live store state rather than a snapshot.

| Category                         | Cases  |
| -------------------------------- | ------ |
| Alias resolution                 | 23     |
| Scope disambiguation             | 12     |
| Session-level noise isolation    | 12     |
| Fuzzy matching and prefix guards | 8      |
| Design boundary cases            | 6      |
| Type routing and open questions  | 6      |
| Budget eviction and capacity     | 5      |
| Relation expansion               | 4      |
| Persona prelude content          | 4      |
| Supersession chain exclusion     | 3      |
| Ranking stability                | 3      |
| Counter-signal retrieval         | 2      |
| Cross-user isolation             | 1      |
| Cold start behavior              | 1      |
| **Total**                        | **89** |

**Alias resolution** — whether variant surface forms (short-form, natural-language, multi-word) resolve to the correct belief.

**Scope disambiguation** — whether scope alone correctly discriminates between beliefs sharing an alias across different domain scopes.

**Supersession chain exclusion** — whether superseded beliefs are excluded at depth in a multi-hop chain. A query matching both a superseded and a superseding term must surface neither superseded belief; the active terminal belief surfaces via the pinned facts tier.

**Fuzzy matching and prefix guards** — whether the retrieval layer correctly handles transpositions and near-miss terms while blocking prefix mismatches that edit distance alone would permit. Both pass and fail behaviors are documented as intentional design properties.

**Counter-signal retrieval** — whether a query referencing a rejected or superseded term surfaces the active replacement belief via a counter-signal alias. Both cases carry an active retrieval precision assertion.

**Relation expansion** — whether relation-type beliefs correctly surface and expand their participants via a one-hop join, with participant type routing and scope filters applied during expansion.

**Session-level noise isolation** — whether beliefs introduced during off-topic drift turns contaminate retrieval on subsequent unrelated turns. The primary case is a 10-turn session with topic drift across 8 turns followed by an implicit return; per-turn assertions verify isolation at re-entry.

**Budget eviction and capacity** — whether the retrieval layer handles slot constraints correctly, including graceful empty returns, single-slot priority, and resistance to high-reinforcement flooding at the budget ceiling.

**Design boundary cases** — cases where both pass and fail behaviors are documented as intentional design properties.

**Type routing and open questions** — whether open questions are retrieved by a separate path that returns only pinned open questions for the active scope and are never returned by text search.

**Ranking stability** — whether retrieval results remain stable across equivalent queries without score-driven reordering artifacts.

**Cross-user isolation** — whether beliefs belonging to a second user are structurally excluded from a primary user's retrieval regardless of semantic proximity.

**Cold start behavior** — whether a new user with zero seeded beliefs returns a fully empty context without error.

**Persona prelude content** — whether the persona prelude generated from the accumulated belief state is injected correctly and reflects the live belief store.

## Metrics

Four metrics are recorded per case:

- **Retrieval precision and recall** — computed over the `relevantBeliefs` tier on cases where that tier carries an active assertion. Cases where this metric is structurally inapplicable record null and are excluded from aggregate computation.
- **Pinned coverage** — recorded on cases where the `pinnedFacts` tier is asserted.
- **Question precision and recall** — recorded on cases where the `openQuestions` tier is asserted.

A pass requires all asserted tiers to be simultaneously satisfied. A case with `retrievalPrecision: 1.0` that also carries an unmet `pinnedCoverage` assertion fails.

**Drift score** is reported for session cases: the fraction of retrieved non-pinned beliefs originating from drift-turn topics. 0 is perfect isolation.

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

### Tenure

Tenure's eval lives in the Tenure repo and runs directly against its `BeliefsReader` and `ContextBuilder` implementations. It is fully self-contained. The Atlas Local container starts and stops automatically. Reports land in `test-results/`. Results are re-produced on every pull request via CI.

```bash
git clone https://github.com/tenurehq/tenure.git
cd tenure
npm i
npm run test:eval
```

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
