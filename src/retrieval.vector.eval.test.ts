/**
 * retrieval.vector.eval.test.ts
 *
 * Vector-search equivalent of retrieval.external.eval.test.ts.
 *
 * Runs the retrieval.cases.json suite against $vectorSearch (mxbai-embed-large
 * via Ollama) instead of BM25, producing a parallel report at:
 *   test-results/retrieval-report-vector.json
 *
 * Prerequisites:
 *   1. Run `npx tsx src/__fixtures__/embed-seed.ts` once to generate
 *      beliefs.seed.embedded.json (commit this file, don't regenerate at test time).
 *   2. Ollama must be running at OLLAMA_URL (default: http://localhost:11434)
 *      only for embed-seed.ts — not at test runtime.
 *
 * The vector index is created in test.before alongside the BM25 index.
 * Atlas Local supports both $search and $vectorSearch indexes simultaneously.
 */

import test from "ava";
import { execSync, spawnSync } from "node:child_process";
import { MongoClient, type Collection, type Db } from "mongodb";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import {
  VECTOR_INDEX_NAME,
  VECTOR_DIMENSIONS,
  ollamaEmbed,
} from "./utils/beliefsReaderVector.js";
import type { ContextBudget, PersonaLookup } from "./adapters/baseAdapter.js";
import type { Belief } from "./types/belief.js";
import { VectorAdapter } from "./adapters/vectorAdapter.js";
import { fileURLToPath } from "node:url";
import { buildReportPayload } from "./utils/buildRetrievalReport.js";

interface PinnedFactsExpect {
  mustInclude?: string[];
  mustExclude?: string[];
}

interface PreludeExpect {
  nonEmpty?: boolean;
  isNull?: boolean;
  contains?: string[];
  mustNotContain?: string[];
}

interface BeliefsExpect {
  mustInclude?: string[];
  mustExclude?: string[];
  shouldInclude?: string[];
  shouldOnlyInclude?: string[];
  orderedBefore?: [string, string][];
  maxCount?: number;
  minCount?: number;
}

interface QuestionsExpect {
  mustInclude?: string[];
  mustExclude?: string[];
}

interface RetrievalCase {
  caseId: string;
  category: string;
  description: string;
  userId?: string;
  scope: string[];
  query: string;
  budget?: Partial<ContextBudget>;
  expect: {
    personaPrelude?: PreludeExpect;
    scopePrelude?: PreludeExpect;
    pinnedFacts?: PinnedFactsExpect;
    relevantBeliefs?: BeliefsExpect;
    openQuestions?: QuestionsExpect;
  };
  notes?: string;
}

interface ReportEntry {
  caseId: string;
  category: string;
  description: string;
  pinnedBeliefs: string[];
  relevantBeliefs: string[];
  retrievedQuestions: string[];
  retrievalPrecision: number | null;
  retrievalRecall: number | null;
  pinnedCoverage: number | null;
  questionPrecision: number | null;
  questionRecall: number | null;
  passed: boolean;
  failures: string[];
  retrievalLatencyMs: number;
}

const ATLAS_IMAGE =
  "mongodb/mongodb-atlas-local@sha256:09dc8bf638086072db0a9ca93e17754a894518fca681a5d16b8c9369bc1a1987";
const CONTAINER_NAME = "memory-eval-atlas-vector";
const HOST_PORT = 27019;
const MONGO_URI = `mongodb://localhost:${HOST_PORT}/?directConnection=true`;

const READY_TIMEOUT_MS = 90_000;
const READY_POLL_MS = 50;

const USER_ID = "test-user";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

const FIXTURE_BELIEFS = resolve(
  __dirname,
  "../fixtures/beliefs.seed.embedded.json",
);
const FIXTURE_CASES = resolve(__dirname, "../fixtures/retrieval.cases.json");
const REPORT_DIR = resolve(__dirname, "../test-results");
const REPORT_PATH = resolve(REPORT_DIR, "retrieval-report-vector.json");

const DATE_FIELDS = [
  "created_at",
  "updated_at",
  "last_reinforced_at",
  "resolved_at",
] as const;

const EVAL_PERSONA: PersonaLookup = {
  get: async (userId: string) =>
    userId === USER_ID
      ? {
          universal:
            "You prefer direct answers without preamble. You push back when plans have problems rather than defaulting to agreement. You edit AI output; you do not let AI edit your prose.",
          per_scope: {
            "domain:code":
              "You work in TypeScript with strict mode, Fastify for HTTP, and MongoDB with the raw driver — never an ORM. You prefer composition over inheritance and Go-style explicit error returns.",
            "domain:writing":
              "You write close third-person, present tense, set in 1970s Lisbon. No omniscient asides, no reconciliation arcs.",
          },
        }
      : null,
};

function containerExists(): boolean {
  const result = spawnSync("docker", [
    "ps",
    "-a",
    "--filter",
    `name=^${CONTAINER_NAME}$`,
    "--format",
    "{{.Names}}",
  ]);
  return result.stdout.toString().trim() === CONTAINER_NAME;
}

function containerRunning(): boolean {
  const result = spawnSync("docker", [
    "ps",
    "--filter",
    `name=^${CONTAINER_NAME}$`,
    "--format",
    "{{.Names}}",
  ]);
  return result.stdout.toString().trim() === CONTAINER_NAME;
}

function startContainer(): void {
  if (containerRunning()) return;
  if (containerExists()) {
    execSync(`docker start ${CONTAINER_NAME}`, { stdio: "pipe" });
    return;
  }
  execSync(
    [
      "docker run -d",
      `--name ${CONTAINER_NAME}`,
      `-p ${HOST_PORT}:27017`,
      ATLAS_IMAGE,
    ].join(" "),
    { stdio: "pipe" },
  );
}

function stopContainer(): void {
  if (containerExists()) {
    execSync(`docker rm -f ${CONTAINER_NAME}`, { stdio: "pipe" });
    try {
      execSync(`docker volume rm ${CONTAINER_NAME}-data`, { stdio: "pipe" });
    } catch {
      // volume may not exist
    }
  }
}

async function waitForMongo(): Promise<void> {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      const c = new MongoClient(MONGO_URI, {
        serverSelectionTimeoutMS: READY_POLL_MS,
      });
      await c.connect();
      await c.db("admin").command({ ping: 1 });
      await c.close();
      return;
    } catch {
      await new Promise((r) => setTimeout(r, READY_POLL_MS));
    }
  }
  throw new Error(
    `Atlas Local container did not become ready within ${READY_TIMEOUT_MS}ms`,
  );
}

function coerceBelief(
  raw: Record<string, unknown>,
): Belief & { embedding?: number[] } {
  const out: Record<string, unknown> = { ...raw };
  for (const f of DATE_FIELDS) {
    const v = out[f];
    if (typeof v === "string") out[f] = new Date(v);
  }
  const prov = out.provenance as Record<string, unknown> | undefined;
  if (prov?.extracted_at && typeof prov.extracted_at === "string") {
    prov.extracted_at = new Date(prov.extracted_at);
  }
  return out as unknown as Belief & { embedding?: number[] };
}

async function retryUntilReady<T>(
  fn: () => Promise<T>,
  label: string,
  timeoutMs = READY_TIMEOUT_MS,
): Promise<T> {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown;
  while (Date.now() < deadline) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      await new Promise((r) => setTimeout(r, READY_POLL_MS));
    }
  }
  throw new Error(
    `${label} did not succeed within ${timeoutMs}ms — last error: ${
      lastErr instanceof Error ? lastErr.message : String(lastErr)
    }`,
  );
}

const embeddingCache = new Map<string, number[]>();

async function cachedEmbed(text: string): Promise<number[]> {
  const cached = embeddingCache.get(text);
  if (cached) return cached;
  const vec = await ollamaEmbed(text);
  embeddingCache.set(text, vec);
  return vec;
}

let client: MongoClient;
let db: Db;
let col: Collection<Belief & { embedding?: number[] }>;
let adapter: VectorAdapter;
let pinnedInSeed: Set<string>;
const report: ReportEntry[] = [];

test.before(async () => {
  startContainer();
  await waitForMongo();

  client = new MongoClient(MONGO_URI);
  await client.connect();

  db = client.db("retrieval_eval_vector_isolated");
  col = db.collection("beliefs");

  await col.drop().catch(() => {});
  await db.createCollection("beliefs");

  await retryUntilReady(
    () =>
      col.createSearchIndex({
        name: "beliefs_search",
        definition: {
          analyzer: "lucene.standard",
          analyzers: [
            {
              name: "aliases_light",
              tokenizer: { type: "standard" },
              tokenFilters: [
                { type: "lowercase" },
                { type: "englishPossessive" },
                { type: "kStemming" },
              ],
            },
          ],
          mappings: {
            dynamic: false,
            fields: {
              user_id: { type: "token" },
              canonical_name: { type: "string", analyzer: "lucene.english" },
              aliases: { type: "string", analyzer: "aliases_light" },
              content: { type: "string", analyzer: "lucene.english" },
              superseded_by: { type: "token" },
              resolved_at: { type: "date" },
              type: { type: "token" },
              subtype: { type: "token" },
              scope: { type: "token" },
            },
          },
        },
      }),
    "createSearchIndex(bm25)",
    60_000,
  );

  await retryUntilReady(
    () =>
      col.createSearchIndex({
        name: VECTOR_INDEX_NAME,
        type: "vectorSearch",
        definition: {
          fields: [
            {
              type: "vector",
              path: "embedding",
              numDimensions: VECTOR_DIMENSIONS,
              similarity: "cosine",
            },
            { type: "filter", path: "user_id" },
            { type: "filter", path: "superseded_by" },
            { type: "filter", path: "resolved_at" },
          ],
        },
      } as Parameters<typeof col.createSearchIndex>[0]),
    "createSearchIndex(vector)",
    60_000,
  );

  await retryUntilReady(
    async () => {
      const indexes = (await col.listSearchIndexes().toArray()) as Array<
        Record<string, unknown>
      >;
      const bm25 = indexes.find((i) => i.name === "beliefs_search");
      const vec = indexes.find((i) => i.name === VECTOR_INDEX_NAME);
      if (bm25?.status !== "READY")
        throw new Error(`bm25 status: ${bm25?.status}`);
      if (vec?.status !== "READY")
        throw new Error(`vector status: ${vec?.status}`);
    },
    "waitForSearchIndexes",
    60_000,
  );

  const raw = JSON.parse(readFileSync(FIXTURE_BELIEFS, "utf8")) as Record<
    string,
    unknown
  >[];

  const first = raw[0] as Record<string, unknown>;
  if (!Array.isArray(first?.embedding) || first.embedding.length === 0) {
    throw new Error(
      "beliefs.seed.embedded.json is missing embeddings. " +
        "Run `npx tsx src/__fixtures__/embed-seed.ts` first.",
    );
  }

  await col.insertMany(raw.map(coerceBelief));

  adapter = new VectorAdapter(col);

  await retryUntilReady(
    async () => {
      const probe = await cachedEmbed("typescript");
      const results = await col
        .aggregate([
          {
            $vectorSearch: {
              index: VECTOR_INDEX_NAME,
              path: "embedding",
              queryVector: probe,
              numCandidates: 10,
              limit: 1,
            },
          },
        ])
        .toArray();
      if (results.length === 0) throw new Error("vector index not synced yet");
    },
    "waitForVectorSync",
    READY_TIMEOUT_MS,
  );

  await retryUntilReady(
    async () => {
      const results = await col
        .aggregate([
          {
            $search: {
              index: "beliefs_search",
              text: { query: "TypeScript", path: "aliases" },
            },
          },
          { $limit: 1 },
        ])
        .toArray();
      if (results.length === 0)
        throw new Error("Atlas Search index not synced");
    },
    "waitForBM25Sync",
    READY_TIMEOUT_MS,
  );

  const pinnedDocs = await col
    .find({ user_id: USER_ID, pinned: true }, { projection: { _id: 1 } })
    .toArray();
  pinnedInSeed = new Set(pinnedDocs.map((d) => d._id as string));
});

test.after.always(async () => {
  mkdirSync(REPORT_DIR, { recursive: true });
  writeFileSync(
    REPORT_PATH,
    JSON.stringify(
      buildReportPayload(
        {
          provider: "vector",
          entries: report,
          caseCount: cases.length,
        },
        report,
      ),
      null,
      2,
    ),
  );
  await client?.close();
  stopContainer();
});

const cases = JSON.parse(
  readFileSync(FIXTURE_CASES, "utf8"),
) as RetrievalCase[];

for (const tc of cases) {
  test.serial(`[vector] ${tc.caseId}: ${tc.description}`, async (t) => {
    const buildStart = performance.now();
    const ctx = await adapter.buildContext(
      tc.userId ?? USER_ID,
      tc.scope,
      tc.query,
      EVAL_PERSONA,
      tc.budget,
    );
    const buildEnd = performance.now();

    const pinned = JSON.parse(ctx.pinnedFactsJson) as Array<
      Record<string, unknown>
    >;
    const relevant = JSON.parse(ctx.relevantBeliefsJson) as Array<
      Record<string, unknown>
    >;
    const questions = JSON.parse(ctx.openQuestionsJson) as Array<
      Record<string, unknown>
    >;

    const pinnedIds = new Set(pinned.map((b) => b.id as string));
    const relevantIds = new Set(relevant.map((b) => b.id as string));
    const questionIds = new Set(questions.map((q) => q.id as string));
    const unionIds = new Set<string>([...pinnedIds, ...relevantIds]);

    const failures: string[] = [];
    const check = (cond: boolean, msg: string): void => {
      if (!cond) failures.push(msg);
    };

    const rb = tc.expect.relevantBeliefs ?? {};

    for (const id of rb.mustInclude ?? []) {
      check(unionIds.has(id), `missing expected belief: ${id}`);
    }
    for (const id of rb.mustExclude ?? []) {
      check(!unionIds.has(id), `forbidden belief surfaced: ${id}`);
    }
    for (const id of rb.shouldInclude ?? []) {
      check(unionIds.has(id), `expected belief missing (shouldInclude): ${id}`);
    }

    const onlyExpected = rb.shouldOnlyInclude;
    if (onlyExpected) {
      const expectedSet = new Set(onlyExpected);
      for (const id of relevantIds) {
        check(
          expectedSet.has(id),
          `unexpected belief in relevantBeliefs: ${id}`,
        );
      }
      for (const id of expectedSet) {
        check(relevantIds.has(id), `missing expected belief: ${id}`);
      }
    }

    if (rb.maxCount != null) {
      check(
        relevantIds.size <= rb.maxCount,
        `relevantBeliefs count ${relevantIds.size} > maxCount ${rb.maxCount}`,
      );
    }
    if (rb.minCount != null) {
      check(
        relevantIds.size >= rb.minCount,
        `relevantBeliefs count ${relevantIds.size} < minCount ${rb.minCount}`,
      );
    }

    const relevantArray = relevant.map((r) => r.id as string);
    for (const [a, b] of rb.orderedBefore ?? []) {
      const idxA = relevantArray.indexOf(a);
      const idxB = relevantArray.indexOf(b);
      check(idxA !== -1, `orderedBefore: ${a} not in relevantBeliefs`);
      check(idxB !== -1, `orderedBefore: ${b} not in relevantBeliefs`);
      if (idxA !== -1 && idxB !== -1) {
        check(
          idxA < idxB,
          `ranking: ${a} (idx ${idxA}) should precede ${b} (idx ${idxB})`,
        );
      }
    }

    const pf = tc.expect.pinnedFacts ?? {};

    for (const id of pf.mustInclude ?? []) {
      check(pinnedIds.has(id), `missing pinned belief: ${id}`);
    }
    for (const id of pf.mustExclude ?? []) {
      check(!pinnedIds.has(id), `forbidden belief in pinnedFacts: ${id}`);
    }

    const oq = tc.expect.openQuestions ?? {};

    for (const id of oq.mustInclude ?? []) {
      check(questionIds.has(id), `missing expected question: ${id}`);
    }
    for (const id of oq.mustExclude ?? []) {
      check(!questionIds.has(id), `forbidden question surfaced: ${id}`);
    }

    const pp = tc.expect.personaPrelude;
    if (pp?.nonEmpty)
      check(ctx.personaPrelude.length > 0, "personaPrelude empty");
    if (pp?.isNull)
      check(ctx.personaPrelude === "", "personaPrelude not empty");
    for (const s of pp?.contains ?? []) {
      check(ctx.personaPrelude.includes(s), `personaPrelude missing "${s}"`);
    }
    for (const s of pp?.mustNotContain ?? []) {
      check(!ctx.personaPrelude.includes(s), `personaPrelude contains "${s}"`);
    }

    const mustIncludeQuestions = new Set(oq.mustInclude ?? []);

    const expectedRelevant = rb.shouldOnlyInclude
      ? new Set(rb.shouldOnlyInclude)
      : new Set(
          [...(rb.mustInclude ?? [])].filter((id) => !pinnedInSeed.has(id)),
        );

    const retrievalHits = [...expectedRelevant].filter((id) =>
      relevantIds.has(id),
    ).length;

    const retrievalPrecision =
      relevantIds.size === 0 && expectedRelevant.size === 0
        ? null
        : relevantIds.size === 0
          ? null
          : retrievalHits / relevantIds.size;

    const retrievalRecall =
      expectedRelevant.size === 0
        ? null
        : retrievalHits / expectedRelevant.size;

    const expectedPinnedFromFacts = new Set(pf.mustInclude ?? []);
    const pinnedHits = [...expectedPinnedFromFacts].filter((id) =>
      pinnedIds.has(id),
    ).length;
    const pinnedCoverage =
      expectedPinnedFromFacts.size === 0
        ? null
        : pinnedHits / expectedPinnedFromFacts.size;

    const questionHits = [...mustIncludeQuestions].filter((id) =>
      questionIds.has(id),
    ).length;
    const questionPrecision =
      questionIds.size === 0 || mustIncludeQuestions.size === 0
        ? null
        : questionHits / questionIds.size;
    const questionRecall =
      mustIncludeQuestions.size === 0
        ? null
        : questionHits / mustIncludeQuestions.size;

    report.push({
      caseId: tc.caseId,
      category: tc.category,
      description: tc.description,
      pinnedBeliefs: [...pinnedIds],
      relevantBeliefs: [...relevantIds],
      retrievedQuestions: [...questionIds],
      retrievalPrecision,
      retrievalRecall,
      pinnedCoverage,
      questionPrecision,
      questionRecall,
      passed: failures.length === 0,
      failures,
      retrievalLatencyMs: Math.round((buildEnd - buildStart) * 100) / 100,
    });

    t.is(failures.length, 0, failures.join(" | "));
  });
}
