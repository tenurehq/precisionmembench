/**
 * session-retrieval.vector.eval.test.ts
 *
 * Vector-search equivalent of session-retrieval.external.eval.test.ts.
 *
 * Runs the same session cases against $vectorSearch (mxbai-embed-large via Ollama)
 * instead of BM25, producing a parallel report at:
 *   test-results/session-retrieval-report-vector.json
 *
 * Prerequisites:
 *   1. Run `npx tsx src/__fixtures__/embed-seed.ts` once to generate
 *      beliefs.seed.embedded.json (commit this file, don't regenerate at test time).
 *   2. Ollama must be running at OLLAMA_URL only for embed-seed.ts — not at test runtime.
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
  beliefEmbedText,
} from "./utils/beliefsReaderVector.js";
import type { PersonaLookup } from "./adapters/baseAdapter.js";
import type { Belief } from "./types/belief.js";
import { VectorAdapter } from "./adapters/vectorAdapter.js";
import {
  buildReportPayload,
  type ReportSummaryOptions,
} from "./utils/buildRetrievalReport.js";
import { fileURLToPath } from "node:url";

interface BeliefsExpect {
  mustInclude?: string[];
  mustExclude?: string[];
  shouldOnlyInclude?: string[];
}

interface PinnedFactsExpect {
  mustInclude?: string[];
  mustExclude?: string[];
}

interface QuestionsExpect {
  mustInclude?: string[];
  mustExclude?: string[];
}

interface NoiseCheck {
  mustNotSurface: string[];
}

interface SessionTurnExpect {
  relevantBeliefs?: BeliefsExpect;
  pinnedFacts?: PinnedFactsExpect;
  openQuestions?: QuestionsExpect;
  noiseCheck?: NoiseCheck;
}

interface SessionTurn {
  turnIndex: number;
  label: "establishes_topic" | "drift" | "implicit_continuation" | "re_entry";
  scope: string[];
  userMessage: string;
  assistantMessage: string;
  createBeliefAtTurn?: Record<string, unknown>;
  updateBeliefAtTurn?: {
    beliefId: string;
    addAliases?: string[];
    setContent?: string;
    setCanonicalName?: string;
    _note?: string;
  };
  topics: string[];
  expect: SessionTurnExpect;
}

interface SessionEvalCase {
  caseId: string;
  description: string;
  turns: SessionTurn[];
  notes?: string;
}

interface SessionReportEntry {
  caseId: string;
  category: string;
  turnIndex: number;
  label: string;
  retrievedBeliefIds: string[];
  pinnedBeliefIds: string[];
  noiseBeliefIds: string[];
  driftScore: number;
  passed: boolean;
  failures: string[];
  retrievalLatencyMs: number;
  retrievalPrecision: number | null;
  retrievalRecall: number | null;
}

const __dirname = fileURLToPath(new URL(".", import.meta.url));

const ATLAS_IMAGE =
  "mongodb/mongodb-atlas-local@sha256:09dc8bf638086072db0a9ca93e17754a894518fca681a5d16b8c9369bc1a1987";
const CONTAINER_NAME = "memory-eval-session-atlas-vector";
const HOST_PORT = 27021;
const MONGO_URI = `mongodb://localhost:${HOST_PORT}/?directConnection=true`;

const READY_TIMEOUT_MS = 90_000;
const READY_POLL_MS = 50;

const USER_ID = "test-user";
const FIXTURE_BELIEFS = resolve(
  __dirname,
  "../fixtures/beliefs.seed.embedded.json",
);
const FIXTURE_SESSION_CASES = resolve(
  __dirname,
  "../fixtures/session-retrieval.cases.json",
);
const REPORT_DIR = resolve(__dirname, "../test-results");
const REPORT_PATH = resolve(REPORT_DIR, "session-retrieval-report-vector.json");

const RESEED = process.env.RESEED === "true";

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
let beliefsCol: Collection<Belief & { embedding?: number[] }>;
let adapter: VectorAdapter;
const sessionReport: SessionReportEntry[] = [];

test.before(async () => {
  startContainer();
  await waitForMongo();

  client = new MongoClient(MONGO_URI);
  await client.connect();

  db = client.db("session_eval_vector_isolated");
  beliefsCol = db.collection("beliefs");

  await beliefsCol.drop().catch(() => {});
  await db.createCollection("beliefs");

  await retryUntilReady(
    () =>
      beliefsCol.createSearchIndex({
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
              reinforcement_count: { type: "number" },
              confidence: { type: "number" },
            },
          },
        },
      }),
    "createBeliefsSearchIndex(bm25)",
    60_000,
  );

  await retryUntilReady(
    () =>
      beliefsCol.createSearchIndex({
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
      } as Parameters<typeof beliefsCol.createSearchIndex>[0]),
    "createBeliefsSearchIndex(vector)",
    60_000,
  );

  await retryUntilReady(
    async () => {
      const indexes = (await beliefsCol.listSearchIndexes().toArray()) as Array<
        Record<string, unknown>
      >;
      const bm25 = indexes.find((i) => i.name === "beliefs_search");
      const vec = indexes.find((i) => i.name === VECTOR_INDEX_NAME);
      if (bm25?.status !== "READY")
        throw new Error(`beliefs bm25 status: ${bm25?.status}`);
      if (vec?.status !== "READY")
        throw new Error(`beliefs vector status: ${vec?.status}`);
    },
    "waitForBeliefsIndexes",
    60_000,
  );

  const rawBeliefs = JSON.parse(
    readFileSync(FIXTURE_BELIEFS, "utf8"),
  ) as Record<string, unknown>[];

  const first = rawBeliefs[0] as Record<string, unknown>;
  if (!Array.isArray(first?.embedding) || first.embedding.length === 0) {
    throw new Error(
      "beliefs.seed.embedded.json is missing embeddings. " +
        "Run `npx tsx src/__fixtures__/embed-seed.ts` first.",
    );
  }

  await beliefsCol.insertMany(rawBeliefs.map(coerceBelief));

  adapter = new VectorAdapter(beliefsCol);

  await retryUntilReady(
    async () => {
      const probe = await cachedEmbed("typescript");
      const results = await beliefsCol
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
    "waitForBeliefsVectorSync",
    READY_TIMEOUT_MS,
  );

  await retryUntilReady(
    async () => {
      const results = await beliefsCol
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
        throw new Error("beliefs BM25 index not synced");
    },
    "waitForBeliefsBM25Sync",
    READY_TIMEOUT_MS,
  );
});

test.after.always(async () => {
  const ingestionLatencies = adapter.ingestionReport.map((r) => r.latencyMs);
  const totalIngestionMs = ingestionLatencies.reduce((s, v) => s + v, 0);
  const meanIngestionMs =
    ingestionLatencies.length > 0
      ? Math.round((totalIngestionMs / ingestionLatencies.length) * 100) / 100
      : 0;

  const opts: ReportSummaryOptions = {
    provider: "vector",
    entries: sessionReport,
    caseCount: sessionCases.length,
    turnCount: sessionReport.length,
  };

  if (RESEED) {
    opts.ingestion = {
      beliefCount: adapter.ingestionReport.length,
      totalMs: Math.round(totalIngestionMs * 100) / 100,
      meanPerBeliefMs: meanIngestionMs,
      perBelief: adapter.ingestionReport,
    };
  }

  mkdirSync(REPORT_DIR, { recursive: true });
  writeFileSync(
    REPORT_PATH,
    JSON.stringify(buildReportPayload(opts, sessionReport), null, 2),
  );

  await client?.close();
  stopContainer();
});

function loadSessionCases(): SessionEvalCase[] {
  let raw: string;
  try {
    raw = readFileSync(FIXTURE_SESSION_CASES, "utf8");
  } catch {
    throw new Error(
      `Session eval fixture not found at ${FIXTURE_SESSION_CASES}. ` +
        `Create the file with at least an empty array [].`,
    );
  }
  const parsed: unknown = JSON.parse(raw);
  if (!Array.isArray(parsed)) {
    throw new Error(
      `Session eval fixture must be a JSON array, got ${typeof parsed}`,
    );
  }
  return parsed as SessionEvalCase[];
}

const sessionCases = loadSessionCases();

for (const sessionCase of sessionCases) {
  test.serial(`[vector] session: ${sessionCase.caseId}`, async (t) => {
    let casePassedSoFar = true;

    for (const turn of sessionCase.turns) {
      const buildStart = performance.now();
      const ctx = await adapter.buildContext(
        USER_ID,
        turn.scope,
        turn.userMessage,
        EVAL_PERSONA,
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

      const failures: string[] = [];
      const check = (cond: boolean, msg: string): void => {
        if (!cond) failures.push(msg);
      };

      const rb = turn.expect.relevantBeliefs;
      if (rb) {
        if (rb.shouldOnlyInclude) {
          const expectedSet = new Set(rb.shouldOnlyInclude);
          for (const id of relevantIds) {
            check(
              expectedSet.has(id),
              `[vector] turn ${turn.turnIndex}: unexpected relevant belief: ${id}`,
            );
          }
          for (const id of expectedSet) {
            check(
              relevantIds.has(id),
              `[vector] turn ${turn.turnIndex}: missing expected relevant belief: ${id}`,
            );
          }
        }
        for (const id of rb.mustInclude ?? []) {
          check(
            relevantIds.has(id) || pinnedIds.has(id),
            `[vector] turn ${turn.turnIndex}: missing expected belief: ${id}`,
          );
        }
        for (const id of rb.mustExclude ?? []) {
          check(
            !relevantIds.has(id) && !pinnedIds.has(id),
            `[vector] turn ${turn.turnIndex}: forbidden belief surfaced: ${id}`,
          );
        }
      }

      const pf = turn.expect.pinnedFacts;
      if (pf) {
        for (const id of pf.mustInclude ?? []) {
          check(
            pinnedIds.has(id),
            `[vector] turn ${turn.turnIndex}: missing pinned belief: ${id}`,
          );
        }
        for (const id of pf.mustExclude ?? []) {
          check(
            !pinnedIds.has(id),
            `[vector] turn ${turn.turnIndex}: forbidden pinned belief: ${id}`,
          );
        }
      }

      const oq = turn.expect.openQuestions;
      if (oq) {
        for (const id of oq.mustInclude ?? []) {
          check(
            questionIds.has(id),
            `[vector] turn ${turn.turnIndex}: missing expected question: ${id}`,
          );
        }
        for (const id of oq.mustExclude ?? []) {
          check(
            !questionIds.has(id),
            `[vector] turn ${turn.turnIndex}: forbidden question surfaced: ${id}`,
          );
        }
      }

      const noiseBeliefIds: string[] = [];
      const noiseCheck = turn.expect.noiseCheck;
      if (noiseCheck) {
        for (const id of noiseCheck.mustNotSurface) {
          if (relevantIds.has(id) || pinnedIds.has(id)) {
            noiseBeliefIds.push(id);
            failures.push(
              `[vector] turn ${turn.turnIndex}: noise belief surfaced: ${id}`,
            );
          }
        }
      }

      const goldSet = rb?.shouldOnlyInclude
        ? new Set(rb.shouldOnlyInclude)
        : new Set(rb?.mustInclude ?? []);

      const hits = [...goldSet].filter(
        (id) => relevantIds.has(id) || pinnedIds.has(id),
      ).length;

      const retrievalRecall = goldSet.size === 0 ? null : hits / goldSet.size;
      const retrievalPrecision =
        relevantIds.size === 0 ? null : hits / relevantIds.size;

      const driftScore =
        retrievalPrecision !== null ? 1 - retrievalPrecision : 0;

      sessionReport.push({
        caseId: sessionCase.caseId,
        category: "Session-level noise isolation",
        turnIndex: turn.turnIndex,
        label: turn.label,
        retrievedBeliefIds: [...relevantIds],
        pinnedBeliefIds: [...pinnedIds],
        noiseBeliefIds,
        driftScore,
        passed: failures.length === 0,
        failures,
        retrievalLatencyMs: Math.round((buildEnd - buildStart) * 100) / 100,
        retrievalPrecision,
        retrievalRecall,
      });

      if (failures.length > 0) {
        casePassedSoFar = false;
      }

      if (turn.createBeliefAtTurn) {
        const beliefDoc = coerceBelief(
          turn.createBeliefAtTurn as Record<string, unknown>,
        );

        const embedText = beliefEmbedText({
          canonical_name: beliefDoc.canonical_name as string,
          aliases: (beliefDoc.aliases as string[]) ?? [],
          ...(beliefDoc.content != null && {
            content: beliefDoc.content as string,
          }),
          ...(beliefDoc.why_it_matters != null && {
            why_it_matters: beliefDoc.why_it_matters as string,
          }),
        });
        const embedding = await cachedEmbed(embedText);
        const beliefWithEmbedding = { ...beliefDoc, embedding };

        await beliefsCol.insertOne(
          beliefWithEmbedding as Belief & { embedding: number[] },
        );

        await retryUntilReady(
          async () => {
            const probe = await cachedEmbed(beliefDoc.canonical_name as string);
            const results = await beliefsCol
              .aggregate([
                {
                  $vectorSearch: {
                    index: VECTOR_INDEX_NAME,
                    path: "embedding",
                    queryVector: probe,
                    numCandidates: 20,
                    limit: 5,
                    filter: { user_id: { $eq: USER_ID } },
                  },
                },
                { $match: { _id: beliefDoc._id } },
                { $limit: 1 },
              ])
              .toArray();
            if (results.length === 0)
              throw new Error(
                `Belief ${beliefDoc._id} not yet indexed in vector`,
              );
          },
          `waitForBeliefVectorSync:${beliefDoc._id}`,
          READY_TIMEOUT_MS,
        );
      }

      if (turn.updateBeliefAtTurn) {
        const { beliefId, addAliases, setContent, setCanonicalName } =
          turn.updateBeliefAtTurn;

        const existing = await beliefsCol.findOne({ _id: beliefId });
        if (!existing) {
          throw new Error(`updateBeliefAtTurn: belief ${beliefId} not found`);
        }

        const setFields: Record<string, unknown> = { updated_at: new Date() };
        if (setContent) setFields.content = setContent;
        if (setCanonicalName) setFields.canonical_name = setCanonicalName;

        if (Object.keys(setFields).length > 1) {
          await beliefsCol.updateOne({ _id: beliefId }, { $set: setFields });
        }

        const normalisedAliases =
          addAliases?.map((a) => a.trim().toLowerCase()) ?? [];

        if (normalisedAliases.length > 0) {
          await beliefsCol.updateOne(
            { _id: beliefId },
            { $addToSet: { aliases: { $each: normalisedAliases } } },
          );
        }

        const updatedCanonicalName =
          setCanonicalName ?? (existing.canonical_name as string);
        const updatedAliases = [
          ...new Set([
            ...((existing.aliases as string[]) ?? []),
            ...normalisedAliases,
          ]),
        ];

        const resolvedContent =
          setContent ?? (existing.content as string | undefined);

        const embedText = beliefEmbedText({
          canonical_name: updatedCanonicalName,
          aliases: updatedAliases,
          ...(resolvedContent != null && {
            content: resolvedContent as string,
          }),
          ...(existing.why_it_matters != null && {
            why_it_matters: existing.why_it_matters as string,
          }),
        });

        const updatedEmbedding = await cachedEmbed(embedText);

        await beliefsCol.updateOne(
          { _id: beliefId },
          { $set: { embedding: updatedEmbedding } },
        );

        const probeText =
          normalisedAliases.length > 0
            ? normalisedAliases[0]
            : updatedCanonicalName;

        await retryUntilReady(
          async () => {
            const probeVec = await cachedEmbed(probeText);
            const results = await beliefsCol
              .aggregate([
                {
                  $vectorSearch: {
                    index: VECTOR_INDEX_NAME,
                    path: "embedding",
                    queryVector: probeVec,
                    numCandidates: 20,
                    limit: 5,
                    filter: { user_id: { $eq: USER_ID } },
                  },
                },
                { $match: { _id: beliefId } },
                { $limit: 1 },
              ])
              .toArray();
            if (results.length === 0)
              throw new Error(
                `Belief ${beliefId} not yet updated in vector index for probe "${probeText}"`,
              );
          },
          `waitForBeliefVectorUpdateSync:${beliefId}`,
          READY_TIMEOUT_MS,
        );
      }
    }

    if (!casePassedSoFar) {
      const allFailures = sessionReport
        .filter((r) => r.caseId === sessionCase.caseId && !r.passed)
        .flatMap((r) => r.failures);
      t.is(0, allFailures.length, allFailures.join(" | "));
    } else {
      t.pass();
    }
  });
}
