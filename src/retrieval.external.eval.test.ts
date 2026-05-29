/**
 * retrieval.external.eval.test.ts
 *
 * Universal eval runner for external memory providers.
 * Runs the retrieval.cases.json suite against whichever provider is selected
 * via the MEMORY_PROVIDER env var, then writes a JSON report.
 *
 * Usage:
 *   MEMORY_PROVIDER=mem0     npx ava retrieval.external.eval.test.ts
 *   MEMORY_PROVIDER=zep      npx ava retrieval.external.eval.test.ts
 *   MEMORY_PROVIDER=hindsight npx ava retrieval.external.eval.test.ts
 *
 * Set RESEED=true to seed beliefs to the provider before running:
 *   MEMORY_PROVIDER=mem0 RESEED=true npx ava retrieval.external.eval.test.ts
 *
 * Reports land at:
 *   test-results/retrieval-report-mem0.json
 *
 * To add a new provider: edit providers.config.json and add a resolver block
 * to universalAdapter.ts. This file never needs to change.
 */

import test from "ava";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import type { Belief } from "./types/belief.js";
import {
  BaseAdapter,
  type ContextBudget,
  type PersonaLookup,
  type ProviderName,
} from "./adapters/baseAdapter.js";
import { fileURLToPath } from "node:url";
import { buildReportPayload } from "./utils/buildRetrievalReport.js";

const rawProvider = process.env.MEMORY_PROVIDER?.trim().toLowerCase();
if (!rawProvider) {
  throw new Error("MEMORY_PROVIDER is not set.");
}

const PROVIDER = rawProvider as ProviderName;
const RESEED = process.env.RESEED === "true";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

const USER_ID = "test-user";
const FIXTURE_BELIEFS = resolve(__dirname, "../fixtures/beliefs.seed.json");
const FIXTURE_CASES = resolve(__dirname, "../fixtures/retrieval.cases.json");
const REPORT_DIR = resolve(__dirname, "../test-results");
const REPORT_PATH = resolve(REPORT_DIR, `retrieval-report-${PROVIDER}.json`);

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
  passed: boolean;
  failures: string[];
  retrievalLatencyMs: number;
}

const DATE_FIELDS = [
  "created_at",
  "updated_at",
  "last_reinforced_at",
  "resolved_at",
] as const;

function coerceBelief(raw: Record<string, unknown>): Belief {
  const out: Record<string, unknown> = { ...raw };
  for (const f of DATE_FIELDS) {
    const v = out[f];
    if (typeof v === "string") out[f] = new Date(v);
  }
  const prov = out.provenance as Record<string, unknown> | undefined;
  if (prov?.extracted_at && typeof prov.extracted_at === "string") {
    prov.extracted_at = new Date(prov.extracted_at);
  }
  return out as unknown as Belief;
}

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

const report: ReportEntry[] = [];
let adapter: BaseAdapter;
let pinnedInSeed: Set<string>;

test.before(async () => {
  const raw = JSON.parse(readFileSync(FIXTURE_BELIEFS, "utf8")) as Record<
    string,
    unknown
  >[];
  const beliefs = raw.map(coerceBelief);

  adapter = new BaseAdapter(PROVIDER);

  if (RESEED) {
    await adapter.seed(beliefs);
  } else {
    adapter.loadFixture(beliefs);
  }

  pinnedInSeed = new Set(
    beliefs
      .filter((b) => b.pinned === true && b.user_id === USER_ID)
      .map((b) => b._id as string),
  );
});

test.after.always(() => {
  const ingestionLatencies = adapter.ingestionReport.map((r) => r.latencyMs);
  const totalIngestionMs = ingestionLatencies.reduce((s, v) => s + v, 0);
  const meanIngestionMs =
    ingestionLatencies.length > 0
      ? Math.round((totalIngestionMs / ingestionLatencies.length) * 100) / 100
      : 0;

  mkdirSync(REPORT_DIR, { recursive: true });
  writeFileSync(
    REPORT_PATH,
    JSON.stringify(
      buildReportPayload(
        {
          provider: PROVIDER,
          entries: report,
          caseCount: cases.length,
          ingestion: {
            beliefCount: adapter.ingestionReport.length,
            totalMs: Math.round(totalIngestionMs * 100) / 100,
            meanPerBeliefMs: meanIngestionMs,
            perBelief: adapter.ingestionReport,
          },
        },
        report,
      ),
      null,
      2,
    ),
  );
});

const cases = JSON.parse(
  readFileSync(FIXTURE_CASES, "utf8"),
) as RetrievalCase[];

for (const tc of cases) {
  test.serial(`[${PROVIDER}] ${tc.caseId}: ${tc.description}`, async (t) => {
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
          ? 0.0 //
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
      passed: failures.length === 0,
      failures,
      retrievalLatencyMs: Math.round((buildEnd - buildStart) * 100) / 100,
    });

    t.is(failures.length, 0, failures.join(" | "));
  });
}
