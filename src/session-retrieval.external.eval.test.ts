/**
 * session-retrieval.external.eval.test.ts
 *
 * Universal eval runner for external memory providers.
 * Runs the session-retrieval.cases.json suite against whichever provider is
 * selected via the MEMORY_PROVIDER env var, then writes a JSON report.
 *
 * Usage:
 *   MEMORY_PROVIDER=mem0     npx ava session-retrieval.external.eval.test.ts
 *   MEMORY_PROVIDER=zep      npx ava session-retrieval.external.eval.test.ts
 *   MEMORY_PROVIDER=hindsight npx ava session-retrieval.external.eval.test.ts
 *
 * Set RESEED=true to seed beliefs to the provider before running:
 *   MEMORY_PROVIDER=mem0 RESEED=true npx ava session-retrieval.external.eval.test.ts
 *
 * Reports land at:
 *   test-results/session-retrieval-report-mem0.json
 *
 * To add a new provider: edit providers.config.json and add a resolver block
 * to universalSessionAdapter.ts. This file never needs to change.
 */

import test from "ava";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";
import type { Belief } from "./types/belief.js";
import {
  type PersonaLookup,
  type ProviderName,
} from "./adapters/baseAdapter.js";
import { UniversalSessionAdapter } from "./adapters/universalSessionAdapter.js";
import { fileURLToPath } from "node:url";
import {
  buildReportPayload,
  type ReportSummaryOptions,
} from "./utils/buildRetrievalReport.js";

const rawProvider = process.env.MEMORY_PROVIDER?.trim().toLowerCase();
if (!rawProvider) {
  throw new Error("MEMORY_PROVIDER is not set.");
}

const __dirname = fileURLToPath(new URL(".", import.meta.url));

const PROVIDER = rawProvider as ProviderName;
const RESEED = process.env.RESEED === "true";

const USER_ID = "test-user";
const FIXTURE_BELIEFS = resolve(__dirname, "../fixtures/beliefs.seed.json");
const FIXTURE_SESSION_CASES = resolve(
  __dirname,
  "../fixtures/session-retrieval.cases.json",
);
const REPORT_DIR = resolve(__dirname, "../test-results");
const REPORT_PATH = resolve(
  REPORT_DIR,
  `session-retrieval-report-${PROVIDER}.json`,
);

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
  label: string;
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

const sessionReport: SessionReportEntry[] = [];
let adapter: UniversalSessionAdapter;

test.before(async () => {
  const raw = JSON.parse(readFileSync(FIXTURE_BELIEFS, "utf8")) as Record<
    string,
    unknown
  >[];
  const beliefs = raw.map(coerceBelief);

  adapter = new UniversalSessionAdapter(PROVIDER);

  if (RESEED) {
    await adapter.seed(beliefs);
  } else {
    adapter.loadFixture(beliefs);
  }
});

test.after.always(() => {
  const ingestionLatencies = adapter.ingestionReport.map((r) => r.latencyMs);
  const totalIngestionMs = ingestionLatencies.reduce((s, v) => s + v, 0);
  const meanIngestionMs =
    ingestionLatencies.length > 0
      ? Math.round((totalIngestionMs / ingestionLatencies.length) * 100) / 100
      : 0;

  const opts: ReportSummaryOptions = {
    provider: PROVIDER,
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
});

const sessionCases = JSON.parse(
  readFileSync(FIXTURE_SESSION_CASES, "utf8"),
) as SessionEvalCase[];

for (const sessionCase of sessionCases) {
  test.serial(`[${PROVIDER}-session] ${sessionCase.caseId}`, async (t) => {
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
              `turn ${turn.turnIndex}: unexpected relevant belief: ${id}`,
            );
          }
          for (const id of expectedSet) {
            check(
              relevantIds.has(id),
              `turn ${turn.turnIndex}: missing expected belief: ${id}`,
            );
          }
        }
        for (const id of rb.mustInclude ?? []) {
          check(
            relevantIds.has(id) || pinnedIds.has(id),
            `turn ${turn.turnIndex}: missing belief: ${id}`,
          );
        }
        for (const id of rb.mustExclude ?? []) {
          check(
            !relevantIds.has(id) && !pinnedIds.has(id),
            `turn ${turn.turnIndex}: forbidden belief: ${id}`,
          );
        }
      }

      const pf = turn.expect.pinnedFacts;
      if (pf) {
        for (const id of pf.mustInclude ?? []) {
          check(
            pinnedIds.has(id),
            `turn ${turn.turnIndex}: missing pinned: ${id}`,
          );
        }
        for (const id of pf.mustExclude ?? []) {
          check(
            !pinnedIds.has(id),
            `turn ${turn.turnIndex}: forbidden pinned: ${id}`,
          );
        }
      }

      const oq = turn.expect.openQuestions;
      if (oq) {
        for (const id of oq.mustInclude ?? []) {
          check(
            questionIds.has(id),
            `turn ${turn.turnIndex}: missing question: ${id}`,
          );
        }
        for (const id of oq.mustExclude ?? []) {
          check(
            !questionIds.has(id),
            `turn ${turn.turnIndex}: forbidden question: ${id}`,
          );
        }
      }

      const noiseBeliefIds: string[] = [];
      if (turn.expect.noiseCheck) {
        for (const id of turn.expect.noiseCheck.mustNotSurface) {
          if (relevantIds.has(id) || pinnedIds.has(id)) {
            noiseBeliefIds.push(id);
            failures.push(
              `turn ${turn.turnIndex}: noise belief surfaced: ${id}`,
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

      if (failures.length > 0) casePassedSoFar = false;

      if (turn.createBeliefAtTurn) {
        const beliefDoc = coerceBelief(
          turn.createBeliefAtTurn as Record<string, unknown>,
        );
        await adapter.ingestBelief(beliefDoc);
      }

      if (turn.updateBeliefAtTurn) {
        const { beliefId, addAliases } = turn.updateBeliefAtTurn;
        if (addAliases?.length) {
          adapter.updateBeliefAliases(beliefId, addAliases);
        }
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
