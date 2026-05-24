# Refactor Spec: `context-1-data-gen` → Heavy Equipment RAG (Agent-Driven)

**Base:** https://github.com/chroma-core/context-1-data-gen
**Target:** synthetic data generator for closed-corpus heavy equipment manuals. Retriever (client OpenSearch) is out of scope. Pipeline is driven by a coding agent (Claude Code / Cursor / similar) reading this repo, not a human running scripts.

---

## Agent-accessibility requirements (apply to every file)

The whole repo must be navigable and editable by an agent with no prior context. Concretely:

1. **Root `AGENTS.md`** — single entry point. Names the modules, the contracts between them, the invariants, and where to look for what. Agent reads this first, always.
2. **Per-module `SKILL.md`** — every directory under `core/` and `domains/equipment/` has one. States: purpose, public API, invariants, failure modes, test entry point, do-not-touch list.
3. **Typed contracts everywhere.** Every public function has a `dataclass` or `TypedDict` for input and output. No dict-shaped magic. Pydantic for anything crossing a process boundary.
4. **No hidden state.** No module-level mutable singletons. Clients (LLM, OpenSearch, corpus) are injected. Agent can swap them in tests without monkey-patching.
5. **CLI entry per stage.** Each stage runnable in isolation via `python -m synth.<stage> --in <path> --out <path>`. Agent can test stages independently without rebuilding the whole pipeline.
6. **JSON artifacts on disk between stages.** Pipeline writes `exploration.jsonl`, `candidates.jsonl`, `verified.jsonl`, `tasks.jsonl`. Agent can inspect or hand-edit between stages. Resume is free.
7. **Structured logs, not prints.** Every stage emits JSONL traces to `runs/<run_id>/trace.jsonl`. Agent greps for failure causes without re-running.
8. **Deterministic seeds.** Every randomized choice (sampling, LLM temperature, ordering) takes a `seed: int`. Reproducible by default.
9. **Test scaffolding mandatory.** Every module has a `tests/` sibling with a fixtures-based test that runs offline (no LLM calls, no OpenSearch). Agent verifies its own edits.
10. **Failure surfaces, not exceptions.** Stages return `Result[T, FailureReason]`-shaped objects. Drops are first-class data, not silent skips. Agent can audit drop distribution.

---

## Root layout

```
agentic_search_data_gen/
├── AGENTS.md                       # entry point — read this first
├── README.md                       # human-oriented, secondary
├── pyproject.toml
├── .env.example
├── core/
│   ├── SKILL.md
│   ├── explore.py
│   ├── verify.py
│   ├── distract.py
│   ├── extend.py
│   ├── normalize.py
│   ├── stress.py
│   ├── types.py                    # all shared dataclasses
│   ├── results.py                  # Result[T, FailureReason]
│   ├── tracing.py                  # JSONL trace emitter
│   └── tests/
└── domains/
   └── equipment/
       ├── SKILL.md
       ├── explore.py
       ├── generate.py
       ├── verify.py
       ├── distract.py
       ├── extend.py
       ├── seed.py
       ├── tools.py                # adapter to heavyequipemnt-rag/app/tools.py
       ├── run.py                  # full pipeline driver
       ├── prompts/
       │   ├── explorer.md
       │   ├── generator.md
       │   └── judge.md
       ├── fixtures/               # offline test fixtures
       │   ├── corpus_sample.jsonl
       │   ├── exploration_sample.json
       │   └── candidate_sample.json
       └── tests/
```

---

## `AGENTS.md` (root) — required contents

```markdown
# Agent guide

## What this repo does
Generates verified (question, answer, supporting_chunks, distractors) tasks
from a heavy equipment manual corpus. Every fact is substring-grounded in
real chunks. Output is ranked by failure of the client's OpenSearch retriever.

## Pipeline
seed → explore → generate → verify → distract → (optional) extend → stress → emit
Each stage reads JSONL from disk, writes JSONL to disk. Stages are independently runnable.

## Where to look
- Contracts and dataclasses: core/types.py
- Failure taxonomy: core/results.py
- Per-module guides: */SKILL.md
- Prompts: domains/equipment/prompts/*.md
- Offline tests: */tests/  — run these before any change
- Pipeline entry: domains/equipment/run.py

## Invariants (never violate)
1. Every Clue.verbatim_span must be a substring of corpus[Clue.chunk_id]
  after normalize.normalize_all. Enforced in core/verify.py.
2. Generator and judge are DIFFERENT models. Configured via env.
3. No external web calls anywhere. Corpus is the only source of truth.
4. No mutation of input chunks. Read-only.
5. Drops are logged with FailureReason. Never silent.

## How to add a feature
1. Update core/types.py first if the contract changes.
2. Add an offline fixture in domains/equipment/fixtures/.
3. Write the test in */tests/ before the code.
4. Implement.
5. Run `make test` (offline, no API calls).

## How to debug a drop
1. Inspect runs/<run_id>/trace.jsonl
2. Filter by stage and FailureReason
3. Stage-specific replay: `python -m domains.equipment.<stage> --in <jsonl> --out /tmp/out`
```

---

## Keep from Context-1

- `core/explore.py` — `BaseExplorerAgent` abstraction
- `core/verify.py` — `BaseVerifier` extraction-then-substring pattern
- `core/distract.py` — `BaseDistractorAgent`
- `core/extend.py` — `BaseExtenderAgent` chaining
- `core/utils.py` → split into `core/utils.py` (Anthropic client wrapper) + `core/types.py` (dataclasses) for agent clarity
- Explore → Verify → Distract → Extend control flow
- Audit gate (≥80% human alignment on 200 hand-labeled tasks)

## Delete from Context-1

- `domains/web/`, `domains/sec/`, `domains/patents/`, `domains/epstein/`
- `core/rerank.py` (Baseten)
- `core/indexing.py` (Chroma indexing — corpus is pre-indexed in client OpenSearch)
- env: `SERPER_API_KEY`, `JINA_API_KEY`, `BASETEN_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`

## Add

- `domains/equipment/` (only domain)
- `core/normalize.py`, `core/stress.py`, `core/types.py`, `core/results.py`, `core/tracing.py`
- env: `OPENSEARCH_ENDPOINT`, `OPENSEARCH_INDEX`, `OPENSEARCH_AUTH`, `GENERATOR_MODEL`, `JUDGE_MODEL`, `EQUIPMENT_RAG_APP_PATH`

---

## `core/types.py` — single source of truth for contracts

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ChunkRef:
   chunk_id: str
   doc_id: str
   section_id: str | None
   model: str | None
   doc_type: str | None
   revision: str | None
   year: int | None

@dataclass(frozen=True)
class Clue:
   text: str
   chunk_id: str
   verbatim_span: str

@dataclass(frozen=True)
class Seed:
   doc_id: str
   section_id: str
   rationale: str  # why this seed was picked

@dataclass(frozen=True)
class ExplorationResult:
   seed: Seed
   linked_chunks: list[ChunkRef]
   chunk_texts: dict[str, str]
   anchors: list[str]

@dataclass(frozen=True)
class TaskCandidate:
   clues: list[Clue]
   question: str
   answer: str
   answer_chunk_id: str
   answer_span: str

@dataclass(frozen=True)
class StressResult:
   top_k_used: int
   found_at_rank: int | None
   retriever_failed: bool
   retrieved_chunk_ids: list[str]

@dataclass(frozen=True)
class Task:
   task_id: str
   candidate: TaskCandidate
   supporting_chunk_ids: list[str]
   distractor_chunk_ids: list[str]
   hops: int
   provenance: dict[str, dict]  # chunk_id -> ChunkRef as dict
   stress: StressResult | None

FailureReason = Literal[
   "explorer_no_anchors",
   "explorer_boilerplate_only",
   "explorer_provenance_conflict",
   "generator_invalid_json",
   "verifier_substring_failed",
   "verifier_judge_rejected",
   "verifier_provenance_failed",
   "verifier_conditional_missing",
   "distractor_leakage",
   "extender_no_bridge",
   "stress_unreachable",
]
```

## `core/results.py`

```python
from dataclasses import dataclass
from typing import Generic, TypeVar
T = TypeVar("T")

@dataclass(frozen=True)
class Ok(Generic[T]):
   value: T

@dataclass(frozen=True)
class Drop:
   reason: FailureReason
   detail: dict  # arbitrary structured info for tracing

Result = Ok[T] | Drop
```

Stages return `Result`. Drops are emitted to the trace and to a `drops.jsonl` file. Never silently swallowed.

## `core/tracing.py`

```python
def emit(run_id: str, stage: str, event: dict) -> None: ...
# appends to runs/<run_id>/trace.jsonl
# event always includes: ts, stage, task_id (if any), level
```

Every stage emits start, decision, drop, and complete events. Agent greps `trace.jsonl` to diagnose.

---

## Per-stage refactor (with agent-accessibility additions)

Each stage below: (a) what it does, (b) CLI entry, (c) input/output JSONL contracts, (d) what changed from Context-1.

### `domains/equipment/seed.py`

- **Replaces:** all four Context-1 domain seed strategies.
- **CLI:** `python -m domains.equipment.seed --corpus-meta <path> --n 1000 --out seeds.jsonl --seed 42`
- **Input:** corpus metadata dump (one row per `(doc_id, section_id)`).
- **Output:** `seeds.jsonl` of `Seed` records.
- **Behavior:** stratified sampling across `(model, doc_type)`. Prefer sections rich in numeric specs, fault codes, part numbers. Deterministic under `--seed`.

### `domains/equipment/explore.py`

- **Replaces:** `domains/web/explore.py` Serper+Jina loop.
- **CLI:** `python -m domains.equipment.explore --seeds seeds.jsonl --out exploration.jsonl`
- **Input:** `seeds.jsonl`.
- **Output:** `exploration.jsonl` of `Result[ExplorationResult, FailureReason]`.
- **Tool surface:** injected `EquipmentTools` (wraps `heavyequipemnt-rag/app/tools.py`).
- **Rejection rules logged as drops:** boilerplate-only, no anchors, provenance conflict.

### `domains/equipment/generate.py`

- **Replaces:** the combined explore+generate web step. Now isolated.
- **CLI:** `python -m domains.equipment.generate --in exploration.jsonl --out candidates.jsonl`
- **Output:** `candidates.jsonl` of `Result[TaskCandidate, FailureReason]`.
- **Prompt:** `prompts/generator.md`. Strict JSON output schema enforced by Pydantic parse; invalid JSON → drop with `generator_invalid_json`.
- **Hard prompt rules:** every `verbatim_span` literal substring; conditional language propagates into clues; tabular answers carry header context.

### `core/normalize.py`

- **New module.**
- **CLI:** `python -m core.normalize --build-aliases --corpus <path> --out aliases.json`
- **Functions:** `normalize`, `normalize_units`, `normalize_codes`, `normalize_part_numbers`, `normalize_all`.
- **Alias tables:** built once from corpus, versioned, committed to repo. Agent can regenerate.

### `domains/equipment/verify.py`

- **Extends:** `core/verify.py`.
- **CLI:** `python -m domains.equipment.verify --in candidates.jsonl --out verified.jsonl`
- **Stages:**
 - A: deterministic substring (uses `normalize_all`)
 - B: independent judge model (different from generator, enforced at startup)
 - C: provenance consistency (all supporting chunks share `(model, revision)` unless cross-model task)
 - D: conditional integrity (if answer span contains conditional, clues must too)
- **Output:** `verified.jsonl` of `Result[TaskCandidate, FailureReason]` with `verifier_*` drop reasons.

### `domains/equipment/distract.py`

- **Extends:** `core/distract.py`.
- **CLI:** `python -m domains.equipment.distract --in verified.jsonl --out enriched.jsonl --n-distractors 10`
- **Mining:** hybrid_search on `answer_span` via `EquipmentTools`.
- **Leakage filter:** multi-normalizer (units, codes, part numbers). Drops logged with `distractor_leakage` and the offending normalized form.

### `domains/equipment/extend.py`

- **Refactor of:** web chaining.
- **CLI:** `python -m domains.equipment.extend --in enriched.jsonl --out chained.jsonl --hop-cap 4`
- **Bridge:** new task's exploration uses prior answer as a corpus anchor via `hybrid_search`, not a web query.

### `core/stress.py`

- **New module.**
- **CLI:** `python -m core.stress --in chained.jsonl --out tasks.jsonl --top-k 20`
- **Behavior:** POST each task's question to client OpenSearch, record `StressResult`. Tasks where `retriever_failed=True` are flagged as tier-1 output.

### `domains/equipment/run.py`

- **CLI:** `python -m domains.equipment.run --n 1000 --hop-dist '{"1":0.4,"2":0.4,"3":0.2}' --run-id $(date +%s)`
- **Behavior:** orchestrates the above in order, writing one JSONL per stage under `runs/<run_id>/`. Resumable: if `verified.jsonl` exists, skip ahead.

---

## `domains/equipment/tools.py` — agent-injectable tool surface

```python
from dataclasses import dataclass
from typing import Protocol

class EquipmentTools(Protocol):
   def vector_search(self, query: str, k: int) -> list[ChunkRef]: ...
   def keyword_search(self, query: str, k: int) -> list[ChunkRef]: ...
   def get_document_chunks(self, doc_id: str, section_id: str | None) -> list[tuple[ChunkRef, str]]: ...
   def list_documents(self) -> list[dict]: ...
   def rerank(self, query: str, candidates: list[ChunkRef]) -> list[ChunkRef]: ...

@dataclass
class ProductionTools(EquipmentTools):
   """Wraps heavyequipemnt-rag/app/tools.py."""
   ...

@dataclass
class FixtureTools(EquipmentTools):
   """Reads from domains/equipment/fixtures/corpus_sample.jsonl.
   Used in offline tests. Agent uses this to verify changes without API calls."""
   ...
```

Tests inject `FixtureTools`. Production injects `ProductionTools`. Agent edits either freely; the protocol is the contract.

---

## Per-module `SKILL.md` template

Every module under `core/` and `domains/equipment/` ships with:

```markdown
# <module>

## Purpose
One sentence.

## Public API
- function/class signatures with one-line descriptions

## Inputs / Outputs
- JSONL contract (which dataclass per line)

## Invariants
- bullet list. these are checked in tests.

## Failure modes
- which FailureReason values this stage can emit

## How to test
`pytest <module>/tests/ -q`

## Do not touch
- anything that would break the listed invariants without updating core/types.py first
```

---

## Test scaffolding (mandatory)

```
core/tests/
 test_normalize.py        # offline, no API
 test_verify.py           # offline, fixture-based
 test_results.py
domains/equipment/tests/
 test_seed.py             # offline, against fixtures/corpus_sample.jsonl
 test_explore.py          # uses FixtureTools, mock LLM client
 test_generate.py         # mock LLM returns canned JSON
 test_verify.py           # exercises all 4 verification stages
 test_distract.py         # exercises leakage filter against unit aliases
 test_extend.py
 test_run_smoke.py        # end-to-end on 5 seeds, offline
```

`make test` runs all of them with zero network calls. Agent verifies its own work before declaring a change complete.

---

## Env vars

Delete: `SERPER_API_KEY`, `JINA_API_KEY`, `BASETEN_API_KEY`, `CHROMA_API_KEY`, `CHROMA_DATABASE`.

Add:
```
OPENSEARCH_ENDPOINT
OPENSEARCH_INDEX
OPENSEARCH_AUTH
EQUIPMENT_RAG_APP_PATH       # path to heavyequipemnt-rag/app for tool imports
GENERATOR_MODEL              # e.g. claude-sonnet-4
JUDGE_MODEL                  # different family/size from generator; enforced at startup
RUN_DIR                      # default ./runs
```

Keep: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (if embedding queries are used anywhere).

---

## Output schema diff vs Context-1

Context-1 emits: `{clues, question, answer, supporting_chunk_ids, distractor_chunk_ids, hops}`.

Equipment emits the above plus:
```python
task_id: str
provenance: dict[chunk_id, ChunkRef]
stress: StressResult
```

Final ranking key: `(stress.retriever_failed desc, stress.found_at_rank desc nulls first)`.

---

## Summary table

| Concern | Context-1 | Equipment refactor |
|---|---|---|
| Entry point | per-domain README | root `AGENTS.md` + per-module `SKILL.md` |
| Contracts | dict-shaped | typed dataclasses in `core/types.py` |
| Failure handling | exceptions / skips | `Result` type, `FailureReason` enum, logged |
| Tracing | ad hoc | structured JSONL per run |
| Stage execution | monolithic per-domain run | CLI per stage, JSONL between |
| Tests | none shipped | offline fixtures + per-module tests, `make test` |
| Tool surface | hard-coded externals | injected `EquipmentTools` protocol |
| Determinism | mixed | explicit `--seed`, reproducible runs |
| Resume | no | JSONL per stage on disk, re-run skips done stages |
