# Math Research Platform Direction

This document captures the second-stage direction for building a mathematics
research platform on top of Hermes Simple.

The assumption is that stage one has already made Hermes Simple internally
consistent, stable, and well tested as an agent runtime. This document focuses
on what should be built above that runtime, not on the current simplification
work.

## Core Position

Hermes Simple is a good foundation for a mathematics research platform when it
is treated as the agent runtime and tool orchestration layer.

The mathematics platform should not be implemented by adding domain logic
directly into `run_agent.py`. Instead, it should be a separate domain layer
connected through tools, plugins, structured storage, and UI surfaces.

Hermes should provide:

- agent loop and tool orchestration
- session and memory infrastructure
- file, terminal, browser, and code execution tools
- CLI, one-shot, TUI, dashboard, and gateway entry points
- logging, profiles, configuration, and runtime management

The mathematics layer should provide:

- mathematical objects
- formal verification workflows
- literature ingestion and provenance
- computation adapters
- research-state tracking
- a research-oriented UI

## Missing Platform Pieces

### 1. Formal Verification Layer

Lean 4 and mathlib should be the first-class verification backend.

The platform needs workflows for:

- theorem statement creation
- proof attempt generation
- Lean file management
- proof checking
- error feedback loops
- verified theorem status

LLM output should be treated as a proof candidate, not as a verified result.
A statement or proof only becomes formal when Lean or another proof assistant
checks it successfully.

### 2. Mathematical Literature Layer

The platform needs a structured literature system rather than ordinary file
search alone.

Required capabilities include:

- arXiv, PDF, and TeX ingestion
- extraction of definitions, lemmas, theorems, proofs, formulas, and citations
- citation graph construction
- theorem and definition retrieval
- paper notes linked to exact sources
- provenance for every imported claim

Research output should be able to answer: where did this claim come from?
Possible provenance units include paper, section, page, theorem number, TeX
source location, or quoted source text when available.

### 3. Research Object Model

The platform should store research state as structured objects, not only as
chat messages.

Important object types:

- project
- problem
- conjecture
- definition
- lemma
- theorem
- proof attempt
- counterexample
- computation
- bibliography item
- Lean file
- LaTeX note
- research log entry

Each object should have stable identity, status, timestamps, links to related
objects, and provenance.

### 4. Computation Tool Layer

Mathematics research often needs symbolic, numeric, and domain-specific
computation. The platform should expose these through dedicated adapters,
not through ad hoc shell commands only.

Likely adapters:

- SageMath
- SymPy
- PARI/GP
- GAP
- Macaulay2
- Mathematica or Magma where available
- project-local Python experiments

Computations should be recorded with input, output, environment, and status so
they can be inspected and rerun.

### 5. Trust And Status System

The platform should make trust level explicit for every claim or result.

Suggested statuses:

- `speculative`: generated idea or conjecture
- `informal`: natural-language reasoning only
- `cited`: supported by literature provenance
- `computational`: checked by computation or experiment
- `formal`: verified by a proof assistant
- `failed`: attempted and failed, with error record
- `refuted`: counterexample or contradiction found

This is a critical product boundary. A mathematics platform that does not
separate informal reasoning from verified results will not be reliable enough
for serious research use.

### 6. Research UI

The primary interface should not be just a chat window.

A useful mathematics research UI should look more like an integrated research
workspace:

- project and object tree
- theorem / lemma / conjecture list
- Lean editor or Lean file view
- LaTeX notes
- proof attempt history
- computation panels
- reference and citation panel
- agent chat as an assistant surface

Chat should remain available, but it should be one interaction mode around
structured research objects, not the whole product.

## Suggested Layering

```text
Hermes Simple
  - agent loop
  - tool registry
  - sessions and memory
  - terminal/browser/file/code tools
  - CLI/TUI/dashboard/gateway

Math Research Layer
  - project store
  - research object model
  - Lean adapter
  - CAS adapters
  - paper ingestion
  - theorem and formula retrieval
  - provenance graph
  - trust/status system
  - research workspace UI
```

## First Minimum Viable Loop

The first end-to-end loop should be small and strict:

```text
User enters a mathematics problem
  -> agent searches project notes and papers
  -> platform creates structured conjectures / lemmas
  -> agent generates proof or computation attempts
  -> Lean or CAS checks local steps
  -> results are saved with status and provenance
  -> UI shows what is verified, failed, cited, or speculative
```

This loop is more important than a broad feature list. Once it works, the
platform can expand safely into richer literature ingestion, larger proof
automation, collaborative workflows, and specialized mathematical domains.

## Open Questions

- Which mathematical domain should be the first target?
- Should Lean 4 be mandatory in the first version or optional but integrated?
- What storage backend should hold research objects: SQLite first, or a more
  specialized document/graph store?
- Should papers be ingested as source TeX first, PDF first, or both?
- What is the first UI surface: CLI/TUI, dashboard, or a separate web app?
- How much of the current Hermes memory system should be reused versus creating
  a domain-specific research store?
