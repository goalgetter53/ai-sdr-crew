# ADR-0001: Explicit state machine instead of an agent framework

- Status: accepted
- Date: 2026-11-19

## Context

The crew has four LLM roles (researcher, strategist, copywriter, critic), a revise loop, budgets and
several deterministic gates (CRM status, fact grounding, linter, citation checker, output guard).
Frameworks such as LangGraph, CrewAI or AutoGen would give us orchestration for free, but they also
hide control flow, add a large dependency surface, and make it harder to prove properties like
"nothing is sent" or "a do-not-contact account never costs a token".

Outbound is a regulated, reputation-sensitive workflow. Reviewers (and auditors) need to read the
control flow in one file.

## Decision

Orchestrate with a small explicit state machine (`orchestrator.py`):

```
START -> GATE -> RESEARCH -> STRATEGY -> COPY -> CRITIQUE -> DONE
                                                  ^         |
                                                  +-REVISE<-+
any non-terminal stage -> NEEDS_HUMAN (abort: budget, agent failure, max revisions, no signal)
GATE -> HALTED (do-not-contact, existing customer, open opportunity)
```

- `TRANSITIONS` is a literal dict; `StateMachine.advance` raises `InvalidTransition` on anything
  else, and tests assert both the happy path and illegal moves.
- Deterministic checks run before the LLM critic; the critic is only called on drafts that are
  already lint-, citation- and guard-clean, which saves a call per failed draft.
- The agent runtime is ~100 lines: a tool-use loop where structured output is a `submit_<name>`
  tool whose input schema is the pydantic model's JSON schema. Validation errors are returned to
  the model as `is_error` tool results so it can self-correct.

## Consequences

- Positive: the whole control flow is greppable and testable; budgets are enforced before every
  call; adding a stage is a one-line transition change plus a test.
- Positive: zero framework lock-in; the same loop drives the live Anthropic backend and the mock.
- Negative: no built-in parallelism, persistence or visual graph editor. Runs are per-account and
  independent, so parallelism belongs at the batch level (roadmap), not inside a run.
- Negative: we maintain our own retry/turn-limit logic (small, covered by tests).
