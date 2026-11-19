# ADR-0003: Deterministic, prompt-sensitive mock LLM and honestly labelled evals

- Status: accepted
- Date: 2026-11-19

## Context

We want CI that exercises the full system (tool loop, schema validation, budgets, caching
accounting, revise loop, guardrails, HITL) on every commit, without an API key, network access or
flaky model output. We also want the eval and regression tooling to be real, so it can be pointed
at a live model unchanged.

The risk is the classic portfolio trap: presenting numbers from a stub as if they were model
quality.

## Decision

1. `MockBackend` implements the same `create(LLMRequest) -> LLMResponse` interface as
   `AnthropicBackend`, returning `tool_use` blocks, so every code path above the backend is shared.
2. The mock is **prompt-sensitive**: it follows a writing rule only if that rule's phrase is in the
   system prompt (`RULE_PHRASES`). Prompt v1 lacks most rules, so v1 drafts contain links in
   email 1, hype words, unfilled tokens, uncited stats and occasional invented facts; v2 does not.
   That makes the v1 -> v2 regression report meaningful as a test of the harness.
3. The mock is **gullible**: it reads through obfuscation and obeys instructions in web text, so
   red-team results measure our deterministic defences rather than model virtue.
4. Token counts are estimated at ~4 characters per token. Caching follows Anthropic's rules: the
   prefix (tools, then system blocks up to the `cache_control` breakpoint) must reach 1,024 tokens.
5. Every report says `MOCK MODE` in its title; costs are labelled as estimates with assumed
   Sonnet-class prices (`config.Pricing`). Live numbers are TBD until someone runs
   `sdr-crew eval --live` with a key.

## Consequences

- CI runs the whole system in ~5 seconds with 183 tests.
- Mock quality numbers (judge scores) say nothing about real persuasion quality. The README and
  every report say so.
- The mock surfaced a real design issue: prompt v1's shared prompt was below the 1,024-token
  cacheable minimum for 3 of 5 roles, so caching silently did nothing for them. This is the kind of
  finding the harness exists for, and it carries over to the live API.
