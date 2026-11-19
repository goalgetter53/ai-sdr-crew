# ADR-0002: Layered prompt-injection defence for fetched web content

- Status: accepted
- Date: 2026-11-19

## Context

The researcher reads prospect websites. Any page can carry text aimed at the model ("ignore
previous instructions", hidden CSS, comments, alt text, homoglyphs, base64, fake `</system>` tags).
The blast radius in an SDR tool is real: links to attacker domains in outbound email, exfiltrated
contact data, CRM writes, SSRF to cloud metadata, or fabricated claims sent under our name.

No single control is reliable. Regex detection has false negatives; models can still follow
quarantined text; output filters miss plain-text payloads.

## Decision

Defence in depth, each layer independently testable and switchable for ablation
(`config.Guardrails`):

1. **SSRF-safe fetcher**: scheme/port allowlist, every resolved IP must be public, connection
   pinned to the validated IP (Host header + SNI), redirects re-validated, byte/time caps,
   content-type allowlist.
2. **Sanitiser + quarantine**: visible text only (hidden elements, comments, attributes, scripts
   dropped but still scanned for reporting), NFKC + zero-width removal + homoglyph folding,
   sentence-level instruction-pattern detection incl. base64-decoded blobs, flagged sentences
   replaced by a redaction marker, `<`/`>` neutralised, wrapped in `<untrusted_content>`. The shared
   system prompt tells every role that this block is data.
3. **Per-role tool allowlist**: only the researcher has tools, and only read tools. The write tool
   (`crm_log_activity`) exists but no LLM role may call it; non-tool roles are forced to their
   submit tool via `tool_choice`.
4. **Fact grounding**: a research fact survives only if its quote appears verbatim in a tool output
   and its text adds no numbers. This also kills hallucinated facts.
5. **Citation checker**: every claim in the copy must cite a grounded fact, and every number must
   appear in the cited fact.
6. **Output guard**: no links outside the seller and prospect domains, no email addresses, no
   prompt/quarantine artefacts.
7. **Human approval**: nothing is sent; approval re-runs the linter and output guard.

## Evidence

`fixtures/redteam/` holds 14 attack pages. `sdr-crew redteam` runs each through the full pipeline
with a deliberately gullible mock model under 8 guardrail configurations. With all layers on:
0/14 compromised and 14/14 detected. With all switchable layers off: 10/14 compromised. Removing
only the sanitiser: 4/14 compromised, which shows the sanitiser is load-bearing for plain-text
payloads while tool-hijack and SSRF attacks are still stopped by the allowlist and the fetcher.
Zero false positives on the 21 golden-set websites. See `reports/redteam.md`.

## Consequences

- Regex detection will miss novel phrasings; that is why layers 3-7 do not depend on it.
- Sentence-level redaction can remove a benign sentence that looks like an instruction
  (for example a careers page saying "include your portfolio in the email"). The cost is a lost
  fact, not a wrong email.
- Live-model resistance is not measured yet (the mock is gullible by design); see Roadmap.
