# Setup evidence check, 2026-09-18

Read-only investigation of saved diagnostics and public documentation. No new
inference calls or GCP jobs were launched. The runtime qualification policy has
not been changed by this investigation.

## Doubleword: missing echo is real; the mandatory-echo policy is ours

`artifacts/cache-evidence-2026-09-17/probe_v41.py` uses urllib directly, bypassing
both our gateway and the OpenAI SDK. Its saved JSON records HTTP 200 on four
DeepSeek V4.1 Flash Chat Completions calls: two requested `priority`, two `flex`.
All four omit top-level `service_tier`; their saved usage also lacks a tier, and
no saved header name identifies a tier. Each arm's second request reports 100%
cache reads. The artifact stores top-level keys, usage and header diagnostics,
not a complete raw response archive, so do not claim every possible metadata
location or today's API behavior has been exhaustively tested.

This is evidence against our gateway stripping the tier field. Our gateway also
uses `json.load` directly before reading `raw.get("service_tier")`.

Doubleword's [official introduction](https://docs.doubleword.ai/inference-api/intro-to-doubleword-inference)
explicitly documents `priority` and `flex` on Chat Completions. The
[Supermemory integration](https://docs.doubleword.ai/inference-api/integrations/supermemory)
also forwards Chat Completions with `service_tier="flex"`. The documentation
checked does not establish a mandatory response echo. Missing echo therefore
does not establish that Flex is unsupported, ignored or billed incorrectly.
Latency and infrastructure-header differences are not independent tier proof.

The implemented gate deliberately requires stronger evidence than the API
contract establishes. Calling this an unavoidable provider blocker was too
strong. There are two legitimate study designs:

- A study of documented requested service policies: log the exact request tier,
  leave served tier unverified, and describe results as outcomes when requesting
  each policy. Cost derived from the requested tier's rate card remains
  conditional, not independent billing evidence.
- A study claiming independently verified served tiers and actual discounts:
  require response or request-linked provider receipt evidence.

For the exploratory pilot, the first design can be defensible if declared in
advance. It requires an explicit protocol/code amendment and tests; the current
runner still blocks absent echoes. Do not hand-edit confirmation fields or
silently bypass the gate. Billing reconciliation remains necessary before
claiming independently measured savings from derived prices.

## Fireworks: public DeepSWE setup exists

The [official DeepSWE repository](https://github.com/datacurve-ai/deep-swe)
publishes 113 tasks, instructions, environment definitions, reference solutions,
verifiers and a Pier quickstart using mini-SWE-agent. Its README describes the
separate verifier environment and says leaderboard runs use Pier/mini-SWE-agent
on Modal. Public benchmark availability was not the missing piece; our earlier
handoff did not identify it clearly enough.

The [Fireworks article](https://fireworks.ai/blog/DeepSeek-V4.1-Flash-Astra)
provides model/effort, aggregate pass rates and token/cost breakdowns. I did not
find its exact run manifest, pinned agent version/configuration, task revision,
trial accounting or complete launch command in the article or the searches
performed. The public benchmark quickstart does not prove Fireworks used those
exact defaults. Treat their exact configuration as unverified, not proven
unpublished everywhere.

Claude can prepare a pinned DeepSWE comparison using the public benchmark. It
must distinguish that independent comparison from reproduction of Fireworks'
exact score. SWE-bench Verified is a different workload; mini-SWE-agent itself
is not the incompatibility. No new launch budget or executable Fireworks spec
is established by this discovery.
