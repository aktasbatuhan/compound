# Two experiments to run next

Both are written after the caching fix of 2026-09-15. Every paid path now sends
a `cache_control` marker on hosts whose caching is opt-in, and
`tests/test_cache_policy.py` fails the build if one does not. Neither experiment
below starts until a two-episode probe shows `cached_tokens > 0` in
`calls.jsonl`.

Measured inputs both plans rely on, from the aborted 913-episode run:

| Quantity | Value |
|---|---|
| Agent tokens per retail episode | 65,031 input, 4,161 output (16:1) |
| Agent cost per episode, uncached | $0.0060 |
| Agent cost per episode, cached (measured, 79% hit) | $0.0021 |
| Simulator cost per episode, DeepSeek V4.1 Flash | $0.0014 |

## 1. Flex tier APIs benchmark

**Question.** Across providers that sell a cheaper asynchronous tier, what does
that tier actually cost you in wall-clock time, reliability and task success?
The pilot answered this for cost and latency on five routes and failed to answer
it for quality, because per-cell samples of six flip about a fifth of the time.

**Why it is not just a rerun.** Three things changed. Caching is on, which moves
the cost axis by roughly 65% on Doubleword and makes the earlier cost numbers
unusable. The simulator is DeepSeek V4.1 Flash rather than GPT-5.6 Sol, which
moves absolute pass rates. And the equivalence protocol in
`tier-equivalence.md` replaces the old difference test.

**Shape.** Two layers, because they have different sample requirements:

- *Breadth*, for the map: all five routes on both tiers, roughly 80 attempts per
  route, enough to pin a median cost and a median duration. About $19 with the
  old simulator costs, less now. This regenerates the positioning map with one
  simulator and one caching regime, which the current map does not have.
- *Depth*, for the quality claim: one route pair at 88 tasks x 7 trials, 1232
  episodes, using the frozen analysis in `scripts/analyze_tier_equivalence.py`.
  DeepSeek on Doubleword is the only affordable candidate: about **$4.30** now
  ($0.0021 + $0.0014 per attempt), against $17 for GLM and $157 for Astra.

**Do not** buy depth on more than one or two routes. Simulation puts power at 81%
for a +/- 7.5 point margin at 616 per arm; below that the answer is
"inconclusive", which is not worth paying for.

## 2. Fireworks claim replication on Doubleword

**The claim.** Fireworks published DeepSeek-V4.1-Flash at 74.34% pass@1 on
DeepSWE for $0.430 per task, against GPT-6 Astra at 74.12% for $6.524, and call
it 15x cheaper at equal quality. Their arithmetic reproduces exactly from their
published token mix of 148,741 uncached input, 36.7M cached input and 211,513
output tokens. It is an honest, auditable post.

**The gap worth attacking.** It is a claim about a model, not about a host. The
same model is served by seventeen providers, and on Fireworks' own token mix:

| Host | $/task | vs Fireworks | vs Astra |
|---|---:|---:|---:|
| DeepSeek direct | 0.2595 | 1.66x cheaper | 25.1x |
| GMICloud | 0.3632 | 1.18x | 18.0x |
| DeepInfra | 0.3771 | 1.14x | 17.3x |
| Fireworks | 0.4295 | - | 15.2x |
| Doubleword batch 24h | 0.4428 | 1.03x more | 14.7x |
| Doubleword async | 0.4868 | 1.13x more | 13.4x |
| Doubleword realtime | 0.5167 | 1.20x more | 12.6x |

Fireworks is fourth. Because 99.6% of input is cached on this workload, the
cache-read rate dominates the bill, and Doubleword charges $0.010/M against
Fireworks' $0.007 and DeepSeek's $0.003. The headline would be 1/25th on
DeepSeek's own endpoint.

**Do not replicate the quality half.** Their reported model gap is 0.22 points,
their own stated run-to-run variation is 1.4 to 3.2 points, and at 500 tasks the
standard error on a 74% pass rate is 2.0 points. Their full run cannot separate
those models and neither could ours. Paying for that is buying a known
"inconclusive".

**Replicate the economics instead**, which is the load-bearing part: does the
99.6% cache-hit rate hold off Fireworks, and what is the real token mix per host?
Near-deterministic, so 30 to 50 tasks is enough.

| Scope | 50 tasks | 100 | 500 |
|---|---:|---:|---:|
| Fireworks + both Doubleword tiers | $72 | $143 | $717 |
| Add DeepSeek direct | $85 | $169 | $847 |
| Add the GPT-6 Astra baseline | +$326 | +$652 | +$3,262 |

Skip the Astra arm; their table already reports it and it is the least contested
number in the post.

**Prerequisites.** DeepSWE needs Docker and the grader, so this one wants a VM,
about $10. We have SWE-bench Verified wired up, not DeepSWE, so the honest
framing is "their workload shape on our harness", not "we reproduced DeepSWE".
And Doubleword must run on chat completions with markers, or it shows up 11x
worse ($5.66 per task against $0.52) for a reason that has nothing to do with
Doubleword.
