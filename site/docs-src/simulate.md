title: Simulated users
order: 68

# Simulated users

A single prompt and a single answer is easy to compare across models. A real
conversation is not: the user reacts to what the model said, the model calls
tools, and a different model takes a different path. `compound simulate`
replays your real conversations with a simulated user in the seat of the real
one, so a candidate model meets the same situations your production model met,
and grades what it does.

```bash
# see the plan first; nothing is sent without --go
compound simulate traces/ --task billing \
    --incumbent openrouter/auto=openai/gpt-6-sol \
    --candidates openrouter/deepinfra/fp8=deepseek/deepseek-v4.1-flash,openrouter/auto=z-ai/glm-5.3

compound simulate traces/ ... --user llm --user-model openrouter/auto=openai/gpt-6-luna \
    --max-usd 10 --go
```

It reads `compound.trace` documents, which the trace pipeline writes from
Langfuse, plain JSON or OpenTelemetry exports.

## The simulated user

**Scripted** replays the real user's messages in order, whatever the model
says. It costs nothing and is exact when the user's next message does not
depend much on the answer.

**LLM** has a model play the user. It sees what the real user said, in order,
and keeps pursuing the same goal in the same language one message at a time,
reacting to what the candidate actually replies, and it ends the conversation
when the goal is met or the real user would have given up.

## Tools

Tool calls are answered by each tool's replay policy, the same four the rest
of Compound uses. **Recorded** returns what the real tool returned for the same
name and arguments; a call the recording never saw has no honest answer, so
the conversation stops there. **Mocked** returns a fixed result. **Live
read-only** calls a function you supply, for tools that only read. **Blocked**
stops the conversation, for tools with side effects that must never run.

## Grading

A conversation passes when it ends normally and the model made the same set
of tool calls, by name and arguments, as the real conversation did. An extra
refund or a missing lookup is a different outcome for the business, whatever
the final text says, so the check is strict, and recall and precision are kept
next to it so a near miss shows as one. A judge for the text of the answers can
be added per task once it is calibrated against human labels; it is not
assumed.

## First, check the simulator

Before any candidate is called, the production model runs through the same
simulator. If it cannot reproduce its own real conversations at the fidelity
threshold (80% by default), the simulator or the replay policy is not good
enough for this task, and the verdict is `simulator_unfaithful`. No candidate
is run, and no number is reported that would mislead.

## The verdict

Each candidate is compared with the production model scenario by scenario. A
paired bootstrap gives an interval for the difference in pass rate, and the
non-inferiority margin you set before the run (5 points by default) decides:
`meets_gate` when even the low end of the interval is above minus the margin,
`fails_gate` when even the high end is below it, and `insufficient_data`
otherwise or with fewer than `--min-scenarios` conversations.

Every episode, with its transcript, tool calls, stop reason, tokens and cost,
is written to `episodes.jsonl`, and the verdicts to `verdict.json`. Both hold
your conversation text, so they stay on your machine.

## Money

It is a dry run unless you pass `--go`. Conversations run one model call at a
time and stop once spend reaches `--max-usd`, so the cap can be passed by at
most one call.
