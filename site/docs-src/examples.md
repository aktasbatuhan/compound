title: Examples
order: 5
out: examples
nav: top

# Examples

Two runs made with Compound, published with every per-call record behind them.
They are small: one model each, a handful of hosts, and within-mode host
differences that mostly do not reach significance. Read them as worked
examples of what a report looks like, not as a leaderboard. The point of the
tool is that you run your own.

## Same model, eight hosts

`deepseek-v4-flash` on terminal-bench: eight serving configurations, two
pinned reasoning modes, 588 scored episodes, task time limits tripled so no
host fails only for being slow. Quality tied across hosts with reasoning off;
cost per resolved task did not. Whether reasoning helped depended on the host's
decode speed. Includes a streamed timing harness and a cache-behavior replay.

[Read the report](../report/)

## Pinned host vs router

`glm-5.3-flash` on Terminal-Bench 4.0: five OpenRouter routes (auto plus four
pinned hosts) run at the same time on separate VMs, five tasks, two attempts,
1,801 traced calls. Compares how often each route's calls stalled, what each
paid per prompt token, and where the router actually sent traffic. Quality is
not compared: at five tasks almost nothing resolved on any arm.

[Read the report](../report/tb4/)

## Reproduce either

Both reports were produced by the commands in the [docs](../docs/). The second
one is a single `compound-bench harbor` invocation per arm; the first is a
`compound-bench run terminal_bench --providers ...` sweep followed by
`bench_report`.
