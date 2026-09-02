title: Serving metrics
order: 60

# Serving metrics

Task success is one axis. `compound-bench serving` measures the others
directly: time to first token, decode speed, and cost per route, repeated on a
schedule so you can see how a host moves through the day.

```bash
compound-bench serving \
    --providers openrouter/deepinfra,doubleword/realtime,doubleword/flex,openrouter/auto \
    --model-or deepseek/deepseek-v4-flash-0731 \
    --model deepseek-ai/DeepSeek-V4-Flash-0731 \
    --shapes shapes.json --rounds 8 --interval 3600 --reps 3 \
    --out artifacts/serving
```

- `--model-or` is the OpenRouter slug; `--model` is the id Doubleword or a direct host uses for the same weights. Hosts name models differently.
- `--shapes` is a JSON file mapping a name to `{messages, response_format}`, so you measure the request shapes your workload actually sends, including structured output.
- `--rounds` and `--interval` schedule repeated measurements. `--reps` repeats each (route, mode, shape) cell within a round.

Every call is streamed so first-token and decode time are measured, not
inferred. Results land in `results.jsonl` under `--out`, one row per call, and
feed the per-host profile in the sweep report.
