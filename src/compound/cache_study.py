"""Controlled provider cache experiments with explicit prime/wait/probe phases."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from compound import serving_metrics as sm

REUSE = ("exact", "prefix", "none")


def plan(specs, shapes, reuse, delays, concurrency, trials):
    if not specs or any(s.kind not in ("openrouter", "doubleword") for s in specs):
        raise ValueError("cache-study supports OpenRouter and Doubleword routes")
    if len({s.label for s in specs}) != len(specs):
        raise ValueError("duplicate provider routes")
    if not shapes or any(
        not isinstance(s, dict)
        or not isinstance(s.get("messages"), list)
        or not s["messages"]
        or any(not isinstance(m, dict) for m in s["messages"])
        for s in shapes.values()
    ):
        raise ValueError("each shape needs a nonempty messages list")
    if not reuse or any(r not in REUSE for r in reuse) or len(set(reuse)) != len(reuse):
        raise ValueError("reuse must contain unique values from exact,prefix,none")
    if not delays or any(not math.isfinite(d) or d < 0 for d in delays):
        raise ValueError("delays must be finite, nonnegative seconds")
    if not concurrency or any(
        isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in concurrency
    ):
        raise ValueError("concurrency must contain positive integers")
    if len(set(delays)) != len(delays) or len(set(concurrency)) != len(concurrency):
        raise ValueError("duplicate delay or concurrency values")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if any(
        not isinstance(s.get("max_tokens"), int)
        or isinstance(s["max_tokens"], bool)
        or s["max_tokens"] < 1
        for s in shapes.values()
    ):
        raise ValueError("each shape must declare a positive max_tokens output budget")
    conditions = list(
        itertools.product(range(len(specs)), shapes, reuse, delays, concurrency, range(trials))
    )
    return conditions


def payload(shape, namespace, policy, probe):
    """Fixed-length IDs isolate trials/routes; suffix controls prefix reuse."""
    marker = (
        namespace
        if policy != "none" or probe == 0
        else hashlib.sha256(f"{namespace}:{probe}".encode()).hexdigest()[:32]
    )
    copied = json.loads(json.dumps(shape))
    copied["messages"] = sm.prepend_nonce(copied["messages"], f"[cache-study {marker}]\n")
    suffix = 0 if policy == "exact" else probe
    copied["messages"].append(
        {"role": "user", "content": f"Request ID: {suffix:08d}. Reply with OK."}
    )
    return copied


def run(
    specs,
    model,
    shapes,
    conditions,
    output,
    *,
    direct_model=None,
    seed=0,
    max_calls=1000,
    call=None,
    sleep=time.sleep,
    clock=time.monotonic,
):
    """Execute randomized independent trials; never retries or overlaps priming with probes."""
    models = {s.token: sm.model_for(s, model, direct_model) for s in specs}
    if any(s.cache_strategy == "explicit_marker" for s in specs) and not sm.cache_optin_enabled():
        raise ValueError("cache-study requires explicit cache markers; enable COMPOUND_DW_CACHE")
    count = sum(1 + c[4] for c in conditions)
    if count > max_calls:
        raise ValueError(f"planned {count} calls exceeds --max-calls {max_calls}")
    # Exclusive output creation prevents accidentally appending another experiment.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    call = call or sm.one_call
    order = list(conditions)
    random.Random(seed).shuffle(order)
    run_id = uuid.uuid4().hex
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": seed,
        "model": model,
        "route_models": models,
        "routes": [s.token for s in specs],
        "planned_calls": count,
        "conditions": order,
        "shape_sha256": hashlib.sha256(json.dumps(shapes, sort_keys=True).encode()).hexdigest(),
        "method": (
            "One priming call, then an idle delay measured from its completion, "
            "then a burst. Trials run serially in shuffled order. Route/trial prefixes "
            "are isolated. Prime failures remain recorded and probes still run. "
            "No retries; no dollar cap."
        ),
    }
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    path = output / "results.jsonl"
    with path.open("x") as f:

        def execute_trial(condition, namespace):
            route, shape_name, policy, delay, workers, trial = condition
            spec = specs[route]
            primed_at = None

            def execute(probe):
                shaped = payload(shapes[shape_name], namespace, policy, probe)
                started = clock()
                result = call(
                    spec,
                    models[spec.token],
                    sm.REASONING_OFF,
                    shape_name,
                    shaped,
                    trial + 1,
                    probe,
                    cache_mode=sm.CACHE_WARM,
                    temperature=0,
                )
                result.update(
                    experiment="cache-study",
                    run_id=run_id,
                    trial=trial,
                    reuse_policy=policy,
                    idle_s=delay,
                    concurrency=workers,
                    phase="prime" if probe == 0 else "probe",
                    cache_mode="prime" if probe == 0 else "probe",
                    prompt_sha256=hashlib.sha256(
                        json.dumps(shaped["messages"], sort_keys=True).encode()
                    ).hexdigest(),
                    actual_idle_s=None if primed_at is None else started - primed_at,
                )
                return result

            def save(result):
                f.write(json.dumps(result) + "\n")
                f.flush()

            save(execute(0))
            primed_at = clock()
            sleep(delay)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Submit the whole burst before collecting any results.
                futures = [pool.submit(execute, i + 1) for i in range(workers)]
                for future in futures:
                    save(future.result())

        for cell, condition in enumerate(order):
            namespace = hashlib.sha256(f"{run_id}:{cell}".encode()).hexdigest()[:32]
            execute_trial(condition, namespace)
    return path
