"""Run-scoped metered gateway for the agentic pilot (loopback only).

Conservative reservations precede every request. An ambiguous failed call keeps
its reservation, because the provider may still have processed and billed it.
No automatic retries. This is a spend guard, not a provider billing guarantee.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from compound.cache_policy import is_marked, mark_cache_prefix, needs_marker

VALIDATION_WAIVER_SOURCE = (
    "https://openrouter.zendesk.com/hc/en-us/articles/"
    "51693138951451-Was-I-charged-for-a-failed-errored-or-empty-response-Zero-Completion-Insurance"
)


def is_context_rejection(status, body):
    if status != 400:
        return False
    try:
        outer = json.loads(body)["error"]
        inner = json.loads(outer["metadata"]["raw"])["error"]
        return (
            outer["code"] == 400
            and inner["code"] == 400
            and inner["status"] == "INVALID_ARGUMENT"
            and inner["message"].startswith(
                "The input token count exceeds the maximum number of tokens allowed"
            )
        )
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


class ProviderResponseError(RuntimeError):
    def __init__(self, code, message, role="agent"):
        super().__init__(message)
        self.code, self.role = int(code or 502), role


def check_response_errors(raw, role="agent"):
    """A 200 envelope may still contain an upstream availability failure."""
    error = raw.get("error")
    for choice in raw.get("choices") or []:
        if choice.get("error") or choice.get("finish_reason") == "error":
            error = choice.get("error") or {"message": "upstream generation failed"}
            break
    if error:
        raise ProviderResponseError(error.get("code", 502), error.get("message", "error"), role)


class SpendGuard:
    def __init__(self, path: Path, limit: float, role_limits=None):
        if not 0 < limit <= 12:
            raise ValueError("inference allowance must be within $12")
        self.path, self.limit = path, limit
        self.lock = threading.Lock()
        self.entries = json.loads(path.read_text()) if path.exists() else []
        self.role_limits = role_limits or {}

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.entries, indent=2) + "\n")
        temporary.replace(self.path)

    def remaining(self, episode, role, episode_limit):
        with self.lock:
            used = sum(
                e["charged_or_reserved_usd"]
                for e in self.entries
                if e["episode_id"] == episode and e["role"] == role
            )
            role_remaining = self.role_limits.get(role, self.limit) - sum(
                e["charged_or_reserved_usd"] for e in self.entries if e["role"] == role
            )
            return min(
                role_remaining,
                episode_limit - used,
                self.limit - sum(e["charged_or_reserved_usd"] for e in self.entries),
            )

    def reserve(self, amount, episode, role, episode_limit=None, attempt_id=None):
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("invalid reservation")
        with self.lock:
            if amount + sum(
                e["charged_or_reserved_usd"] for e in self.entries if e["role"] == role
            ) > self.role_limits.get(role, self.limit):
                raise ValueError("study role budget exhausted")
            if (
                episode_limit is not None
                and amount
                + sum(
                    e["charged_or_reserved_usd"]
                    for e in self.entries
                    if e["episode_id"] == episode and e["role"] == role
                )
                > episode_limit
            ):
                raise ValueError("episode budget exhausted")
            if sum(e["charged_or_reserved_usd"] for e in self.entries) + amount > self.limit:
                raise ValueError("study inference budget exhausted")
            index = len(self.entries)
            self.entries.append(
                {
                    "episode_id": episode,
                    "attempt_id": attempt_id,
                    "role": role,
                    "reservation_usd": amount,
                    "charged_or_reserved_usd": amount,
                    "settled": False,
                }
            )
            self.save()
            return index

    def settle(self, index, cost):
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("invalid settlement")
        with self.lock:
            e = self.entries[index]
            e.update(charged_or_reserved_usd=cost, settled=True)
            self.save()


def responses_body(body, tier):
    """Translate ordinary tool conversations to stateless Open Responses input."""
    inputs = []
    for m in body["messages"]:
        if m["role"] == "tool":
            inputs.append(
                {
                    "type": "function_call_output",
                    "call_id": m["tool_call_id"],
                    "output": m.get("content") or "",
                }
            )
            continue
        content = m.get("content")
        if content:
            if isinstance(content, list):
                content = "\n".join(x.get("text", "") for x in content)
            inputs.append(
                {
                    "role": m["role"],
                    "content": [
                        {
                            "type": "input_text" if m["role"] != "assistant" else "output_text",
                            "text": content,
                        }
                    ],
                }
            )
        for call in m.get("tool_calls") or []:
            inputs.append(
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
            )
    # Doubleword supports prompt caching on Chat Completions and Messages,
    # not on this Responses endpoint. Markers here cannot enable it.
    result = {
        "model": body["model"],
        "input": inputs,
        "service_tier": tier,
        "max_output_tokens": body["max_tokens"],
        "reasoning": {"effort": "medium"},
    }
    if body.get("tools"):
        result["tools"] = [{"type": "function", **t["function"]} for t in body["tools"]]
    return result


def chat_response(raw):
    text, calls = [], []
    for item in raw.get("output", []):
        if item.get("type") == "function_call":
            calls.append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
            )
        for block in item.get("content") or []:
            if block.get("type") == "output_text":
                text.append(block["text"])
    usage = raw.get("usage") or {}
    return {
        "id": raw.get("id"),
        "object": "chat.completion",
        "model": raw.get("model"),
        "service_tier": raw.get("service_tier"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": "\n".join(text),
                    **({"tool_calls": calls} if calls else {}),
                },
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "prompt_tokens_details": usage.get("input_tokens_details") or {},
            "completion_tokens_details": usage.get("output_tokens_details") or {},
        },
    }


# Input/cache-read/output dollars per million; upper rates cover long context.
#: A 5 minute cache write bills at 1.25x standard input (Doubleword docs).
CACHE_WRITE_MULTIPLIERS = {"5m": 1.25, "1h": 2.0}
# Compatibility name retained for downstream imports and audit checks.
CACHE_WRITE_MULTIPLIER = CACHE_WRITE_MULTIPLIERS["5m"]

RATES = {
    ("gpt-astra", "standard"): (10, 1, 50),
    ("gpt-astra", "flex"): (5, 0.5, 25),
    ("glm-flash-dw", "standard"): (0.15, 0.03, 0.50),
    ("glm-flash-dw", "flex"): (0.11, 0.02, 0.38),
    ("gpt-sol", "standard"): (4, 0.4, 15),
    ("gpt-sol", "flex"): (2, 0.2, 7.5),
    ("gemini-vertex", "standard"): (0.75, 0.075, 3.75),
    ("gemini-vertex", "flex"): (0.375, 0.0375, 1.875),
    ("gemini-studio", "standard"): (0.75, 0.075, 3.75),
    ("gemini-studio", "flex"): (0.375, 0.0375, 1.875),
    ("deepseek-dw", "standard"): (0.09, 0.02, 0.18),
    ("deepseek-dw", "flex"): (0.07, 0.01, 0.14),
    ("glm-dw", "standard"): (1.4, 0.28, 4.4),
    ("glm-dw", "flex"): (1.05, 0.21, 3.3),
    # Peak rates. This model is discounted off-peak, so reserving at peak stays
    # conservative; billed cost is taken from the provider response, not from here.
    ("ds-sim", "standard"): (0.30, 0.006, 1.20),
    ("ds-sim", "flex"): (0.30, 0.006, 1.20),
}

# The compatibility table above was written for these exact model identities.
# Reusing a route id for a replacement model must not silently reuse its prices.
LEGACY_RATE_MODELS = {
    "gpt-astra": "openai/gpt-6-astra",
    "glm-flash-dw": "zai-org/GLM-5.3-Flash",
    "gpt-sol": "openai/gpt-5.6-sol",
    "gemini-vertex": "google/gemini-3.8-flash",
    "gemini-studio": "google/gemini-3.8-flash",
    "deepseek-dw": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "glm-dw": "zai-org/GLM-5.3",
    "ds-sim": "deepseek/deepseek-v4.1-flash",
}


def _declared_rates(model, tier):
    """Return a model-bound rate tuple, or ``None`` for legacy specs.

    New specs declare rates beside the exact provider model so replacing a
    model cannot inherit an older route id's price. Values are USD per million
    tokens. ``rate_model`` is optional only because the enclosing model already
    supplies the identity; when present it must match exactly.
    """
    cards = model.get("rates") or model.get("rate_card")
    if cards is None:
        return None
    declared_model = model.get("rate_model")
    if isinstance(cards, dict) and "model" in cards:
        declared_model = cards["model"]
    if declared_model is not None and declared_model != model.get("model"):
        raise ValueError("rate card model does not match requested model")
    card = cards.get(tier) if isinstance(cards, dict) else None
    if not isinstance(card, dict):
        raise ValueError(f"missing declared {tier} rate card")
    try:
        result = (
            float(card["input_per_million"]),
            float(card["cache_read_per_million"]),
            float(card["output_per_million"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid declared rate card") from exc
    if any(not math.isfinite(value) or value < 0 for value in result):
        raise ValueError("invalid declared rate card")
    return result


def rates_for(model, route, tier):
    declared = _declared_rates(model, tier)
    if declared is not None:
        return declared, "model_declared"
    expected_model = LEGACY_RATE_MODELS.get(route)
    if expected_model is not None and model.get("model") not in (None, expected_model):
        raise ValueError(
            f"route {route} changed model identity; declare rates for {model.get('model')}"
        )
    try:
        return RATES[route, tier], "legacy_route_table"
    except KeyError as exc:
        raise ValueError(f"missing rate card for {route}/{tier}") from exc


def missing_rates(spec):
    """Routes in the spec with no price entry, which would fail mid-run otherwise."""
    models = list(spec["models"]) + list(spec.get("auxiliary_models", []))
    missing = []
    for model in models:
        for tier in ("standard", "flex"):
            try:
                rates_for(model, model["id"], tier)
            except ValueError:
                missing.append(f"{model['id']}/{tier}")
    return sorted(set(missing))


class Gateway:
    def __init__(self, spec, study, directory, limit=12, stop_event=None):
        self.models = {m["id"]: m for m in spec["models"]}
        self.models.update({m["id"]: m for m in spec.get("auxiliary_models", [])})
        self.episodes = {e["episode_id"]: e for e in study["episodes"]}
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.guard = SpendGuard(
            self.directory / "spend.json", limit, spec["controls"].get("role_budget_usd")
        )
        self.calls_lock = threading.Lock()
        self.max_output = spec["controls"]["max_output_tokens"]
        self.spec = spec
        self.reasoning_details = {}
        self.input_evidence = {}
        self.stop_event = stop_event or threading.Event()
        # The runner sets a fresh opaque id before each task attempt. Calls take
        # one snapshot, so later mutations cannot relabel an in-flight request.
        self.active_attempts = {}

    def input_bound(self, eid, role, payload, attempt_id=None):
        canonical = json.dumps(
            {k: v for k, v in payload.items() if k not in {"max_tokens", "max_output_tokens"}},
            sort_keys=True,
        ).encode()
        bound = len(canonical) + 4096
        state_key = (eid, role) if attempt_id is None else (eid, attempt_id, role)
        prior = self.input_evidence.get(state_key)
        if (
            self.spec["controls"].get("reservation_policy") == "observed_tokens_plus_changed_bytes"
            and prior
        ):
            old, tokens = prior
            prefix = 0
            while prefix < min(len(old), len(canonical)) and old[prefix] == canonical[prefix]:
                prefix += 1
            suffix = 0
            while (
                suffix < min(len(old), len(canonical)) - prefix
                and old[len(old) - suffix - 1] == canonical[len(canonical) - suffix - 1]
            ):
                suffix += 1
            # Never subtract tokens for deleted text. Charge every new/changed byte
            # as a token, plus 4096 for framing and boundary retokenization.
            bound = min(bound, tokens + len(canonical) - prefix - suffix + 4096)
        return bound, canonical

    def call(self, eid, role, body):
        if self.stop_event.is_set():
            raise ValueError("study stop requested; refusing provider call")
        attempt_id = self.active_attempts.get(eid)
        episode = self.episodes[eid]
        route = (
            episode["route"]
            if role == "agent"
            else self.spec["controls"].get("simulator_route", "gpt-sol")
        )
        tier = episode["tier"] if role == "agent" else "standard"
        model = self.models[route]
        (rate_in, rate_cache, rate_out), rate_source = rates_for(model, route, tier)
        cache_ttl = self.spec["controls"].get("cache_ttl", "5m")
        if cache_ttl not in CACHE_WRITE_MULTIPLIERS:
            raise ValueError("cache_ttl must be 5m or 1h")
        cache_write_multiplier = CACHE_WRITE_MULTIPLIERS[cache_ttl]
        requested_output = int(
            body.get("max_tokens") or body.get("max_completion_tokens") or self.max_output
        )
        # The harness asks for its own cap; the spec control is the ceiling. Record
        # when the ceiling binds, so a reply cut short by the study's own control is
        # never scored as a model that simply answered badly.
        output = min(requested_output, self.max_output)
        if output < 1:
            raise ValueError("invalid output token limit")
        payload = {"model": model["model"], "messages": body["messages"], "max_tokens": output}
        # Some upstream harness message types discard opaque provider fields.
        # Restore only details returned for this episode and these exact call IDs.
        for message in payload["messages"]:
            ids = tuple(t["id"] for t in message.get("tool_calls") or [])
            reasoning_key = (eid, role, ids) if attempt_id is None else (eid, attempt_id, role, ids)
            details = self.reasoning_details.get(reasoning_key) if ids else None
            if details and not message.get("reasoning_details"):
                message["reasoning_details"] = details
        if body.get("tools"):
            payload["tools"] = body["tools"]
        if model["provider"] == "openrouter":
            effort = self.spec["controls"].get("reasoning_effort")
            payload.update(
                service_tier="default" if tier == "standard" else "flex",
                provider={
                    "only": [model[tier + "_endpoint"]],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                usage={"include": True},
            )
            if effort:
                payload["reasoning"] = {"effort": effort}
            base, key = "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"
        else:
            # Chat completions, not Responses: Doubleword's Responses endpoint
            # accepts a cache marker and never reports a hit (measured
            # 2026-09-15), so an agent loop there re-reads its whole context at
            # full price every turn. Chat completions caches at ~99% with the
            # marker. The cost of the switch is that chat completions does not
            # echo service_tier. Such calls remain explicitly unverified here;
            # a separate reconciliation may later add real billing evidence.
            payload["messages"] = mark_cache_prefix(payload["messages"], ttl=cache_ttl)
            payload["service_tier"] = "priority" if tier == "standard" else "flex"
            # Chat completions takes a flat reasoning_effort; the Responses API
            # took a nested reasoning.effort. Dropping it on the switch would have
            # silently run the study on the model's default.
            effort = self.spec["controls"].get("reasoning_effort")
            if effort:
                payload["reasoning_effort"] = effort
            base, key = "https://api.doubleword.ai/v1/chat/completions", "DOUBLEWORD_API_KEY"
        endpoint_identity = model.get(tier + "_endpoint")
        if needs_marker(base) and not is_marked(payload):
            raise ValueError(f"refusing to send an uncached request to {base}")
        cache_requested = is_marked(payload)
        wire = json.dumps(payload).encode()
        input_bound, input_evidence = self.input_bound(eid, role, payload, attempt_id=attempt_id)
        episode_limit = (
            episode.get("budget_usd")
            if role == "agent"
            else self.spec["controls"].get("simulator_episode_budget_usd")
        )
        if episode_limit is not None:
            remaining = self.guard.remaining(eid, role, episode_limit)
            input_reservation = input_bound * rate_in * cache_write_multiplier / 1e6
            affordable = math.floor((remaining - input_reservation) * 1e6 / rate_out)
            legacy_shrink = self.spec["controls"].get("budget_output_policy") == "shrink_legacy"
            if affordable < output and not legacy_shrink:
                event = {
                    "episode_id": eid,
                    "attempt_id": attempt_id,
                    "role": role,
                    "reason": "full fixed output cap is unaffordable",
                    "remaining_usd": remaining,
                    "input_reservation_usd": input_reservation,
                    "required_output_tokens": output,
                    "affordable_output_tokens": max(affordable, 0),
                }
                with self.calls_lock, (self.directory / "budget-stops.jsonl").open("a") as f:
                    f.write(json.dumps(event) + "\n")
                raise ValueError("episode budget exhausted: fixed output cap unaffordable")
            if affordable < 1:
                raise ValueError("episode budget exhausted")
            if legacy_shrink:
                output = min(output, affordable)
            # Both routes speak chat completions now, so the cap is always max_tokens.
            payload["max_tokens"] = output
            wire = json.dumps(payload).encode()
        # UTF-8 bytes + protocol allowance conservatively exceed ordinary token
        # counts; all input is reserved at cache-write price, with output capped.
        reservation = (input_bound * rate_in * cache_write_multiplier + output * rate_out) / 1e6
        index = self.guard.reserve(reservation, eid, role, episode_limit, attempt_id=attempt_id)
        start = time.monotonic()
        row = {
            "started_at": datetime.now(UTC).isoformat(),
            "episode_id": eid,
            "attempt_id": attempt_id,
            "role": role,
            "route": route,
            "requested_model": model["model"],
            "requested_endpoint": endpoint_identity,
            "request_url": base,
            "requested_tier": tier,
            "cache_requested": cache_requested,
            "rate_source": rate_source,
            "reservation_usd": reservation,
            "input_token_bound": input_bound,
            "requested_output_tokens": requested_output,
            "max_output_tokens": output,
            "output_cap_applied": output < requested_output,
            "request_sha256": __import__("hashlib").sha256(wire).hexdigest(),
        }
        try:
            request = urllib.request.Request(
                base,
                data=wire,
                headers={
                    "Authorization": "Bearer " + os.environ[key],
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = json.load(response)
            reply = raw
            # A successful HTTP envelope can still carry a provider error. Check
            # it before settling or doing any further response processing; a 402
            # reaches the shared stop event through the exception path below.
            check_response_errors(raw, role)
            usage = reply.get("usage") or {}
            if isinstance(usage.get("prompt_tokens"), int) and usage["prompt_tokens"] > 0:
                state_key = (eid, role) if attempt_id is None else (eid, attempt_id, role)
                self.input_evidence[state_key] = (input_evidence, usage["prompt_tokens"])
            cost = usage.get("cost")
            kind = "reported"
            if cost is None and model["provider"] == "doubleword" and raw.get("usage"):
                prompt = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                if prompt is None or completion is None:
                    # Missing evidence is unknown cost, never zero. Leaving it None
                    # keeps the reservation held instead of releasing it as free.
                    cost = None
                elif (
                    not isinstance(prompt, int)
                    or isinstance(prompt, bool)
                    or prompt < 0
                    or not isinstance(completion, int)
                    or isinstance(completion, bool)
                    or completion < 0
                ):
                    raise ValueError("invalid provider token counts for derived cost")
                else:
                    read = usage.get("cache_read_input_tokens")
                    if read is None:
                        read = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                    written = usage.get("cache_creation_input_tokens") or 0
                    if (
                        any(
                            not isinstance(value, int) or isinstance(value, bool) or value < 0
                            for value in (read, written)
                        )
                        or read + written > prompt
                    ):
                        raise ValueError("invalid provider token counts for derived cost")
                    # Cache-write pricing follows the declared 5m/1h TTL.
                    cost = (
                        max(prompt - read - written, 0) * rate_in
                        + read * rate_cache
                        + written * rate_in * cache_write_multiplier
                        + completion * rate_out
                    ) / 1e6
                    kind = "derived_from_rate_card"
            if cost is not None:
                cost = float(cost)
                if cost < 0 or not __import__("math").isfinite(cost):
                    raise ValueError("invalid provider cost")
                self.guard.settle(index, cost)
                usage["cost"] = cost
            row.update(
                status=200,
                usage=usage,
                cache_requested=cache_requested,
                cost_usd=cost,
                cost_kind=kind,
                served_tier=raw.get("service_tier"),
                served_provider=raw.get("provider"),
                served_model=raw.get("model"),
                response_status=raw.get("status"),
                response_id=raw.get("id"),
            )
            # A reply stopped at the token ceiling is truncated output, not a wrong
            # answer. Without this the two are indistinguishable in the outcomes.
            finish_reason = ((reply.get("choices") or [{}])[0] or {}).get("finish_reason")
            row["finish_reason"] = finish_reason
            row["output_truncated"] = finish_reason == "length"
            if raw.get("status") in ("failed", "cancelled", "incomplete"):
                raise ValueError("Responses request did not complete: " + str(raw.get("status")))
            # A tier that is echoed back must match: a contradicting echo means we
            # measured the wrong thing and is always fatal. Some endpoints never
            # echo the field at all, including Doubleword chat completions, which
            # is the only endpoint there that caches. For those the tier cannot be
            # confirmed per call, so it is recorded as unverified. A requested
            # tier or a locally derived price is not evidence that it was served.
            expected = {"flex"} if tier == "flex" else {"default", "priority"}
            served = raw.get("service_tier")
            if served is not None and served not in expected:
                raise ValueError("provider served a different tier than requested")
            row["tier_confirmed"] = served is not None
            row["tier_evidence"] = "response_echo" if served is not None else "unverified"
            message = reply["choices"][0]["message"]
            ids = tuple(t["id"] for t in message.get("tool_calls") or [])
            if ids and message.get("reasoning_details"):
                reasoning_key = (
                    (eid, role, ids) if attempt_id is None else (eid, attempt_id, role, ids)
                )
                self.reasoning_details[reasoning_key] = message["reasoning_details"]
            return reply
        except Exception as exc:
            if getattr(exc, "code", None) == 402:
                self.stop_event.set()
            if isinstance(exc, (ValueError, KeyError, TypeError, IndexError)):
                row["failure_reason"] = "invalid_provider_evidence"
                self.stop_event.set()
            row.update(
                status=getattr(exc, "code", 502),
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
            )
            if isinstance(exc, urllib.error.HTTPError):
                row["error_body"] = exc.read(4000).decode("utf-8", errors="replace")
                if model["provider"] == "openrouter" and is_context_rejection(
                    exc.code, row["error_body"]
                ):
                    self.guard.settle(index, 0)
                    row.update(
                        cost_usd=0,
                        cost_kind="documented_validation_waiver",
                        cost_source=VALIDATION_WAIVER_SOURCE,
                        failure_reason="context_limit",
                    )
            raise
        finally:
            row["duration_s"] = time.monotonic() - start
            row["finished_at"] = datetime.now(UTC).isoformat()
            with self.calls_lock, (self.directory / "calls.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

    def serve(self):
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                try:
                    parts = self.path.strip("/").split("/")
                    eid, role = parts[:2]
                    if role not in ("agent", "auxiliary") or parts[2:] != [
                        "v1",
                        "chat",
                        "completions",
                    ]:
                        raise ValueError("unknown gateway route")
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    if body.get("stream"):
                        raise ValueError("pilot records completion latency; use stream=false")
                    result = gateway.call(eid, role, body)
                    status = 200
                except Exception as exc:
                    result = {"error": {"message": str(exc), "type": type(exc).__name__}}
                    status = (
                        429
                        if "budget exhausted" in str(exc)
                        else int(getattr(exc, "code", 502) or 502)
                    )
                data = json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server
