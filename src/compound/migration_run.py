"""Private, staged migration study. No calls without --go; no product gate verdicts.

This path deliberately uses the audited migration I/O and grounded rubric rather
than weakening the product pipeline's human-calibration requirement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from compound.migration_grading import (
    accepted_grade,
    candidate_messages,
    case_from_trace,
    judge_messages,
    judge_response_format,
    parse_candidate_output,
    parse_judge_grade,
    project_candidate_output,
    quality_score,
    synthetic_negative_controls,
)
from compound.migration_io import (
    HTTPCallError,
    InferenceService,
    Ledger,
    SafetyStop,
    UnresolvedCall,
)

FINAL_JUDGE_WORKERS = 2
MAX_FINAL_CASE_WINDOW = 4
FINAL_DRAIN_TIMEOUT_S = 300


class _FinalStageStopped(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.chmod(0o600)
    temp.replace(path)


def prepare(source, root, catalog_dir):
    """Only preparation reads the whole export; optimization gets train/val files."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "spec.json").exists():
        raise ValueError("Refusing to replace an existing study specification")
    raw = source.read_bytes()
    traces = [json.loads(line) for line in raw.splitlines() if line.strip()]
    cases = [case_from_trace(t) for t in traces]
    if len(cases) != 200 or len({c.case_id for c in cases}) != 200:
        raise ValueError("Expected the audited 200 distinct traces")
    if len({digest(t["steps"][0]["input"]) for t in traces}) != 200:
        raise ValueError("Duplicate inputs require grouped splitting")
    by_role = {role: [] for role in ("mobile", "acquisition")}
    for trace, case in zip(traces, cases, strict=True):
        by_role[case.role_id].append((trace, case))
    parts = {p: [] for p in ("train", "validation", "test")}
    for role, items in by_role.items():
        # Independent of incumbent scores or recommendation. Source audit found
        # no repeated applicant labels, exact inputs or >=.8 five-gram duplicates.
        items.sort(key=lambda x: digest(["migration-v1-seed-7419", x[1].case_id]))
        ntrain, nval = (32, 22) if role == "mobile" else (28, 18)
        for part, chosen in zip(
            parts, (items[:ntrain], items[ntrain:ntrain+nval], items[ntrain+nval:]),
            strict=True,
        ):
            parts[part].extend(t for t, _ in chosen)
    for name, rows in parts.items():
        rows.sort(key=lambda t: digest(["execution-order-7419", t["trace_id"]]))
        save(root / f"{name}.json", rows)
    base_or = "https://openrouter.ai/api/v1"
    dw = "deepseek-ai/DeepSeek-V4.1-Flash"
    ort = "deepseek/deepseek-v4.1-flash"
    routes = [
        dict(id="doubleword-realtime", model=dw, api="chat_completions",
             base_url="https://api.doubleword.ai/v1", api_key_env="DOUBLEWORD_API_KEY",
             cache_marker=True),
        dict(id="doubleword-async", model=dw, api="responses",
             base_url="https://api.doubleword.ai/v1", api_key_env="DOUBLEWORD_API_KEY",
             service_tier="flex", background=True),
    ]
    for label, pin in (("auto", None), ("modal", "modal"), ("novita", "novita/fp8")):
        provider = {"require_parameters": True}
        if pin:
            provider.update(only=[pin], allow_fallbacks=False)
        routes.append(dict(id="openrouter-"+label, model=ort, api="chat_completions",
                           base_url=base_or, api_key_env="OPENROUTER_API_KEY",
                           provider=provider))
    judges = []
    for name, model, pin in (("glm", "z-ai/glm-5.3-flash", "baseten/fp8"),
                             ("sol", "openai/gpt-5.6-sol", "openai")):
        judges.append(dict(id="judge-"+name, model=model, api="chat_completions",
                           base_url=base_or, api_key_env="OPENROUTER_API_KEY",
                           expected_upstream="BaseTen" if name == "glm" else "OpenAI",
                           provider=dict(only=[pin], allow_fallbacks=False,
                                         require_parameters=True)))
    judges[0].pop("expected_upstream")
    judges[0]["expected_upstreams"] = ["BaseTen", "Sail Research", "Phala", "Reka"]
    judges[0]["provider"].update(
        only=["baseten/fp8", "sail-research/fp8", "phala/fp8", "reka/fp8"],
        allow_fallbacks=True)
    rates = {
        "doubleword-realtime": dict(input_per_million=.15, output_per_million=.60,
                                    cached_input_per_million=.003,
                                    cache_write_per_million=.1875),
        "doubleword-async": dict(input_per_million=.12, output_per_million=.48),
        "openrouter-auto": dict(input_per_million=.375, output_per_million=1.5),
        "openrouter-modal": dict(input_per_million=.30, output_per_million=1.20,
                                 cached_input_per_million=.03),
        "openrouter-novita": dict(input_per_million=.30, output_per_million=1.20,
                                  cached_input_per_million=.006),
        "judge-glm": dict(input_per_million=.15, output_per_million=.50,
                          cached_input_per_million=.03),
        "judge-sol": dict(input_per_million=2., output_per_million=10.,
                          cached_input_per_million=.2),
    }
    spec = dict(
        version=7, source_sha256=hashlib.sha256(raw).hexdigest(),
        source_revision="6be33b470ae429e71862b9d0fbd4cee2610c6f3f",
        total_cap_usd=25., inference_cap_usd=23.5, compute_reserve_usd=1.5,
        stage_caps=dict(smoke=2.25, baseline=3.5, gepa=4., final=12., reserve=1.75),
        stop_at="2026-09-14T08:20:00+00:00", seed=7419,
        routes=routes, judges=judges, rates=rates,
        candidate_params=dict(max_tokens=4096, reasoning_effort="medium",
                              response_format={"type": "json_object"}),
        judge_params=dict(max_tokens=4096, reasoning_effort="medium",
                          response_format=judge_response_format()),
        retries=0, judge_max_attempts=3, judge_concurrency=2,
        concurrency=5, per_call_deadline_s=300,
        smoke_gate="At least one valid output per route; every valid answer and reference "
                   "graded by both judges; exact expected critical flags on negative controls. "
                   "Invalid candidate outputs remain failures, never retried or excluded.",
        deadline_metrics_s=[10, 30, 60, 300],
        acceptance_score=.75, noninferiority_margin=.05,
        critical_error_rule="No observed increase; uncertain rare-event risk remains",
        primary_judge="sol", secondary_judge="glm", human_calibrated=False,
        interpretation="Reconstructed shared-field comparison; exploratory LLM judgment",
        baseline_selection="GLM quality within .05 of Gemini, >=95% valid delivery, "
                           "then lowest mean route cost; if none, highest mean quality "
                           "for optimization only, not a qualified provider",
        final_comparison="100 sealed cases: recorded Gemini vs selected route "
                         "unoptimized vs frozen optimized, both judges independently",
        judge_failure_policy="Missing assessment; never silently omitted or scored as model fail",
        model_failure_policy="Zero usable quality and not acceptable; cost retained",
        split={k: dict(count=len(v), sha256=digest(v),
                       ids=[x["trace_id"] for x in v]) for k, v in parts.items()},
        catalog_snapshot_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in catalog_dir.glob("*endpoints.json")},
    )
    save(root / "spec.json", spec)
    save(root / "manifest.json", dict(spec_sha256=digest(spec)))
    print(json.dumps({"prepared": str(root), "spec_sha256": digest(spec),
                      "counts": {p: len(x) for p, x in parts.items()}}))


class Study:
    def __init__(self, root):
        self.root = root
        self.spec = json.loads((root / "spec.json").read_text())
        if digest(self.spec) != json.loads((root / "manifest.json").read_text())["spec_sha256"]:
            raise ValueError("Study spec fingerprint changed")
        source_manifest = root / "source-hashes.json"
        if source_manifest.exists():
            for filename, expected in json.loads(source_manifest.read_text()).items():
                if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                    raise ValueError("Executed source differs from frozen source manifest")
        self.ledger = Ledger(root / "ledger.sqlite", self.spec["inference_cap_usd"],
                             self.spec["stage_caps"], artifact_dir=root / "calls")
        self.service = InferenceService(self.ledger, self.spec["rates"], timeout_s=240,
                                        poll_interval_s=2)
        self.routes = {r["id"]: r for r in self.spec["routes"]}
        self.judges = {r["id"]: r for r in self.spec["judges"]}
        self.judge_lock = threading.Semaphore(2)
        # Judges share a provider pool but not its rate limits. A shared-pool judge can
        # be refused with HTTP 429 at a concurrency a pinned first-party judge sustains,
        # so each judge gets its own declared ceiling.
        _limits = self.spec.get("judge_concurrency_by_judge") or {}
        _default = self.spec.get("judge_concurrency", 2)
        self.judge_locks = {
            judge_id: threading.Semaphore(int(_limits.get(judge_id, _default)))
            for judge_id in self.judges
        }
        self._call_context = threading.local()
        self._active_final_stop_event = None

    def check_time(self):
        if (datetime.fromisoformat(self.spec["stop_at"])-datetime.now(UTC)).total_seconds() < 300:
            raise TimeoutError("Frozen study dispatch deadline reached")

    def cases(self, partition):
        raw = json.loads((self.root / f"{partition}.json").read_text())
        if digest(raw) != self.spec["split"][partition]["sha256"]:
            raise ValueError("Partition changed")
        return [case_from_trace(t) for t in raw]

    def _call(self, route, messages, params, call_id, stage):
        self.check_time()
        if stage == "final":
            context = getattr(self, "_call_context", None)
            stop_event = (getattr(context, "final_stop_event", None) if context else None)
            stop_event = stop_event or getattr(self, "_active_final_stop_event", None)
            if stop_event is not None and stop_event.is_set():
                raise _FinalStageStopped("Final-stage provider dispatch aborted")
        bound = (self.spec.get("output_token_bounds") or {}).get(route["id"])
        # Only declared when a route needs a ceiling above its own max_tokens, so
        # the common path keeps the historical call signature.
        extra = {"output_token_bound": bound} if bound is not None else {}
        result = self.service.call(route, messages, params, call_id=call_id, stage=stage,
                                   poll_deadline_s=self.spec["per_call_deadline_s"], **extra)
        # Broker pins are supported by catalog + request; the echo verifies host.
        expected = route.get("expected_upstreams") or route.get("expected_upstream") or {
            "openrouter-modal": "modal", "openrouter-novita": "novita",
            "judge-glm": "deepinfra", "judge-sol": "openai"}.get(route["id"])
        def normalized(s):
            return "".join(c for c in str(s).lower() if c.isalnum())
        allowed = expected if isinstance(expected, list) else [expected]
        if expected and normalized(result.upstream) not in {normalized(x) for x in allowed}:
            message = f"Unexpected or absent upstream for {route['id']}"
            self.ledger.update(call_id, status="safety_stop", error=message)
            self.ledger.record_event(call_id, "safety_stop", message)
            raise SafetyStop(message)
        requested_model = route["model"].split("/")[-1].lower()
        served = str(result.raw.get("model", "")).split("/")[-1].lower()
        if not served or not served.startswith(requested_model):
            message = f"Unexpected resolved model for {route['id']}"
            self.ledger.update(call_id, status="safety_stop", error=message)
            self.ledger.record_event(call_id, "safety_stop", message)
            raise SafetyStop(message)
        return result

    def generate(self, case, route, methodology, stage, replicate="0"):
        call_id = (f"candidate/{stage}/{case.case_id}/{route['id']}/"
                   f"{digest(methodology)}/{replicate}")
        path = self.root / "outcomes" / (digest(call_id)+".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("reason") == "deadline_exceeded":
                return saved
            if (saved.get("status") == "valid" and
                    saved.get("call", {}).get("latency_s", 0) >= self.spec["per_call_deadline_s"]):
                raise RuntimeError("Stored late candidate requires the offline deadline repair")
            recorded = self.ledger.get(call_id)
            if not (recorded and recorded["status"] in ("submitted", "polling")
                    and recorded.get("response_id")):
                return saved
            save(path.with_suffix(".pre-resume.json"), saved)
        messages = candidate_messages(case, methodology)
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        content = system
        if route.get("cache_marker"):
            # Content parts are named differently by the two APIs. Chat Completions takes
            # "text"; the Responses API takes "input_text" and rejects the request outright
            # if given the other name, which looks identical to the cache marker itself
            # being unsupported.
            part_type = "input_text" if route["api"] == "responses" else "text"
            content = [dict(type=part_type, text=system,
                            cache_control=dict(type="ephemeral", ttl="5m"))]
        messages = [dict(role="system", content=content),
                    *[m for m in messages if m["role"] != "system"]]
        row = dict(case_id=case.case_id, role=case.role_id, route=route["id"],
                   stage=stage, methodology_sha256=digest(methodology), call_id=call_id,
                   # A route is a provider, a tier and an API. Caching, latency and price
                   # can each depend on the API alone, so the three are recorded together
                   # and never collapsed into the route name.
                   serving=dict(api=route["api"],
                                service_tier=route.get("service_tier"),
                                background=bool(route.get("background")),
                                cache_requested=bool(route.get("cache_marker"))))
        try:
            params = route.get("candidate_params") or self.spec["candidate_params"]
            result = self._call(route, messages, params, call_id, stage)
            row.update(call=result.as_dict())
            if row["call"].get("latency_s", 0) >= self.spec["per_call_deadline_s"]:
                row.update(status="delivery_failure", reason="deadline_exceeded",
                           deadline_s=self.spec["per_call_deadline_s"])
                save(path, row)
                return row
            finish = (result.raw.get("choices") or [{}])[0].get("finish_reason")
            truncated = finish == "length" or result.raw.get("status") == "incomplete"
            try:
                if truncated:
                    raise ValueError("truncated generation")
                parsed = parse_candidate_output(case, result.output_text)
                row.update(status="valid", strict_status="valid",
                           contract="strict", output=parsed.to_dict())
            except ValueError as exc:
                row.update(status="invalid_output", strict_status="invalid_output",
                           contract="strict", reason=str(exc))
                # A projection declared in the frozen spec before any paid call. It drops
                # unknown top-level keys only; it never repairs values, never invents
                # missing fields, and never rescues a truncated generation.
                if (self.spec.get("contract_projection") == "drop_unknown_top_level_keys"
                        and not truncated):
                    try:
                        projected, dropped = project_candidate_output(
                            case, result.output_text)
                    except ValueError:
                        pass
                    else:
                        if dropped:
                            row.update(status="valid", contract="tolerant",
                                       dropped_top_level_keys=list(dropped),
                                       output=projected.to_dict())
        except (HTTPCallError, UnresolvedCall) as exc:
            reason = "deadline_exceeded" if "deadline" in str(exc).lower() else type(exc).__name__
            row.update(status="delivery_failure", reason=reason)
            receipt = self.ledger.get(call_id)
            if receipt and receipt.get("result_json"):
                row["call"] = json.loads(receipt["result_json"])
        save(path, row)
        return row

    def _judge_request(self, case, output, judge):
        parsed = parse_candidate_output(case, output)
        messages = judge_messages(case, parsed)
        # Same evidence/answer/judge is judged once, independent of its provider.
        call_id = "judge/" + digest([case.case_id, messages, judge, self.spec["judge_params"]])
        return parsed, messages, call_id

    def judge(self, case, output, judge, stage):
        parsed, messages, call_id = self._judge_request(case, output, judge)
        path = self.root / "grades" / (digest(call_id)+".json")
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved.get("status") == "graded" and
                    saved.get("call", {}).get("latency_s", 0) >= self.spec["per_call_deadline_s"]):
                raise RuntimeError("Stored late judge requires explicit deadline review")
            recorded = self.ledger.get(call_id)
            if not (recorded and recorded["status"] in ("submitted", "polling")
                    and recorded.get("response_id")):
                return saved
            save(path.with_suffix(".pre-resume.json"), saved)
        row = dict(case_id=case.case_id, judge=judge["id"], call_id=call_id, attempts=[])
        # Judge retries concern the measurement instrument only. Candidate
        # requests retain zero retries. Every attempt has its own durable charge.
        for attempt in range(self.spec.get("judge_max_attempts", 1)):
            attempt_id = call_id if attempt == 0 else call_id+f"/attempt/{attempt+1}"
            if attempt:
                # Rate-limit refusals need room to clear, not an immediate retry.
                time.sleep(self.spec.get("judge_retry_backoff_s", 2) * (2 ** (attempt - 1)))
            try:
                with getattr(self, "judge_locks", {}).get(judge["id"], self.judge_lock):
                    result = self._call(judge, messages, self.spec["judge_params"],
                                        attempt_id, stage)
                row["call"] = result.as_dict()
                if row["call"].get("latency_s", 0) >= self.spec["per_call_deadline_s"]:
                    raise ValueError("judge_deadline_exceeded")
                grade = parse_judge_grade(result.output_text, case=case, output=parsed)
                row["attempts"].append(dict(call_id=attempt_id, status="graded"))
                row.update(status="graded", score=quality_score(grade),
                           acceptable=accepted_grade(grade), grade=asdict(grade),
                           critical=any(grade.critical_error_map().values()),
                           feedback=grade.summary)
                break
            except (ValueError, HTTPCallError, UnresolvedCall) as exc:
                row["attempts"].append(dict(call_id=attempt_id, status="missing_judgment",
                                            reason=str(exc)[:500]))
                row.update(status="missing_judgment", reason=str(exc)[:500])
        save(path, row)
        return row

    def assess(self, case, outcome, judge, stage):
        if outcome["status"] != "valid":
            return dict(status="model_failure", score=0., acceptable=False, critical=False,
                        feedback=outcome["status"])
        return self.judge(case, outcome["output"], judge, stage)

    def judge_pair(self, case, outcome, stage):
        """Independent judges run together; identical answers remain sequential."""
        names = list(self.judges)
        with ThreadPoolExecutor(max_workers=2) as pool:
            grades = pool.map(lambda j: self.assess(case, outcome, self.judges[j], stage), names)
            return dict(zip(names, grades, strict=True))

    @staticmethod
    def _stop_on_future_exception(future, stop_event):
        if not future.cancelled() and future.exception() is not None:
            stop_event.set()

    def _schedule_final_grades(self, executor, scheduled, stop_event, case, outcomes):
        """Schedule each exact case/answer/judge request once."""
        grade_sources = {}
        for label, outcome in outcomes.items():
            grade_sources[label] = {}
            for judge_id, judge in self.judges.items():
                if stop_event.is_set():
                    raise _FinalStageStopped("Final-stage judge scheduling aborted")
                if outcome["status"] != "valid":
                    grade_sources[label][judge_id] = self.assess(case, outcome, judge, "final")
                    continue
                _, _, call_id = self._judge_request(case, outcome["output"], judge)
                future = scheduled.get(call_id)
                if future is None:
                    case_snapshot = case
                    outcome_snapshot = copy.deepcopy(outcome)
                    judge_snapshot = copy.deepcopy(judge)

                    def run(
                        case_value=case_snapshot,
                        outcome_value=outcome_snapshot,
                        judge_value=judge_snapshot,
                    ):
                        if stop_event.is_set():
                            raise _FinalStageStopped("Final-stage judge dispatch aborted")
                        self._call_context.final_stop_event = stop_event
                        try:
                            return self.assess(case_value, outcome_value, judge_value, "final")
                        finally:
                            del self._call_context.final_stop_event

                    future = executor.submit(run)
                    scheduled[call_id] = future
                    future.add_done_callback(
                        lambda completed: self._stop_on_future_exception(completed, stop_event)
                    )
                grade_sources[label][judge_id] = future
        return grade_sources

    @staticmethod
    def _resolve_final_row(pending):
        case, outcomes, grade_sources = pending
        resolved = {
            label: {
                judge_id: copy.deepcopy(source.result() if isinstance(source, Future) else source)
                for judge_id, source in judges.items()
            }
            for label, judges in grade_sources.items()
        }
        return dict(case_id=case.case_id, role=case.role_id,
                    outcomes=outcomes, judges=resolved)

    @staticmethod
    def _cancel_final_grades(executor, futures):
        """Cancel queued judge work and wait at most five minutes for running calls."""
        for future in futures:
            future.cancel()
        running = [future for future in futures if future.running()]
        wait(running, timeout=FINAL_DRAIN_TIMEOUT_S)
        executor.shutdown(wait=False, cancel_futures=True)

    def smoke(self):
        cases = self.cases("train")
        chosen = [next(c for c in cases if c.role_id == role)
                  for role in ("mobile", "acquisition")]
        rows = []
        references = []
        for case in chosen:
            for route in self.routes.values():
                out = self.generate(case, route, "", "smoke")
                grades = self.judge_pair(case, out, "smoke")
                rows.append(dict(case_id=case.case_id, route=route["id"],
                                 outcome=out, judges=grades))
            references.append(dict(case_id=case.case_id, judges=self.judge_pair(
                case, dict(status="valid", output=case.recorded_output.to_dict()), "smoke")))
        controls = []
        # Intentionally corrupted development outputs test judge operation only.
        for control in synthetic_negative_controls(chosen[0])[:2]:
            for judge in self.judges.values():
                grade = self.judge(control.case, control.output.to_dict(), judge, "smoke")
                controls.append(dict(name=control.name, judge=judge["id"], grade=grade,
                                     expected_issue=control.expected_issue))
        # Smoke verifies a working contract for every route. Genuine invalid
        # model output is retained; validation measures its frequency.
        healthy = all(any(r["route"] == route and r["outcome"]["status"] == "valid"
                          for r in rows) for route in self.routes)
        healthy = healthy and all(
            all(g["status"] in ("graded", "model_failure") for g in r["judges"].values())
            for r in rows)
        healthy = healthy and all(all(g["status"] == "graded" for g in r["judges"].values())
                                  for r in references)
        controls_ok = all(dict(g["grade"].get("grade", {}).get("critical_errors", []))
                          .get(g["expected_issue"]) for g in controls)
        result = dict(status="passed" if healthy and controls_ok else "needs_review",
                      rows=rows, references=references, controls=controls,
                      used_usd=self.ledger.used())
        save(self.root / "smoke.json", result)
        if result["status"] != "passed":
            raise RuntimeError("Smoke failures require inspection before the study continues")
        return result

    def baseline(self):
        if json.loads((self.root / "smoke.json").read_text())["status"] != "passed":
            raise ValueError("Smoke has not passed")
        cases = self.cases("validation")
        rows = []
        for case in cases:
            reference = self.judge(case, case.recorded_output.to_dict(),
                                   self.judges["judge-glm"], "baseline")
            order = list(self.routes.values())
            random.Random(digest([self.spec["seed"], case.case_id])).shuffle(order)
            with ThreadPoolExecutor(max_workers=5) as pool:
                outs = list(pool.map(lambda r, c=case: self.generate(c, r, "", "baseline"), order))
            for out in outs:
                grade = self.assess(case, out, self.judges["judge-glm"], "baseline")
                rows.append(dict(case_id=case.case_id, role=case.role_id,
                                 route=out["route"], outcome=out, grade=grade, reference=reference))
            save(self.root / "baseline-progress.json", dict(completed=len(rows), rows=rows))
            print(json.dumps({"stage": "baseline", "completed": len(rows),
                              "used_usd": self.ledger.used()}), flush=True)
        stats = []
        for route in self.routes:
            cells = [r for r in rows if r["route"] == route]
            known = all("score" in r["grade"] and "score" in r["reference"] for r in cells)
            valid = sum(r["outcome"]["status"] == "valid" for r in cells)
            cost = sum(float((self.ledger.get(r["outcome"]["call_id"]) or {}).get(
                "cost_usd", r["outcome"].get("call", {}).get("cost_usd", 0)) or 0) for r in cells)
            # A terminal provider failure can still have a known charge. Use
            # the durable ledger rather than requiring a successful response.
            receipts = [self.ledger.get(r["outcome"]["call_id"]) or {} for r in cells]
            cost_known = all(receipt.get("cost_kind") in ("reported", "derived")
                             and receipt.get("cost_usd") is not None for receipt in receipts)
            mean = statistics.mean(r["grade"]["score"] for r in cells) if known else None
            delta = statistics.mean(r["grade"]["score"]-r["reference"]["score"]
                                    for r in cells) if known else None
            critical_delta = (sum(r["grade"].get("critical", False) for r in cells) -
                              sum(r["reference"].get("critical", False) for r in cells))
            stats.append(dict(route=route, mean_quality=mean, delta=delta,
                              valid=valid, n=len(cells), mean_cost=cost/len(cells),
                              critical_delta=critical_delta,
                              cost_known=cost_known, eligible=known and cost_known and
                              valid/len(cells) >= .95 and delta >= -.05 and critical_delta <= 0))
        eligible = [s for s in stats if s["eligible"]]
        ranked = sorted(eligible, key=lambda s: (s["mean_cost"], -s["mean_quality"]))
        if not ranked:
            ranked = sorted([s for s in stats if s["mean_quality"] is not None],
                            key=lambda s: (-s["mean_quality"], s["mean_cost"]))
        if not ranked:
            raise RuntimeError("No route with complete validation grading")
        result = dict(status="completed", rows=rows, routes=stats,
                      selected_route=ranked[0]["route"], qualified_on_validation=bool(eligible))
        save(self.root / "baseline.json", result)
        return result

    def optimize(self):
        from compound.migration_gepa import optimize_methodology

        directory = self.spec.get("gepa_directory", "gepa")
        if directory not in {"gepa", "gepa-v2"}:
            raise ValueError("unsupported GEPA development directory")
        baseline = json.loads((self.root / "baseline.json").read_text())
        route = self.routes[baseline["selected_route"]]
        train, val = self.cases("train"), self.cases("validation")
        known = {c.case_id: c for c in train+val}

        def evaluate(raw, methodology):
            case = known[raw["case_id"]]
            try:
                for representative in (train[0], next(c for c in train
                                                       if c.role_id != train[0].role_id)):
                    candidate_messages(representative, methodology)
            except ValueError as exc:
                return dict(case_id=case.case_id, score=0.,
                            feedback="Invalid methodology component: "+str(exc), output=None)
            # Reuse original validation generation, not a second paid attempt.
            stage = "baseline" if not methodology and case in val else "gepa"
            outcome = self.generate(case, route, methodology, stage)
            grade = self.assess(case, outcome, self.judges["judge-glm"], "gepa")
            if "score" not in grade:
                raise RuntimeError("GEPA judge returned no valid assessment")
            return dict(case_id=case.case_id, score=grade["score"],
                        feedback=grade["feedback"], output=outcome.get("output"))

        def reflect(messages):
            params = dict(max_tokens=2048, reasoning_effort="medium")
            return self._call(self.judges["judge-glm"], messages, params,
                              "reflection/"+digest(messages), "gepa").output_text

        def encode(c):
            return dict(case_id=c.case_id, system_prompt=c.original_system,
                        user_prompt=c.candidate_evidence)
        result = optimize_methodology([encode(c) for c in train], [encode(c) for c in val],
                                      evaluate, reflect, self.root / directory, seed=7419,
                                      max_metric_calls=self.spec.get("gepa_max_metric_calls", 160),
                                      max_reflections=self.spec.get("gepa_max_reflections", 4),
                                      seed_methodology="")
        chosen = result["optimized_methodology"]
        # Structural review of every role before opening the held-out partition.
        for case in val:
            candidate_messages(case, chosen)
        frozen = dict(status="completed", route=route["id"], methodology=chosen,
                      methodology_sha256=digest(chosen), gepa_status=result["status"],
                      validation_result=result, frozen_at=datetime.now(UTC).isoformat())
        save(self.root / "frozen-selection.json", frozen)
        return frozen

    def final(self):
        frozen = json.loads((self.root / "frozen-selection.json").read_text())
        if digest(frozen["methodology"]) != frozen["methodology_sha256"]:
            raise ValueError("Selected prompt changed after freezing")
        route = self.routes[frozen["route"]]
        workers = self.spec.get("final_judge_workers", FINAL_JUDGE_WORKERS)
        if (isinstance(workers, bool) or not isinstance(workers, int)
                or workers != FINAL_JUDGE_WORKERS):
            raise ValueError("final_judge_workers must be exactly 2")
        window = self.spec.get("final_case_window", MAX_FINAL_CASE_WINDOW)
        if (isinstance(window, bool) or not isinstance(window, int)
                or not 1 <= window <= MAX_FINAL_CASE_WINDOW):
            raise ValueError("final_case_window must be an integer from 1 through 4")
        rows = []
        pending = deque()
        scheduled: dict[str, Future] = {}
        stop_event = threading.Event()
        if not hasattr(self, "_call_context"):
            self._call_context = threading.local()
        self._active_final_stop_event = stop_event
        executor = ThreadPoolExecutor(max_workers=FINAL_JUDGE_WORKERS)
        try:
            for case in self.cases("test"):
                variants = [("original", ""), ("optimized", frozen["methodology"])]
                random.Random(digest([case.case_id, "final"])).shuffle(variants)
                outcomes = {"gemini": dict(status="valid", output=case.recorded_output.to_dict())}
                for label, prompt in variants:
                    if stop_event.is_set():
                        raise _FinalStageStopped("Final-stage candidate dispatch aborted")
                    outcomes[label] = self.generate(case, route, prompt, "final")
                grade_sources = self._schedule_final_grades(
                    executor, scheduled, stop_event, case, outcomes
                )
                pending.append((case, outcomes, grade_sources))
                if len(pending) >= window:
                    rows.append(self._resolve_final_row(pending.popleft()))
                    save(self.root / "final-progress.json", dict(completed=len(rows), rows=rows))
                    print(json.dumps({"stage": "final", "completed": len(rows),
                                      "used_usd": self.ledger.used()}), flush=True)
            while pending:
                rows.append(self._resolve_final_row(pending.popleft()))
                save(self.root / "final-progress.json", dict(completed=len(rows), rows=rows))
                print(json.dumps({"stage": "final", "completed": len(rows),
                                  "used_usd": self.ledger.used()}), flush=True)
        except BaseException:
            stop_event.set()
            self._cancel_final_grades(executor, tuple(scheduled.values()))
            raise
        else:
            executor.shutdown(wait=True)
            self._active_final_stop_event = None
        result = dict(status="completed", selected=frozen, rows=rows,
                      used_usd=self.ledger.used())
        save(self.root / "final.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[
        "prepare", "smoke", "baseline", "optimize", "final", "all"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--catalog-dir", type=Path)
    parser.add_argument("--keys", type=Path)
    parser.add_argument("--go", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    if args.stage == "prepare":
        prepare(args.source, args.root, args.catalog_dir)
        return
    if not args.go:
        print("Dry run: --go is required for paid execution")
        return
    import fcntl

    with (args.root / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.keys:
            for key, value in json.loads(args.keys.read_text()).items():
                os.environ[key] = value
        study = Study(args.root)
        stages = ["smoke", "baseline", "optimize", "final"] if args.stage == "all" else [args.stage]
        for stage in stages:
            filename = {"optimize": "frozen-selection.json"}.get(stage, stage+".json")
            result_path = args.root / filename
            if result_path.exists() and json.loads(result_path.read_text()).get("status") in (
                "passed", "completed"
            ):
                continue
            try:
                getattr(study, stage)()
            except Exception as exc:
                save(args.root / "controller-stop.json", dict(
                    stage=stage, error_type=type(exc).__name__, error=str(exc)[:1000],
                    at=datetime.now(UTC).isoformat(), used_usd=study.ledger.used()))
                raise
        save(args.root / "controller-complete.json", dict(at=datetime.now(UTC).isoformat(),
                                                         used_usd=study.ledger.used()))


if __name__ == "__main__":
    main()
