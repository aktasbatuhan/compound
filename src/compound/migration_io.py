"""Audited, durable I/O for bounded provider-migration experiments.

This module deliberately does not reuse the repository's general provider path.
Every paid request first acquires an atomic SQLite reservation. A process crash,
transport timeout, or unfinished background response keeps that reservation
forever: only a response with known usage can release unused headroom.

Typical use::

    ledger = Ledger("run/ledger.sqlite", cap_usd=23.5, stage_caps={"smoke": 1.5})
    service = InferenceService(
        ledger,
        rates={"dw-flex": {"input_per_million": 0.20, "output_per_million": 0.60}},
    )
    result = service.call(route, messages, {"max_tokens": 1024},
                          call_id="smoke/example/0", stage="smoke")

``call_id`` must identify the logical attempt, not merely the prompt. A completed
call is reusable only when its canonical request fingerprint is identical. An
unfinished call with a stored response ID is resumed by polling; any other
unresolved call is never submitted again under the same ID.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

NANO_USD = Decimal("1000000000")
FRAMING_TOKEN_BOUND = 4096


class MigrationIOError(RuntimeError):
    """Base class for experiment I/O failures."""


class BudgetExceeded(MigrationIOError):
    """The total or stage cap cannot cover a conservative reservation."""


class RequestIdentityError(MigrationIOError):
    """A call ID was reused for a different canonical request."""


class UnpricedRoute(MigrationIOError):
    """No explicit rate is available for a route that could spend money."""


class UnresolvedCall(MigrationIOError):
    """A request may have reached the provider and cannot safely be repeated."""


class SafetyStop(MigrationIOError):
    """Provider evidence exceeded a bound or contradicted the requested tier."""


class HTTPCallError(MigrationIOError):
    """The provider returned a non-success response."""


class _TransportDeadlineExceeded(TimeoutError):
    """The absolute wall-clock budget expired during one HTTP exchange."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.headers = dict(headers or {})


@dataclass(frozen=True, slots=True)
class Rate:
    """Explicit USD rates per million tokens for one route or model."""

    input_per_million: Decimal
    output_per_million: Decimal
    cached_input_per_million: Decimal | None = None
    cache_write_per_million: Decimal | None = None

    @classmethod
    def from_value(cls, value: Rate | Mapping[str, Any]) -> Rate:
        if isinstance(value, cls):
            rate = value
        else:
            try:
                input_rate = Decimal(str(value["input_per_million"]))
                output_rate = Decimal(str(value["output_per_million"]))
            except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
                raise ValueError("rates need input_per_million and output_per_million") from exc
            cached = value.get("cached_input_per_million")
            cache_write = value.get("cache_write_per_million")
            rate = cls(
                input_per_million=input_rate,
                output_per_million=output_rate,
                cached_input_per_million=None if cached is None else Decimal(str(cached)),
                cache_write_per_million=(
                    None if cache_write is None else Decimal(str(cache_write))
                ),
            )
        rate_values = [rate.input_per_million, rate.output_per_million]
        rate_values.extend(
            value
            for value in (rate.cached_input_per_million, rate.cache_write_per_million)
            if value is not None
        )
        if any(not value.is_finite() for value in rate_values):
            raise ValueError("token rates must be finite")
        if min(rate.input_per_million, rate.output_per_million) < 0 or (
            rate.cached_input_per_million is not None
            and rate.cached_input_per_million < 0
        ) or (
            rate.cache_write_per_million is not None
            and rate.cache_write_per_million < 0
        ):
            raise ValueError("token rates cannot be negative")
        return rate


@dataclass(frozen=True, slots=True)
class HTTPResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] | None = None


class _DeadlineState:
    """Share a live urllib response with the deadline watchdog."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.expired = False
        self.response: Any = None
        self.status: int | None = None
        self.headers: dict[str, str] = {}

    def attach(self, response: Any) -> None:
        with self.lock:
            self.status = int(response.status)
            self.headers = dict(response.headers.items())
            if not self.expired:
                self.response = response
                return
        try:
            response.close()
        except Exception:
            pass
        raise self.error()

    def detach(self, response: Any) -> None:
        with self.lock:
            if self.response is response:
                self.response = None

    def expire(self) -> None:
        with self.lock:
            self.expired = True
            response = self.response
        if response is not None:
            threading.Thread(
                target=self._abort,
                args=(response,),
                daemon=True,
            ).start()

    @staticmethod
    def _abort(response: Any) -> None:
        """Interrupt a reader without ever blocking the deadline caller."""
        stream = getattr(response, "fp", None)
        raw = getattr(stream, "raw", None)
        sock = getattr(raw, "_sock", None) or getattr(stream, "_sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            response.close()
        except Exception:
            pass

    def check(self) -> None:
        with self.lock:
            expired = self.expired
        if expired:
            raise self.error()

    def error(self) -> _TransportDeadlineExceeded:
        return _TransportDeadlineExceeded(
            "absolute HTTP deadline exceeded",
            status=self.status,
            headers=self.headers,
        )


@dataclass(frozen=True, slots=True)
class CallResult:
    call_id: str
    route_id: str
    output_text: str
    usage: dict[str, Any]
    cost_usd: float
    cost_kind: str
    latency_s: float
    raw: dict[str, Any]
    upstream: Any
    requested_service_tier: str | None
    echoed_service_tier: str | None
    service_tier_verified: bool | None
    cache_hit: bool
    response_id: str | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Reservation:
    action: str
    row: dict[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _usd_to_nano(value: Any, *, ceiling: bool) -> int:
    rounding = ROUND_CEILING if ceiling else ROUND_FLOOR
    return int((Decimal(str(value)) * NANO_USD).to_integral_value(rounding=rounding))


def _nano_to_usd(value: int) -> float:
    return float(Decimal(value) / NANO_USD)


class Ledger:
    """SQLite reservation ledger using an independent connection per operation."""

    def __init__(
        self,
        path: str | Path,
        cap_usd: float | Decimal,
        stage_caps: Mapping[str, float | Decimal],
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = (
            Path(artifact_dir)
            if artifact_dir is not None
            else self.path.parent / f"{self.path.stem}_artifacts"
        )
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.artifact_dir, 0o700)
        self.cap_nano = _usd_to_nano(cap_usd, ceiling=False)
        self.stage_caps_nano = {
            str(stage): _usd_to_nano(cap, ceiling=False)
            for stage, cap in stage_caps.items()
        }
        if self.cap_nano < 0 or any(value < 0 for value in self.stage_caps_nano.values()):
            raise ValueError("spend caps cannot be negative")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    api TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reserved_nano INTEGER NOT NULL,
                    accounted_nano INTEGER NOT NULL,
                    input_token_bound INTEGER NOT NULL,
                    output_token_bound INTEGER NOT NULL,
                    request_artifact TEXT NOT NULL,
                    response_artifact TEXT,
                    response_id TEXT,
                    result_json TEXT,
                    cost_usd TEXT,
                    cost_kind TEXT,
                    latency_s REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempt_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    detail_json TEXT,
                    created_at REAL NOT NULL
                );
                """
            )
        os.chmod(self.path, 0o600)

    def used(self, stage: str | None = None) -> float:
        """Return settled charges plus all still-open reservations in USD."""
        query = "SELECT COALESCE(SUM(accounted_nano), 0) FROM calls"
        args: tuple[Any, ...] = ()
        if stage is not None:
            query += " WHERE stage = ?"
            args = (stage,)
        with self._connect() as conn:
            value = int(conn.execute(query, args).fetchone()[0])
        return _nano_to_usd(value)

    def get(self, call_id: str) -> dict[str, Any] | None:
        """Read one raw ledger row for audit or orchestration."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        return None if row is None else dict(row)

    def events(self, call_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM attempt_events"
        args: tuple[Any, ...] = ()
        if call_id is not None:
            query += " WHERE call_id = ?"
            args = (call_id,)
        query += " ORDER BY id"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, args)]

    def record_event(
        self, call_id: str, kind: str, message: str, detail: Mapping[str, Any] | None = None
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO attempt_events(call_id, kind, message, detail_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    call_id,
                    kind,
                    message,
                    None if detail is None else _canonical_bytes(detail).decode("utf-8"),
                    time.time(),
                ),
            )

    def reserve(
        self,
        *,
        call_id: str,
        fingerprint: str,
        route_id: str,
        stage: str,
        api: str,
        reserved_usd: Decimal,
        input_token_bound: int,
        output_token_bound: int,
        request_artifact: str,
    ) -> _Reservation:
        """Atomically acquire budget or classify the existing logical call."""
        reserved_nano = _usd_to_nano(reserved_usd, ceiling=True)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM calls WHERE call_id = ?", (call_id,)
            ).fetchone()
            if existing is not None:
                row = dict(existing)
                if row["fingerprint"] != fingerprint:
                    conn.rollback()
                    self.record_event(call_id, "identity_error", "call_id request mismatch")
                    raise RequestIdentityError(
                        f"call_id {call_id!r} already belongs to a different request"
                    )
                conn.commit()
                if row["status"] == "completed":
                    return _Reservation("cached", row)
                if row["response_id"] and row["status"] in {"submitted", "polling"}:
                    return _Reservation("resume", row)
                if row["status"] == "safety_stop":
                    raise SafetyStop(
                        row.get("error")
                        or f"call_id {call_id!r} previously triggered a safety stop"
                    )
                raise UnresolvedCall(
                    f"call_id {call_id!r} is {row['status']!r}; refusing another POST"
                )

            total = int(
                conn.execute("SELECT COALESCE(SUM(accounted_nano), 0) FROM calls").fetchone()[0]
            )
            stage_total = int(
                conn.execute(
                    "SELECT COALESCE(SUM(accounted_nano), 0) FROM calls WHERE stage = ?",
                    (stage,),
                ).fetchone()[0]
            )
            stage_cap = self.stage_caps_nano.get(stage)
            reason = None
            if total + reserved_nano > self.cap_nano:
                reason = "total"
            elif stage_cap is not None and stage_total + reserved_nano > stage_cap:
                reason = f"stage {stage!r}"
            if reason is not None:
                conn.rollback()
                message = f"{reason} cap cannot cover ${_nano_to_usd(reserved_nano):.9f}"
                self.record_event(call_id, "budget_exceeded", message)
                raise BudgetExceeded(message)

            now = time.time()
            conn.execute(
                """INSERT INTO calls(
                    call_id, fingerprint, route_id, stage, api, status,
                    reserved_nano, accounted_nano, input_token_bound,
                    output_token_bound, request_artifact, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?, ?, ?)""",
                (
                    call_id,
                    fingerprint,
                    route_id,
                    stage,
                    api,
                    reserved_nano,
                    reserved_nano,
                    input_token_bound,
                    output_token_bound,
                    request_artifact,
                    now,
                    now,
                ),
            )
            conn.commit()
            row = self.get(call_id)
            assert row is not None
            return _Reservation("submit", row)
        finally:
            conn.close()

    def update(self, call_id: str, **fields: Any) -> None:
        if not fields:
            return
        allowed = {
            "status",
            "accounted_nano",
            "response_artifact",
            "response_id",
            "result_json",
            "cost_usd",
            "cost_kind",
            "latency_s",
            "error",
        }
        unexpected = set(fields) - allowed
        if unexpected:
            raise ValueError(f"unsupported ledger fields: {sorted(unexpected)}")
        fields["updated_at"] = time.time()
        sql = ", ".join(f"{name} = ?" for name in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE calls SET {sql} WHERE call_id = ?",  # noqa: S608 - names allowlisted
                (*fields.values(), call_id),
            )

    def write_artifact(self, call_id: str, kind: str, payload: Any) -> str:
        """Persist a private, fsynced JSON artifact and return its path."""
        key = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:24]
        suffix = "" if kind == "request" else f"-{time.time_ns()}"
        path = self.artifact_dir / f"{key}.{kind}{suffix}.json"
        data = _canonical_bytes(payload) + b"\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        return str(path)


Transport = Callable[[str, str, Mapping[str, str], bytes | None, float], HTTPResponse]


def _transport_with_deadline(
    transport: Transport,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    idle_timeout_s: float,
    total_timeout_s: float,
) -> HTTPResponse:
    """Run one exchange behind a non-joining absolute-deadline watchdog."""
    if total_timeout_s <= 0:
        raise _TransportDeadlineExceeded("absolute HTTP deadline exceeded")
    done = threading.Event()
    state = _DeadlineState()
    outcome: dict[str, Any] = {}

    def exchange() -> None:
        try:
            state.check()
            if transport is _stdlib_transport:
                outcome["response"] = _stdlib_transport(
                    method,
                    url,
                    headers,
                    body,
                    idle_timeout_s,
                    _deadline_state=state,
                )
            else:
                outcome["response"] = transport(
                    method, url, headers, body, idle_timeout_s
                )
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()

    # A daemon thread avoids ThreadPoolExecutor's blocking shutdown behavior.
    # On urllib exchanges the state object also closes the live response.
    threading.Thread(target=exchange, daemon=True).start()
    if not done.wait(total_timeout_s):
        state.expire()
        if done.is_set() and "response" in outcome:
            return outcome["response"]
        raise state.error()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["response"]


class InferenceService:
    """No-retry OpenRouter/Doubleword HTTP client backed by :class:`Ledger`."""

    def __init__(
        self,
        ledger: Ledger,
        rates: Mapping[str, Rate | Mapping[str, Any]],
        *,
        transport: Transport | None = None,
        timeout_s: float = 300.0,
        poll_interval_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ledger = ledger
        self.rates = {key: Rate.from_value(value) for key, value in rates.items()}
        self.transport = transport or _stdlib_transport
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.sleep = sleep
        self.clock = clock

    def call(
        self,
        route: Mapping[str, Any],
        messages: list[dict[str, Any]],
        params: Mapping[str, Any],
        *,
        call_id: str,
        stage: str,
        poll_deadline_s: float = 300.0,
        output_token_bound: int | None = None,
    ) -> CallResult:
        """Make one paid logical call, or resume/cache its durable record.

        ``output_token_bound`` raises the reservation and safety ceiling only. The
        request body is untouched, so the call identity and the instrument are
        unchanged. Reasoning models can bill output tokens beyond the max_tokens
        they honour for visible completion, and the ceiling must cover that.

        The method never retries a POST. Transport failures remain reserved and
        require a new, explicitly chosen call ID if an operator decides to risk
        another paid submission.
        """
        try:
            return self._call(
                route,
                messages,
                params,
                output_token_bound=output_token_bound,
                call_id=call_id,
                stage=stage,
                poll_deadline_s=poll_deadline_s,
            )
        except Exception as exc:
            self.ledger.record_event(
                call_id,
                "exception",
                f"{type(exc).__name__}: {exc}",
            )
            raise

    def _call(
        self,
        route: Mapping[str, Any],
        messages: list[dict[str, Any]],
        params: Mapping[str, Any],
        *,
        call_id: str,
        stage: str,
        poll_deadline_s: float,
        output_token_bound: int | None = None,
    ) -> CallResult:
        route_data = _validate_route(route)
        body, requested_tokens = _shape_request(route_data, messages, params)
        max_tokens = requested_tokens
        if output_token_bound is not None:
            if isinstance(output_token_bound, bool) or not isinstance(output_token_bound, int):
                raise ValueError("output_token_bound must be an integer")
            if output_token_bound < requested_tokens:
                raise ValueError("output_token_bound cannot be below params.max_tokens")
            max_tokens = output_token_bound
        rate = self._rate_for(route_data)
        request_bytes = _canonical_bytes(body)
        input_bound = len(request_bytes) + FRAMING_TOKEN_BOUND
        reservation = _cost_from_tokens(
            input_bound,
            max_tokens,
            rate,
            cached_tokens=0,
            reservation=True,
        )
        identity = {
            "route": route_data,
            "body": body,
            "rate": {
                "input_per_million": str(rate.input_per_million),
                "output_per_million": str(rate.output_per_million),
                "cached_input_per_million": (
                    None
                    if rate.cached_input_per_million is None
                    else str(rate.cached_input_per_million)
                ),
                "cache_write_per_million": (
                    None
                    if rate.cache_write_per_million is None
                    else str(rate.cache_write_per_million)
                ),
            },
        }
        fingerprint = _fingerprint(identity)
        request_path = self.ledger.artifact_dir / (
            hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:24] + ".request.json"
        )
        reserved = self.ledger.reserve(
            call_id=call_id,
            fingerprint=fingerprint,
            route_id=route_data["id"],
            stage=stage,
            api=route_data["api"],
            reserved_usd=reservation,
            input_token_bound=input_bound,
            output_token_bound=max_tokens,
            request_artifact=str(request_path),
        )
        if reserved.action == "cached":
            result = _result_from_row(reserved.row)
            return replace(result, cache_hit=True)

        total_budget_s = max(0.0, float(poll_deadline_s))
        durable_age_s = max(0.0, time.time() - float(reserved.row["created_at"]))
        absolute_deadline = self.clock() + max(0.0, total_budget_s - durable_age_s)
        if durable_age_s >= total_budget_s:
            self._deadline_exceeded(
                call_id,
                total_budget_s,
                "logical call deadline expired before HTTP exchange",
            )

        if reserved.action == "submit":
            self.ledger.write_artifact(
                call_id,
                "request",
                {
                    "call_id": call_id,
                    "route_id": route_data["id"],
                    "api": route_data["api"],
                    "url": _post_url(route_data),
                    "body": body,
                    "input_token_bound": input_bound,
                    "output_token_bound": max_tokens,
                    "reserved_usd": str(reservation),
                    "rate": identity["rate"],
                },
            )
            payload, latency_s, artifact = self._post(
                call_id,
                route_data,
                body,
                absolute_deadline=absolute_deadline,
                total_budget_s=total_budget_s,
            )
            self.ledger.update(call_id, latency_s=latency_s)
            response_id = _response_id(payload)
            if response_id and (route_data.get("background") or _is_unfinished(payload)):
                # Persist the resumable handle before inspecting status or sleeping.
                self.ledger.update(
                    call_id,
                    status="submitted",
                    response_id=response_id,
                    response_artifact=artifact,
                    error=None,
                )
            if _is_unfinished(payload):
                if not response_id:
                    self._unknown(call_id, "unfinished response did not contain an id", artifact)
                payload, poll_latency, artifact = self._poll(
                    call_id,
                    route_data,
                    response_id,
                    absolute_deadline,
                    total_budget_s,
                )
                latency_s += poll_latency
        else:
            response_id = reserved.row["response_id"]
            prior_latency = float(reserved.row["latency_s"])
            payload, poll_latency, artifact = self._poll(
                call_id,
                route_data,
                response_id,
                absolute_deadline,
                total_budget_s,
            )
            latency_s = prior_latency + poll_latency

        return self._settle(
            call_id,
            route_data,
            payload,
            rate,
            input_bound=input_bound,
            output_bound=max_tokens,
            latency_s=latency_s,
            response_artifact=artifact,
            deadline_exceeded=self.clock() >= absolute_deadline,
            total_budget_s=total_budget_s,
        )

    def _rate_for(self, route: Mapping[str, Any]) -> Rate:
        rate = self.rates.get(str(route["id"])) or self.rates.get(str(route["model"]))
        if rate is None:
            raise UnpricedRoute(
                f"no explicit rate for route {route['id']!r} or model {route['model']!r}"
            )
        return rate

    def _headers(self, route: Mapping[str, Any]) -> dict[str, str]:
        key = os.getenv(str(route["api_key_env"]), "")
        if not key:
            raise MigrationIOError(f"environment variable {route['api_key_env']!r} is empty")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _post(
        self,
        call_id: str,
        route: Mapping[str, Any],
        body: Mapping[str, Any],
        *,
        absolute_deadline: float,
        total_budget_s: float,
    ) -> tuple[dict[str, Any], float, str]:
        started = self.clock()
        try:
            remaining = max(0.0, absolute_deadline - self.clock())
            response = _transport_with_deadline(
                self.transport,
                "POST",
                _post_url(route),
                self._headers(route),
                _canonical_bytes(body),
                min(self.timeout_s, max(0.001, remaining)),
                remaining,
            )
        except _TransportDeadlineExceeded as exc:
            artifact = self._store_deadline_response(call_id, exc)
            self._deadline_exceeded(
                call_id,
                total_budget_s,
                "POST absolute deadline exceeded",
                artifact,
            )
        except BaseException as exc:
            if self.clock() >= absolute_deadline:
                self._deadline_exceeded(
                    call_id,
                    total_budget_s,
                    "POST absolute deadline exceeded",
                )
            self._unknown(call_id, f"POST transport failure: {type(exc).__name__}: {exc}")
        latency = self.clock() - started
        artifact = self._store_response(call_id, "post_response", response)
        try:
            payload = _decode_response(response)
        except HTTPCallError as exc:
            self._unknown(call_id, str(exc), artifact)
        if not 200 <= response.status < 300:
            message = f"POST returned HTTP {response.status}"
            self.ledger.update(call_id, status="unknown", response_artifact=artifact, error=message)
            self.ledger.record_event(call_id, "http_error", message, {"status": response.status})
            raise HTTPCallError(message)
        return payload, latency, artifact

    def _poll(
        self,
        call_id: str,
        route: Mapping[str, Any],
        response_id: str,
        absolute_deadline: float,
        total_budget_s: float,
    ) -> tuple[dict[str, Any], float, str]:
        if not response_id:
            self._unknown(call_id, "cannot resume background call without response id")
        started = self.clock()
        row = self.ledger.get(call_id)
        assert row is not None
        prior_latency = float(row["latency_s"])
        artifact = ""
        while True:
            if self.clock() >= absolute_deadline:
                self._deadline_exceeded(
                    call_id,
                    total_budget_s,
                    f"poll absolute deadline exceeded for response {response_id}",
                )
            try:
                remaining = max(0.0, absolute_deadline - self.clock())
                response = _transport_with_deadline(
                    self.transport,
                    "GET",
                    f"{_responses_url(route)}/{response_id}",
                    self._headers(route),
                    None,
                    min(self.timeout_s, max(0.001, remaining)),
                    remaining,
                )
            except _TransportDeadlineExceeded as exc:
                artifact = self._store_deadline_response(call_id, exc)
                self._deadline_exceeded(
                    call_id,
                    total_budget_s,
                    f"poll absolute deadline exceeded for response {response_id}",
                    artifact,
                )
            except BaseException as exc:
                if self.clock() >= absolute_deadline:
                    self._deadline_exceeded(
                        call_id,
                        total_budget_s,
                        f"poll absolute deadline exceeded for response {response_id}",
                    )
                message = f"poll transport failure for {response_id}: {type(exc).__name__}: {exc}"
                self.ledger.update(
                    call_id,
                    status="submitted",
                    latency_s=prior_latency + self.clock() - started,
                    error=message,
                )
                self.ledger.record_event(call_id, "poll_error", message)
                raise UnresolvedCall(message) from exc
            artifact = self._store_response(call_id, "poll_response", response)
            self.ledger.update(
                call_id,
                status="polling",
                response_artifact=artifact,
                latency_s=prior_latency + self.clock() - started,
                error=None,
            )
            if not 200 <= response.status < 300:
                message = f"poll returned HTTP {response.status} for {response_id}"
                self.ledger.update(
                    call_id,
                    status="submitted",
                    latency_s=prior_latency + self.clock() - started,
                    error=message,
                )
                self.ledger.record_event(call_id, "poll_http_error", message)
                raise HTTPCallError(message)
            try:
                payload = _decode_response(response)
            except HTTPCallError as exc:
                message = str(exc)
                self.ledger.update(
                    call_id,
                    status="submitted",
                    latency_s=prior_latency + self.clock() - started,
                    error=message,
                )
                self.ledger.record_event(call_id, "poll_decode_error", message)
                raise UnresolvedCall(message) from exc
            if not _is_unfinished(payload):
                return payload, self.clock() - started, artifact
            if self.clock() >= absolute_deadline:
                continue
            self.sleep(
                min(self.poll_interval_s, max(0.0, absolute_deadline - self.clock()))
            )

    def _settle(
        self,
        call_id: str,
        route: Mapping[str, Any],
        payload: dict[str, Any],
        rate: Rate,
        *,
        input_bound: int,
        output_bound: int,
        latency_s: float,
        response_artifact: str,
        deadline_exceeded: bool,
        total_budget_s: float,
    ) -> CallResult:
        usage = _usage(payload)
        try:
            input_tokens = _integer_usage(usage, "prompt_tokens", "input_tokens")
            output_tokens = _integer_usage(usage, "completion_tokens", "output_tokens")
            if input_tokens is None or output_tokens is None:
                self._unknown(call_id, "successful response omitted token usage", response_artifact)
            cached_tokens, cache_write_tokens = _cache_token_counts(usage, input_tokens)
            reported = usage.get("cost")
            if reported is not None and "openrouter" in str(route["base_url"]).lower():
                try:
                    cost = Decimal(str(reported))
                except (InvalidOperation, ValueError) as exc:
                    raise SafetyStop(f"provider cost is invalid: {reported!r}") from exc
                cost_kind = "reported"
            else:
                cost = _cost_from_tokens(
                    input_tokens,
                    output_tokens,
                    rate,
                    cached_tokens=cached_tokens,
                    cache_write_tokens=cache_write_tokens,
                    reservation=False,
                )
                cost_kind = "derived"
            if not cost.is_finite() or cost < 0:
                raise SafetyStop(f"provider cost is invalid: {cost}")
        except SafetyStop as exc:
            self.ledger.update(call_id, status="safety_stop", error=str(exc))
            self.ledger.record_event(call_id, "safety_stop", str(exc))
            raise

        row = self.ledger.get(call_id)
        assert row is not None
        # A resumed background response may spend minutes or hours parked between
        # processes. Active HTTP time alone would make that late completion look
        # artificially fast, so the durable logical-call latency includes downtime.
        latency_s = max(latency_s, max(0.0, time.time() - float(row["created_at"])))
        violations: list[str] = []
        if input_tokens > input_bound:
            violations.append(f"input tokens {input_tokens} exceed bound {input_bound}")
        if output_tokens > output_bound:
            violations.append(f"output tokens {output_tokens} exceed bound {output_bound}")
        actual_nano = _usd_to_nano(cost, ceiling=True)
        if actual_nano > int(row["reserved_nano"]):
            violations.append(
                f"cost ${cost} exceeds reservation ${_nano_to_usd(row['reserved_nano']):.9f}"
            )

        requested_tier = route.get("service_tier")
        echoed_tier = payload.get("service_tier")
        tier_verified: bool | None = None
        if requested_tier is not None and echoed_tier is not None:
            tier_verified = str(requested_tier) == str(echoed_tier)
            if not tier_verified:
                violations.append(
                    f"service tier {echoed_tier!r} does not match requested {requested_tier!r}"
                )

        result = CallResult(
            call_id=call_id,
            route_id=str(route["id"]),
            output_text=extract_visible_output_text(payload),
            usage=usage,
            cost_usd=float(cost),
            cost_kind=cost_kind,
            latency_s=latency_s,
            raw=payload,
            upstream=payload.get("provider"),
            requested_service_tier=None if requested_tier is None else str(requested_tier),
            echoed_service_tier=None if echoed_tier is None else str(echoed_tier),
            service_tier_verified=tier_verified,
            cache_hit=False,
            response_id=_response_id(payload) or row["response_id"],
        )
        accounted_nano = max(actual_nano, int(row["reserved_nano"])) if violations else actual_nano
        common = {
            "accounted_nano": accounted_nano,
            "response_artifact": response_artifact,
            "response_id": result.response_id,
            "result_json": _canonical_bytes(result.as_dict()).decode("utf-8"),
            "cost_usd": str(cost),
            "cost_kind": cost_kind,
            "latency_s": latency_s,
        }
        if violations:
            message = "; ".join(violations)
            self.ledger.update(call_id, status="safety_stop", error=message, **common)
            self.ledger.record_event(call_id, "safety_stop", message)
            raise SafetyStop(message)
        terminal_status = str(payload.get("status", "")).lower()
        if terminal_status in {"failed", "cancelled", "canceled", "expired"}:
            message = f"provider response ended with status {terminal_status!r}"
            self.ledger.update(call_id, status="failed", error=message, **common)
            self.ledger.record_event(call_id, "terminal_failure", message)
            raise HTTPCallError(message)
        if deadline_exceeded:
            message = (
                "deadline_exceeded: logical call exceeded absolute "
                f"{total_budget_s:g}s deadline"
            )
            self.ledger.update(call_id, status="deadline_exceeded", error=message, **common)
            self.ledger.record_event(
                call_id,
                "deadline_exceeded",
                message,
                {"deadline_s": total_budget_s, "completed_response": True},
            )
            raise UnresolvedCall(message)
        self.ledger.update(call_id, status="completed", error=None, **common)
        return result

    def _store_response(self, call_id: str, kind: str, response: HTTPResponse) -> str:
        try:
            body: Any = json.loads(response.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = response.body.decode("utf-8", "replace")
        return self.ledger.write_artifact(
            call_id,
            kind,
            {"status": response.status, "headers": dict(response.headers or {}), "body": body},
        )

    def _store_deadline_response(
        self, call_id: str, error: _TransportDeadlineExceeded
    ) -> str | None:
        if error.status is None and not error.headers:
            return None
        return self.ledger.write_artifact(
            call_id,
            "deadline_response",
            {
                "status": error.status,
                "headers": error.headers,
                "body": None,
                "incomplete": True,
            },
        )

    def _deadline_exceeded(
        self,
        call_id: str,
        deadline_s: float,
        message: str,
        artifact: str | None = None,
    ) -> None:
        row = self.ledger.get(call_id)
        fields: dict[str, Any] = {
            "status": "deadline_exceeded",
            "error": message,
        }
        if row is not None:
            fields["latency_s"] = max(
                float(row["latency_s"]),
                max(0.0, time.time() - float(row["created_at"])),
            )
        if artifact is not None:
            fields["response_artifact"] = artifact
        self.ledger.update(call_id, **fields)
        self.ledger.record_event(
            call_id,
            "deadline_exceeded",
            message,
            {"deadline_s": deadline_s, "completed_response": False},
        )
        raise UnresolvedCall("deadline_exceeded: " + message)

    def _unknown(self, call_id: str, message: str, artifact: str | None = None) -> None:
        fields: dict[str, Any] = {"status": "unknown", "error": message}
        if artifact is not None:
            fields["response_artifact"] = artifact
        self.ledger.update(call_id, **fields)
        self.ledger.record_event(call_id, "unknown", message)
        raise UnresolvedCall(message)


def _validate_route(route: Mapping[str, Any]) -> dict[str, Any]:
    required = {"id", "model", "api", "base_url", "api_key_env"}
    missing = required - set(route)
    if missing:
        raise ValueError(f"route missing fields: {sorted(missing)}")
    if route["api"] not in {"chat_completions", "responses"}:
        raise ValueError("route api must be 'chat_completions' or 'responses'")
    provider = route.get("provider")
    if provider is not None:
        allowed = {"only", "allow_fallbacks", "require_parameters"}
        if not isinstance(provider, Mapping) or set(provider) - allowed:
            raise ValueError("provider supports only only/allow_fallbacks/require_parameters")
    return dict(route)


def _shape_request(
    route: Mapping[str, Any], messages: list[dict[str, Any]], params: Mapping[str, Any]
) -> tuple[dict[str, Any], int]:
    allowed = {"max_tokens", "reasoning_effort", "response_format", "temperature"}
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"unsupported common params: {sorted(unknown)}")
    try:
        max_tokens = _strict_nonnegative_int(params["max_tokens"], "params.max_tokens")
    except KeyError as exc:
        raise ValueError("params.max_tokens must be a positive integer") from exc
    if max_tokens <= 0:
        raise ValueError("params.max_tokens must be a positive integer")

    body: dict[str, Any] = {"model": route["model"]}
    if route["api"] == "chat_completions":
        body["messages"] = messages
        body["max_tokens"] = max_tokens
        for name in ("reasoning_effort", "response_format", "temperature"):
            if name in params:
                body[name] = params[name]
    else:
        body["input"] = messages
        body["max_output_tokens"] = max_tokens
        if "reasoning_effort" in params:
            body["reasoning"] = {"effort": params["reasoning_effort"]}
        if "response_format" in params:
            body["text"] = {"format": _responses_text_format(params["response_format"])}
        if "temperature" in params:
            body["temperature"] = params["temperature"]
        # The Responses API has no cache_control content block; it takes a grouping hint
        # instead. Declared per route so the request stays reproducible from the spec.
        if "prompt_cache_key" in route:
            key = route["prompt_cache_key"]
            if not isinstance(key, str) or not key.strip():
                raise ValueError("route.prompt_cache_key must be a nonempty string")
            body["prompt_cache_key"] = key
    if "service_tier" in route:
        body["service_tier"] = route["service_tier"]
    if route.get("background"):
        body["background"] = True
    if "provider" in route:
        body["provider"] = dict(route["provider"])
    return body, max_tokens


def _responses_text_format(response_format: Any) -> Any:
    if not isinstance(response_format, Mapping):
        raise ValueError("response_format must be an object")
    if response_format.get("type") != "json_schema":
        return dict(response_format)
    schema = response_format.get("json_schema")
    if not isinstance(schema, Mapping):
        raise ValueError("json_schema response_format requires json_schema object")
    # Responses uses a flat text.format object, unlike Chat Completions' wrapper.
    return {"type": "json_schema", **dict(schema)}


def _post_url(route: Mapping[str, Any]) -> str:
    base = str(route["base_url"]).rstrip("/")
    return f"{base}/chat/completions" if route["api"] == "chat_completions" else f"{base}/responses"


def _responses_url(route: Mapping[str, Any]) -> str:
    return f"{str(route['base_url']).rstrip('/')}/responses"


def _response_id(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("id")
    return None if value is None else str(value)


def _is_unfinished(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("status", "")).lower() in {
        "queued",
        "in_progress",
        "pending",
        "running",
    }


def _usage(payload: Mapping[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, Mapping) else {}


def _integer_usage(usage: Mapping[str, Any], *keys: str) -> int | None:
    values: list[int] = []
    for key in keys:
        if usage.get(key) is not None:
            values.append(_strict_nonnegative_int(usage[key], f"usage field {key!r}"))
    if len(set(values)) > 1:
        raise SafetyStop(f"conflicting usage fields {keys!r}: {values}")
    return values[0] if values else None


def _strict_nonnegative_int(raw: Any, label: str) -> int:
    if isinstance(raw, bool):
        raise SafetyStop(f"{label} is not an integer token count")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SafetyStop(f"{label} is not an integer token count") from exc
    if isinstance(raw, float) and not raw.is_integer():
        raise SafetyStop(f"{label} is not an integer token count")
    if isinstance(raw, str) and raw.strip() != str(value):
        raise SafetyStop(f"{label} is not a canonical integer token count")
    if value < 0:
        raise SafetyStop(f"{label} is negative")
    return value


def _cache_token_counts(usage: Mapping[str, Any], input_tokens: int) -> tuple[int, int]:
    cached_values: list[int] = []
    for detail_key in ("prompt_tokens_details", "input_tokens_details"):
        details = usage.get(detail_key)
        if isinstance(details, Mapping) and details.get("cached_tokens") is not None:
            cached_values.append(
                _strict_nonnegative_int(details["cached_tokens"], "cached tokens")
            )
    if usage.get("cache_read_input_tokens") is not None:
        cached_values.append(
            _strict_nonnegative_int(usage["cache_read_input_tokens"], "cache-read tokens")
        )
    cache_write_raw = usage.get("cache_creation_input_tokens", 0)
    cache_write = _strict_nonnegative_int(cache_write_raw or 0, "cache-write tokens")
    if len(set(cached_values)) > 1:
        raise SafetyStop(f"conflicting cached-token counts: {cached_values}")
    cached = cached_values[0] if cached_values else 0
    if cached < 0 or cache_write < 0:
        raise SafetyStop("cache token counts cannot be negative")
    if cached + cache_write > input_tokens:
        raise SafetyStop(
            f"cached ({cached}) plus cache-write ({cache_write}) tokens exceed "
            f"input tokens ({input_tokens})"
        )
    return cached, cache_write


def _cost_from_tokens(
    input_tokens: int,
    output_tokens: int,
    rate: Rate,
    *,
    cached_tokens: int,
    cache_write_tokens: int = 0,
    reservation: bool,
) -> Decimal:
    if reservation:
        cached_rate = (
            rate.cached_input_per_million
            if rate.cached_input_per_million is not None
            else rate.input_per_million
        )
        cache_write_rate = (
            rate.cache_write_per_million
            if rate.cache_write_per_million is not None
            else rate.input_per_million
        )
        input_rate = max(
            rate.input_per_million,
            cached_rate,
            cache_write_rate,
        )
        cached_tokens = 0
        cache_write_tokens = 0
    else:
        input_rate = rate.input_per_million
    uncached = input_tokens - cached_tokens - cache_write_tokens
    cached_rate = (
        rate.cached_input_per_million
        if rate.cached_input_per_million is not None
        else input_rate
    )
    cache_write_rate = (
        rate.cache_write_per_million
        if rate.cache_write_per_million is not None
        else input_rate
    )
    return (
        Decimal(uncached) * input_rate
        + Decimal(cached_tokens) * cached_rate
        + Decimal(cache_write_tokens) * cache_write_rate
        + Decimal(output_tokens) * rate.output_per_million
    ) / Decimal(1_000_000)


def extract_visible_output_text(payload: Mapping[str, Any]) -> str:
    """Extract only assistant-visible final text from Chat or Responses output.

    Responses may put private chain-of-thought in sibling ``reasoning`` items
    whose content parts are ``reasoning_text``. Those texts remain in ``raw``
    for the private audit artifact but must never enter candidate or judge text.
    """
    direct = payload.get("output_text")
    if isinstance(direct, str):
        return direct
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            return message["content"]
    pieces: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") != "output_text":
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
    return "".join(pieces)


def repair_stored_output_text(ledger: Ledger, call_id: str) -> CallResult:
    """Repair one stored normalized result without provider or artifact mutation.

    The durable raw response is the source of truth. Only ``calls.result_json``
    is rewritten; status, charge, latency, response ID, and raw response
    artifacts are preserved. Callers repairing outcome/grade documents should
    replace their embedded ``call`` with ``returned.as_dict()`` and rerun their
    existing deterministic parser against ``returned.output_text``.
    """
    row = ledger.get(call_id)
    if row is None:
        raise KeyError(f"unknown call_id {call_id!r}")
    result = _result_from_row(row)
    repaired = replace(result, output_text=extract_visible_output_text(result.raw))
    if repaired != result:
        ledger.update(
            call_id,
            result_json=_canonical_bytes(repaired.as_dict()).decode("utf-8"),
        )
        ledger.record_event(
            call_id,
            "output_text_repaired",
            "normalized output_text rebuilt from visible response content",
        )
    return repaired


def _decode_response(response: HTTPResponse) -> dict[str, Any]:
    try:
        payload = json.loads(response.body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPCallError(f"HTTP {response.status} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPCallError(f"HTTP {response.status} returned a non-object JSON body")
    return payload


def _result_from_row(row: Mapping[str, Any]) -> CallResult:
    raw = row.get("result_json")
    if not raw:
        raise UnresolvedCall(f"completed call {row['call_id']!r} has no reusable result")
    value = json.loads(raw)
    return CallResult(**value)


def _stdlib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout: float,
    *,
    _deadline_state: _DeadlineState | None = None,
) -> HTTPResponse:
    """One urllib exchange with an optional response-closing watchdog."""
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if _deadline_state is not None:
                _deadline_state.attach(response)
            try:
                return HTTPResponse(
                    status=int(response.status),
                    body=response.read(),
                    headers=dict(response.headers.items()),
                )
            finally:
                if _deadline_state is not None:
                    _deadline_state.detach(response)
    except urllib.error.HTTPError as exc:
        if _deadline_state is not None:
            _deadline_state.attach(exc)
        try:
            return HTTPResponse(
                status=int(exc.code),
                body=exc.read(),
                headers=dict(exc.headers.items()) if exc.headers else {},
            )
        finally:
            if _deadline_state is not None:
                _deadline_state.detach(exc)
