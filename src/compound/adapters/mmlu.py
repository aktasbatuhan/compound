"""MMLU adapter: multiple-choice knowledge, graded by exact letter match.

The manifest is self-contained: ``build_manifest`` samples a fixed number of
test questions per subject from the Hugging Face datasets server and embeds
question, choices, and answer in case metadata, so runs are reproducible and
need no dataset dependency. Partitions follow the house 50/20/30 split, stably
hashed per case id, so the decision slice never moves between rebuilds.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

HF_ROWS = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=cais%2Fmmlu&config={subject}&split=test&offset=0&length={n}"
)
HF_SPLITS = "https://datasets-server.huggingface.co/splits?dataset=cais%2Fmmlu"
NON_SUBJECT_CONFIGS = {"all", "auxiliary_train"}
LETTERS = "ABCD"

# House partition split (matches the tau/bfcl manifests' proportions).
PARTITION_WEIGHTS = (
    ("optimizer_train", 5),
    ("optimizer_validation", 2),
    ("decision_test", 3),
)


def stable_partition(case_id: str) -> str:
    """Deterministic 50/20/30 assignment; independent of insertion order."""
    bucket = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 10
    for name, weight in PARTITION_WEIGHTS:
        if bucket < weight:
            return name
        bucket -= weight
    raise AssertionError("unreachable")


def _fetch_json(url: str, attempts: int = 6) -> dict:
    """GET with backoff; the anonymous datasets-server rate-limits bursts."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if attempt == attempts - 1:
                raise
            retry_after = None
            if isinstance(error, urllib.error.HTTPError):
                retry_after = error.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 3 * 2**attempt)
    raise AssertionError("unreachable")


def build_manifest(
    output_path: str | Path,
    *,
    per_subject: int = 5,
    fetch: Callable[[str], dict] = _fetch_json,
) -> Path:
    splits = fetch(HF_SPLITS)
    subjects = sorted(
        {s["config"] for s in splits["splits"]} - NON_SUBJECT_CONFIGS
    )
    cases = []
    for i, subject in enumerate(subjects):
        if i and fetch is _fetch_json:
            time.sleep(1.0)  # stay under the anonymous rate limit
        print(f"[{i + 1}/{len(subjects)}] {subject}")
        rows = fetch(HF_ROWS.format(subject=subject, n=per_subject))["rows"]
        for row in rows:
            payload = row["row"]
            case_id = f"mmlu:{subject}:{row['row_idx']}"
            cases.append(
                {
                    "case_id": case_id,
                    "partition": stable_partition(case_id),
                    "metadata": {
                        "subject": subject,
                        "question": payload["question"],
                        "choices": payload["choices"],
                        "answer": int(payload["answer"]),
                    },
                }
            )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"benchmark": "mmlu", "source": "cais/mmlu test split", "cases": cases},
            indent=1,
        )
    )
    return destination


def render_prompt(case: dict) -> str:
    meta = case["metadata"]
    options = "\n".join(
        f"{LETTERS[i]}. {choice}" for i, choice in enumerate(meta["choices"])
    )
    return (
        f"{meta['question']}\n\n{options}\n\n"
        "Answer with only the letter of the correct choice (A, B, C, or D)."
    )


def parse_letter(text: str) -> str | None:
    """Last standalone choice letter in the reply.

    Models often reason before concluding ("A is wrong ... the answer is C"),
    so the final letter mention is the verdict, not the first.
    """
    matches = re.findall(r"\b([ABCD])\b", text.strip().upper())
    return matches[-1] if matches else None


def grade(case: dict, response_text: str) -> bool:
    return parse_letter(response_text) == LETTERS[case["metadata"]["answer"]]


def run_mmlu(
    cases: list[dict],
    *,
    model: str,
    base_url: str,
    api_key_env: str,
    upstream: str | None = None,
    max_tokens: int = 512,
    output_path: str | Path,
) -> dict[str, Any]:
    """One completion per case through any OpenAI-compatible endpoint."""
    from compound.providers import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(
        name="mmlu-runner", base_url=base_url, api_key_env=api_key_env
    )
    extra_body = (
        {"provider": {"only": [upstream], "allow_fallbacks": False}} if upstream else None
    )
    results = []
    correct = 0
    for case in cases:
        started = time.time()
        response = provider.complete(
            model=model,
            messages=[{"role": "user", "content": render_prompt(case)}],
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        message = (response.output.get("choices") or [{}])[0].get("message", {})
        # Reasoning models may spend the budget in a separate reasoning field
        # and leave content empty; the verdict letter can live in either.
        text = message.get("content") or message.get("reasoning") or ""
        passed = grade(case, text)
        correct += passed
        results.append(
            {
                "case_id": case["case_id"],
                "passed": passed,
                "predicted": parse_letter(text),
                "response_tail": text[-120:],
                "expected": LETTERS[case["metadata"]["answer"]],
                "latency_ms": response.latency_ms,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "seconds": round(time.time() - started, 2),
            }
        )
        print(f"{'PASS' if passed else 'fail'} {case['case_id']}")
    summary = {
        "benchmark": "mmlu",
        "model": model,
        "upstream": upstream,
        "accuracy": correct / len(cases),
        "correct": correct,
        "total": len(cases),
        "results": results,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=1))
    print(f"\naccuracy: {correct}/{len(cases)} ({correct / len(cases):.1%}) -> {destination}")
    return summary
