"""Exercise real retail opening requests through the budget gate, with no network."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from compound.agentic_gateway import Gateway
from compound.agentic_safety import runtime_identity
from compound.agentic_study import fingerprint, plan
from compound.agentic_worker import retail


class ReachedTransport(BaseException):
    pass


class OpeningCaptured(BaseException):
    pass


def qualify(spec, out):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import litellm

    study = plan(spec)
    checks = []
    for episode in study["episodes"]:
        if episode["trial"] != 0:
            continue
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gateway = Gateway(spec, study, root, limit=9.40)

            def no_network(request, timeout):
                payload = json.loads(request.data)
                assert payload.get("temperature") == spec["controls"].get("temperature")
                raise ReachedTransport()

            def completion(*args, gateway=gateway, episode=episode, **kwargs):
                role = "auxiliary" if "/auxiliary/" in kwargs["api_base"] else "agent"
                body = {k: kwargs[k] for k in ("messages", "tools", "max_tokens") if k in kwargs}
                try:
                    gateway.call(episode["episode_id"], role, body)
                except ReachedTransport:
                    entry = gateway.guard.entries[-1]
                    checks.append(
                        {
                            "task_id": episode["task_id"],
                            "tier": episode["tier"],
                            "role": role,
                            "admitted": True,
                            "reservation_usd": entry["reservation_usd"],
                        }
                    )
                else:
                    raise AssertionError("request did not reach transport")
                if role == "agent":
                    raise OpeningCaptured()
                # Deliberately long synthetic user opening; no paid simulator call.
                return litellm.ModelResponse(
                    model="test-only",
                    choices=[
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "I need help with my order. " * 100,
                            },
                        }
                    ],
                    usage={"prompt_tokens": 1000, "completion_tokens": 600, "total_tokens": 1600},
                )

            with (
                patch.dict(
                    os.environ,
                    {"OPENROUTER_API_KEY": "offline-only", "DOUBLEWORD_API_KEY": "offline-only"},
                ),
                patch("urllib.request.urlopen", no_network),
                patch("litellm.completion", completion),
            ):
                try:
                    retail(
                        spec,
                        episode,
                        "http://127.0.0.1:1/" + episode["episode_id"] + "/agent/v1",
                        root,
                    )
                except OpeningCaptured:
                    pass
                else:
                    raise AssertionError("harness never requested an agent response")
    expected = len(spec["sources"]["retail"]["tasks"]) * 4
    result = {
        "spec_sha256": fingerprint(spec),
        "runtime": runtime_identity(),
        "ok": len(checks) == expected,
        "checks": checks,
        "scope": (
            "Real harness opening requests with a synthetic 2600-character "
            "simulator response; "
            "no network. Not proof of entire-episode affordability."
        ),
    }
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    assert result["ok"]
    print(f"Admission passed: {len(checks)} opening requests; no paid calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    qualify(json.loads(Path(args.spec).read_text()), args.out)
