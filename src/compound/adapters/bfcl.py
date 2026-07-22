from __future__ import annotations

import copy
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from compound.contracts import Partition

# The pinned gorilla checkout keeps the BFCL data files inside the package tree.
BFCL_DATA_SUBPATH = Path("berkeley-function-call-leaderboard") / "bfcl_eval" / "data"

# BFCL's AST checker consults MODEL_CONFIG_MAPPING[model_name].underscore_to_dot
# to decide whether dotted ground-truth function names were flattened for the
# model. Compound renders prompting-mode completions, where dots survive, so we
# grade under a pinned prompting-compatible identity (underscore_to_dot=False)
# instead of leaking provider-specific model ids into the official checker.
BFCL_CHECKER_MODEL_NAME = "gorilla-openfunctions-v2"

SINGLE_TURN_STRATUM = "single_turn"
MULTI_TURN_STRATUM = "multi_turn"


@dataclass(frozen=True, slots=True)
class BFCLCase:
    case_id: str
    category: str
    stratum: str
    partition: Partition
    question: list[list[dict[str, Any]]]
    functions: list[dict[str, Any]]
    ground_truth: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def write_bfcl_run_ids(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    partitions: set[Partition],
) -> dict[str, list[str]]:
    """Translate Compound partitions into BFCL's native selective-run file."""
    manifest = json.loads(Path(manifest_path).read_text())
    allowed = {partition.value for partition in partitions}
    categories: dict[str, list[str]] = defaultdict(list)
    for case in manifest["cases"]:
        if case["partition"] not in allowed:
            continue
        category = case["metadata"]["category"]
        categories[category].append(case["case_id"])
    payload = {category: ids for category, ids in sorted(categories.items())}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _read_json_lines_by_id(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    with path.open() as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                entries[str(item["id"])] = item
    return entries


def load_bfcl_cases(
    source_dir: str | Path,
    manifest_path: str | Path,
    *,
    strata: set[str] | None = None,
) -> list[BFCLCase]:
    """Materialize frozen manifest cases from the pinned gorilla checkout."""
    data_dir = Path(source_dir) / BFCL_DATA_SUBPATH
    manifest = json.loads(Path(manifest_path).read_text())
    selected = [
        case for case in manifest["cases"] if strata is None or case["stratum"] in strata
    ]
    categories = sorted({case["metadata"]["category"] for case in selected})
    prompts_by_category: dict[str, dict[str, dict[str, Any]]] = {}
    answers_by_category: dict[str, dict[str, dict[str, Any]]] = {}
    for category in categories:
        prompts_by_category[category] = _read_json_lines_by_id(
            data_dir / f"BFCL_v4_{category}.json"
        )
        answer_path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
        answers_by_category[category] = (
            _read_json_lines_by_id(answer_path) if answer_path.exists() else {}
        )

    cases: list[BFCLCase] = []
    for case in selected:
        case_id = str(case["case_id"])
        category = str(case["metadata"]["category"])
        entry = prompts_by_category[category].get(case_id)
        if entry is None:
            raise KeyError(f"BFCL case not found in pinned source data: {case_id}")
        answer = answers_by_category[category].get(case_id)
        cases.append(
            BFCLCase(
                case_id=case_id,
                category=category,
                stratum=str(case["stratum"]),
                partition=Partition(case["partition"]),
                question=entry["question"],
                # Multi-turn entries carry no top-level function docs; their
                # tools are derived from involved_classes by BFCL's harness.
                functions=entry.get("function") or [],
                ground_truth=answer["ground_truth"] if answer else None,
                metadata=dict(case["metadata"]),
            )
        )
    return cases


def _require_bfcl_eval() -> None:
    try:
        import bfcl_eval  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "the bfcl-eval package is required for BFCL rendering and grading; "
            "install it with: uv sync --extra bfcl"
        ) from error


def render_bfcl_single_turn_prompt(case: BFCLCase) -> tuple[str, str]:
    """Render a case with BFCL's own default prompting-mode system prompt."""
    _require_bfcl_eval()
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model

    if case.stratum != SINGLE_TURN_STRATUM:
        raise ValueError(f"case is not single-turn: {case.case_id}")
    if len(case.question) != 1:
        raise ValueError(f"single-turn case has {len(case.question)} turns: {case.case_id}")
    messages = system_prompt_pre_processing_chat_model(
        copy.deepcopy(case.question[0]), case.functions, case.case_id
    )
    if (
        len(messages) != 2
        or messages[0]["role"] != "system"
        or messages[1]["role"] != "user"
    ):
        raise ValueError(f"unexpected rendered message shape for case: {case.case_id}")
    return str(messages[0]["content"]), str(messages[1]["content"])


def grade_bfcl_single_turn(case: BFCLCase, completion: str) -> dict[str, Any]:
    """Grade a prompting-mode completion with BFCL's official AST checker."""
    _require_bfcl_eval()
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting
    from bfcl_eval.utils import is_function_calling_format_output

    if case.ground_truth is None:
        raise ValueError(f"case has no possible-answer ground truth: {case.case_id}")
    if not completion.strip():
        return {
            "score": 0.0,
            "feedback": "Model returned no final completion.",
            "error_type": "empty_completion",
        }
    try:
        decoded = default_decode_ast_prompting(completion, ReturnFormat.PYTHON)
    except Exception as error:
        return {
            "score": 0.0,
            "feedback": f"Invalid syntax. Failed to decode AST. {error}",
            "error_type": "ast_decoder:decoder_failed",
        }
    if not is_function_calling_format_output(decoded):
        return {
            "score": 0.0,
            "feedback": "Did not output in the specified function-calling format.",
            "error_type": "ast_decoder:decoder_wrong_output_format",
        }
    result = ast_checker(
        copy.deepcopy(case.functions),
        copy.deepcopy(decoded),
        copy.deepcopy(case.ground_truth),
        Language.PYTHON,
        case.category,
        BFCL_CHECKER_MODEL_NAME,
    )
    if result["valid"]:
        return {
            "score": 1.0,
            "feedback": "All function calls matched a possible answer.",
            "error_type": None,
        }
    errors = result.get("error") or ["checker did not provide error details"]
    return {
        "score": 0.0,
        "feedback": " ".join(str(item) for item in errors),
        "error_type": str(result.get("error_type") or "ast_checker:invalid"),
    }
