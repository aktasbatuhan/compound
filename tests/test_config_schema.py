from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from compound.config_schema import (
    ConfigSchemaError,
    iter_schema_errors,
    load_schema,
    schema_path,
    validate_config_file,
    validate_config_schema,
)


def _base_config() -> dict[str, Any]:
    return {
        "version": 1,
        "artifacts_dir": "artifacts",
        "manifests_dir": "benchmarks/manifests",
        "benchmarks": {
            "ds1000": {
                "task_key": "data_processing",
                "sample_count": 4,
                "partitions": {"decision_test": 4},
            }
        },
    }


def test_committed_schema_is_the_generated_v1_artifact() -> None:
    schema = load_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "compound.config.v1" in schema["$id"]
    assert "do not edit by hand" in schema["description"]
    assert json.loads(schema_path().read_text()) == schema


def test_real_compound_yaml_passes() -> None:
    config = validate_config_file("compound.yaml")
    assert config["version"] == 1
    assert set(config["benchmarks"]) == {"ds1000", "bfcl", "tau_bench"}


def test_missing_required_benchmark_keys_fails() -> None:
    config = _base_config()
    del config["benchmarks"]["ds1000"]["task_key"]
    with pytest.raises(ConfigSchemaError) as excinfo:
        validate_config_schema(config)
    assert "benchmarks.ds1000" in str(excinfo.value)


def test_missing_benchmarks_section_fails() -> None:
    config = _base_config()
    del config["benchmarks"]
    with pytest.raises(ConfigSchemaError, match="benchmarks"):
        validate_config_schema(config)


def test_product_sections_are_optional() -> None:
    validate_config_schema(_base_config())


def test_valid_product_sections_pass() -> None:
    config = _base_config()
    config["task_keys"] = {
        "support_agent": {
            "description": "Customer support chat agent",
            "replay": {
                "default_tool_policy": "recorded",
                "per_tool": {"issue_refund": "blocked"},
            },
        }
    }
    config["redaction"] = {
        "rules": [
            {"name": "api_keys", "applies_to": ["steps[*].input"], "detector": "secret"},
            {
                "name": "order_ids",
                "applies_to": ["steps[*].input"],
                "detector": "regex",
                "pattern": "ORD-[0-9]{6}",
            },
        ],
        "field_allowlist": ["metadata.environment"],
    }
    config["ingest"] = {
        "default_permissions": {"judging": True, "optimization": True, "fine_tuning": False},
        "sources": [{"name": "langfuse-prod", "importer": "langfuse", "path": "exports"}],
    }
    validate_config_schema(config)


def test_bad_task_keys_replay_policy_fails() -> None:
    config = _base_config()
    config["task_keys"] = {"support_agent": {"replay": {"default_tool_policy": "replayed"}}}
    with pytest.raises(ConfigSchemaError) as excinfo:
        validate_config_schema(config)
    assert "task_keys.support_agent.replay.default_tool_policy" in str(excinfo.value)


def test_bad_per_tool_replay_policy_fails() -> None:
    config = _base_config()
    config["task_keys"] = {
        "support_agent": {
            "replay": {"default_tool_policy": "recorded", "per_tool": {"search": "cached"}}
        }
    }
    with pytest.raises(ConfigSchemaError, match="per_tool.search"):
        validate_config_schema(config)


def test_regex_rule_without_pattern_fails() -> None:
    config = _base_config()
    config["redaction"] = {
        "rules": [{"name": "order_ids", "applies_to": ["steps[*].input"], "detector": "regex"}]
    }
    assert iter_schema_errors(config)


def test_unknown_key_in_product_section_fails() -> None:
    config = _base_config()
    config["ingest"] = {
        "default_permissions": {"judging": True, "optimization": True, "fine_tuning": False},
        "sources": [{"name": "langfuse-prod", "importer": "langfuse", "project": "prod"}],
    }
    assert iter_schema_errors(config)


def test_unknown_benchmark_keys_are_allowed() -> None:
    config = _base_config()
    config["benchmarks"]["ds1000"]["trials"] = 3
    config["future_section"] = {"anything": True}
    validate_config_schema(config)


def test_error_message_names_the_source() -> None:
    config = copy.deepcopy(_base_config())
    config["version"] = 2
    with pytest.raises(ConfigSchemaError, match="my-config.yaml"):
        validate_config_schema(config, source="my-config.yaml")
