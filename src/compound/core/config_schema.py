"""Validate ``compound.yaml`` against the shared JSON Schema artifact.

The schema is generated from the zod schemas in ``packages/config`` and
committed at ``packages/config/schema/compound.config.v1.schema.json``. Both
languages validate against that one definition; this module is the Python half.

Never edit the schema file by hand — run ``bun run generate:schema`` in
``packages/config``.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema
import yaml

SCHEMA_RELATIVE_PATH = Path("packages/config/schema/compound.config.v1.schema.json")
SCHEMA_PATH_ENV = "COMPOUND_CONFIG_SCHEMA"


class ConfigSchemaError(ValueError):
    """Raised when a config does not satisfy the shared JSON Schema."""


def schema_path() -> Path:
    """Locate the committed JSON Schema artifact.

    ``COMPOUND_CONFIG_SCHEMA`` wins if set; otherwise the repository root is
    found by walking up from this file (``src/compound/`` -> repo root).
    """
    override = os.environ.get(SCHEMA_PATH_ENV)
    if override:
        candidate = Path(override)
        if not candidate.is_file():
            raise ConfigSchemaError(f"{SCHEMA_PATH_ENV}={override} is not a file")
        return candidate
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / SCHEMA_RELATIVE_PATH
        if candidate.is_file():
            return candidate
    raise ConfigSchemaError(
        f"cannot find {SCHEMA_RELATIVE_PATH}; set {SCHEMA_PATH_ENV} to its location"
    )


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    return json.loads(schema_path().read_text())


def _format_path(error: jsonschema.ValidationError) -> str:
    path = ""
    for part in error.absolute_path:
        if isinstance(part, int):
            path += f"[{part}]"
        elif path:
            path += f".{part}"
        else:
            path = str(part)
    return path or "<root>"


def iter_schema_errors(data: Any) -> list[str]:
    """Return path-qualified messages for every schema violation (may be empty)."""
    validator = jsonschema.Draft202012Validator(load_schema())
    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: [str(part) for part in error.absolute_path],
    )
    return [f"{_format_path(error)}: {error.message}" for error in errors]


def validate_config_schema(data: Any, *, source: str = "compound.yaml") -> None:
    """Raise :class:`ConfigSchemaError` if ``data`` violates the shared schema."""
    errors = iter_schema_errors(data)
    if errors:
        joined = "\n".join(f"  - {message}" for message in errors)
        raise ConfigSchemaError(f"{source} does not match the shared config schema:\n{joined}")


def validate_config_file(path: str | Path = "compound.yaml") -> dict[str, Any]:
    """Parse a config file and validate it against the shared schema."""
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    validate_config_schema(data, source=str(config_path))
    return data
