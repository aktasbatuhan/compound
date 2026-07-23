"""Guard against a stale global package shadowing a pinned dependency.

A `gepa` installed in the user site directory (``~/.local/lib/python3.12/
site-packages``) can shadow the version pinned in ``pyproject.toml`` whenever
the test run does not actually use the project's virtualenv — most easily when
``uv run pytest`` falls back to a globally installed pytest because the ``dev``
extra was never synced.

The symptom is deeply misleading: the optimizer tests fail with
``TypeError: optimize() got an unexpected keyword argument 'frontier_type'``,
which reads like the GEPA library changed its API under us. It has not. The
wrong GEPA is being imported. This exact confusion cost a debugging session on
2026-07-23.

These tests turn that into a clear message. If one fails, run:

    uv sync --extra dev
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

import pytest

PINNED_GEPA_VERSION = "0.1.4"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_gepa_is_imported_from_the_project_environment() -> None:
    """The imported gepa must live inside the interpreter running the tests."""
    import gepa

    gepa_path = Path(gepa.__file__).resolve()
    prefix = Path(sys.prefix).resolve()

    assert gepa_path.is_relative_to(prefix), (
        f"gepa was imported from {gepa_path}, which is outside this interpreter's "
        f"environment ({prefix}). A stale global install is shadowing the pinned "
        f"version. Run `uv sync --extra dev` and re-run through the project venv."
    )


def test_gepa_matches_the_pinned_version() -> None:
    """The imported gepa must be the version pyproject.toml pins."""
    installed = metadata.version("gepa")
    assert installed == PINNED_GEPA_VERSION, (
        f"gepa {installed} is installed but pyproject.toml pins "
        f"{PINNED_GEPA_VERSION}. If this is not a deliberate upgrade, a stale "
        f"environment is in use; run `uv sync --extra dev`."
    )


def test_gepa_exposes_the_api_the_engine_calls() -> None:
    """Fail with a readable message rather than a TypeError deep in a run."""
    import inspect

    import gepa.api
    from gepa import EvaluationBatch

    optimize_params = inspect.signature(gepa.api.optimize).parameters
    assert "frontier_type" in optimize_params, (
        "gepa.api.optimize has no 'frontier_type' parameter, so this is not the "
        f"pinned {PINNED_GEPA_VERSION} API. Check for a shadowing install."
    )

    batch_fields = getattr(EvaluationBatch, "__dataclass_fields__", {})
    assert "objective_scores" in batch_fields, (
        "gepa's EvaluationBatch has no 'objective_scores' field, so this is not "
        f"the pinned {PINNED_GEPA_VERSION} API. Check for a shadowing install."
    )


def test_pinned_version_here_matches_pyproject() -> None:
    """Keep this guard honest when the pin is deliberately changed."""
    pyproject = (_project_root() / "pyproject.toml").read_text()
    expected = f'"gepa=={PINNED_GEPA_VERSION}"'
    assert expected in pyproject, (
        f"pyproject.toml no longer pins gepa=={PINNED_GEPA_VERSION}; update "
        f"PINNED_GEPA_VERSION in {Path(__file__).name} to match."
    )


@pytest.mark.parametrize("module_name", ["compound", "compound.gepa_v2"])
def test_engine_modules_come_from_this_checkout(module_name: str) -> None:
    """The engine under test must be this working tree, not another checkout."""
    module = __import__(module_name, fromlist=["__file__"])
    module_path = Path(module.__file__).resolve()
    root = _project_root()

    assert module_path.is_relative_to(root), (
        f"{module_name} was imported from {module_path}, outside this checkout "
        f"({root}). Tests would be exercising different code than you are editing."
    )
