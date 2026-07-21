from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from compound.catalogs import load_bfcl_catalog, load_ds1000_catalog, load_tau_catalog
from compound.contracts import Partition
from compound.manifest import SourceCase, assign_partitions, stratified_sample, write_manifest


def prepare_manifests(
    config: dict[str, Any],
    *,
    sources_dir: str | Path = ".compound/sources",
    output_dir: str | Path = "artifacts/manifests",
) -> dict[str, Path]:
    sources = Path(sources_dir)
    destination = Path(output_dir)
    loaders = {
        "ds1000": (load_ds1000_catalog, sources / "ds1000"),
        "bfcl": (load_bfcl_catalog, sources / "gorilla"),
        "tau_bench": (load_tau_catalog, sources / "tau2-bench"),
    }
    written: dict[str, Path] = {}
    seed = int(config["seed"])
    for benchmark, benchmark_config in config["benchmarks"].items():
        loader, source = loaders[benchmark]
        catalog = loader(source)
        selected = stratified_sample(
            catalog,
            benchmark=benchmark,
            seed=seed,
            count=int(benchmark_config["sample_count"]),
        )
        sizes = benchmark_config["partitions"]
        partitions = assign_partitions(
            selected,
            train=int(sizes["optimizer_train"]),
            validation=int(sizes["optimizer_validation"]),
            decision=int(sizes["decision_test"]),
        )
        output = destination / f"{benchmark}.json"
        write_manifest(
            output,
            benchmark=benchmark,
            revision=str(benchmark_config["revision"]),
            seed=seed,
            cases=selected,
            partitions=partitions,
        )
        written[benchmark] = output
    return written


def prepare_scaled_ds1000_manifest(
    *,
    source_dir: str | Path,
    output_path: str | Path,
    revision: str,
    seed: int,
    count: int,
    train: int,
    validation: int,
    decision: int,
    excluded_case_ids: set[str],
    strata: set[str] | None = None,
) -> Path:
    catalog = load_ds1000_catalog(source_dir)
    allowed = strata or {"numpy", "pandas"}
    eligible: list[SourceCase] = [
        case
        for case in catalog
        if case.stratum in allowed and case.case_id not in excluded_case_ids
    ]
    selected = stratified_sample(
        eligible,
        benchmark="ds1000-scale",
        seed=seed,
        count=count,
    )
    partitions = assign_partitions(
        selected,
        train=train,
        validation=validation,
        decision=decision,
    )
    destination = Path(output_path)
    write_manifest(
        destination,
        benchmark="ds1000",
        revision=revision,
        seed=seed,
        cases=selected,
        partitions=partitions,
    )
    payload = json.loads(destination.read_text())
    payload["excluded_case_ids"] = sorted(excluded_case_ids)
    payload["purpose"] = "fresh scaled recursive-optimization proof"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def prepare_gepa_v2_ds1000_manifest(
    *,
    source_dir: str | Path,
    output_path: str | Path,
    revision: str,
    seed: int,
    train: int,
    validation: int,
    decision: int,
    excluded_case_ids: set[str],
    excluded_origin_groups: set[str],
) -> Path:
    """Create a fresh NumPy/Pandas cohort with no perturbation-family leakage.

    DS-1000 contains rewritten variants of the same underlying problem. This
    sampler admits at most one case from each (library, perturbation origin)
    group and excludes every group touched by an earlier proof.
    """
    catalog = load_ds1000_catalog(source_dir)
    grouped: dict[str, list[SourceCase]] = defaultdict(list)
    for case in catalog:
        if case.stratum not in {"numpy", "pandas"}:
            continue
        origin_group = ds1000_origin_group(case)
        if case.case_id in excluded_case_ids or origin_group in excluded_origin_groups:
            continue
        grouped[origin_group].append(case)

    eligible = [
        min(
            variants,
            key=lambda case: hashlib.sha256(
                f"{seed}:origin-representative:{case.case_id}".encode()
            ).hexdigest(),
        )
        for variants in grouped.values()
    ]

    count = train + validation + decision
    selected = stratified_sample(
        eligible,
        benchmark="ds1000-gepa-v2",
        seed=seed,
        count=count,
    )
    partitions = assign_partitions(
        selected,
        train=train,
        validation=validation,
        decision=decision,
    )
    destination = Path(output_path)
    write_manifest(
        destination,
        benchmark="ds1000",
        revision=revision,
        seed=seed,
        cases=selected,
        partitions=partitions,
    )
    payload = json.loads(destination.read_text())
    payload.update(
        {
            "excluded_case_ids": sorted(excluded_case_ids),
            "excluded_origin_groups": sorted(excluded_origin_groups),
            "origin_group_policy": "one case per library:perturbation_origin_id",
            "purpose": "fresh group-isolated GEPA v2 optimization proof",
        }
    )
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def prepare_ds1000_baseline_matrix_manifest(
    *,
    source_dir: str | Path,
    output_path: str | Path,
    revision: str,
    seed: int,
    count: int,
    excluded_case_ids: set[str],
    excluded_origin_groups: set[str],
) -> Path:
    destination = prepare_gepa_v2_ds1000_manifest(
        source_dir=source_dir,
        output_path=output_path,
        revision=revision,
        seed=seed,
        train=count,
        validation=0,
        decision=0,
        excluded_case_ids=excluded_case_ids,
        excluded_origin_groups=excluded_origin_groups,
    )
    payload = json.loads(destination.read_text())
    payload.update(
        {
            "cohort_role": "opened_baseline",
            "purpose": "six-model pre-optimization baseline matrix; never a decision set",
        }
    )
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def prepare_ds1000_gepa_decision_manifest(
    *,
    source_dir: str | Path,
    output_path: str | Path,
    revision: str,
    seed: int,
    count: int,
    excluded_case_ids: set[str],
    excluded_origin_groups: set[str],
) -> Path:
    """Freeze a decision-only cohort that no optimizer loader can access."""
    destination = prepare_gepa_v2_ds1000_manifest(
        source_dir=source_dir,
        output_path=output_path,
        revision=revision,
        seed=seed,
        train=0,
        validation=0,
        decision=count,
        excluded_case_ids=excluded_case_ids,
        excluded_origin_groups=excluded_origin_groups,
    )
    payload = json.loads(destination.read_text())
    payload.update(
        {
            "cohort_role": "sealed_decision",
            "purpose": "fresh group-isolated post-GEPA decision gate; never optimizer input",
        }
    )
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def prepare_gepa_manifest_from_opened_baseline(
    *,
    baseline_manifest_path: str | Path,
    output_path: str | Path,
    train: int,
    validation: int,
) -> Path:
    source = Path(baseline_manifest_path)
    payload = json.loads(source.read_text())
    if payload.get("cohort_role") != "opened_baseline":
        raise ValueError("source manifest must be an opened_baseline cohort")
    cases = payload.get("cases") or []
    if len(cases) != train + validation:
        raise ValueError("train and validation sizes must cover the opened baseline exactly")
    if train < 1 or validation < 1:
        raise ValueError("opened GEPA manifest requires nonempty train and validation partitions")
    rewritten = []
    for index, case in enumerate(cases):
        record = dict(case)
        record["partition"] = (
            Partition.OPTIMIZER_TRAIN.value
            if index < train
            else Partition.OPTIMIZER_VALIDATION.value
        )
        rewritten.append(record)
    payload.update(
        {
            "cases": rewritten,
            "cohort_role": "opened_optimization",
            "purpose": "GEPA train/validation split derived only from the opened baseline cohort",
            "source_manifest": str(baseline_manifest_path),
            "decision_cases": 0,
        }
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def open_ds1000_decision_manifest_for_diagnostics(
    *,
    decision_manifest_path: str | Path,
    decision_report_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Relabel a completed decision cohort as opened, preserving its audit lineage."""
    source = Path(decision_manifest_path)
    report_path = Path(decision_report_path)
    payload = json.loads(source.read_text())
    report = json.loads(report_path.read_text())
    if payload.get("cohort_role") != "sealed_decision":
        raise ValueError("source manifest must have cohort_role=sealed_decision")
    if report.get("optimizer_accessed_decision_cases") is not False:
        raise ValueError("decision report does not prove the optimizer firewall")
    reported_manifest = Path(str(report.get("decision_manifest")))
    if reported_manifest.resolve() != source.resolve():
        raise ValueError("decision report does not reference the source manifest")
    source_ids = [str(case["case_id"]) for case in payload.get("cases") or []]
    if set(source_ids) != set(report.get("case_ids") or []):
        raise ValueError("decision report cases do not match the source manifest")
    payload["cases"] = [
        {**case, "partition": Partition.OPTIMIZER_TRAIN.value}
        for case in payload["cases"]
    ]
    payload.update(
        {
            "cohort_role": "opened_baseline",
            "purpose": "post-decision opened diagnostic cohort; never a future decision set",
            "source_manifest": str(decision_manifest_path),
            "source_decision_report": str(decision_report_path),
        }
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def ds1000_origin_group(case: SourceCase) -> str:
    return f"{case.stratum}:{case.metadata['perturbation_origin_id']}"
