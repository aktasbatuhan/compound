from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from compound.baseline_matrix import run_ds1000_baseline_matrix
from compound.bfcl_matrix import run_bfcl_baseline_matrix
from compound.config import load_config
from compound.ds1000_cache import migrate_legacy_ds1000_cache, regrade_ds1000_run
from compound.env import load_env
from compound.flex_smoke import run_doubleword_flex_smoke
from compound.gepa_v2_runner import evaluate_gepa_v2_decision, probe_gepa_v2, run_gepa_v2
from compound.prepare import (
    ds1000_origin_group,
    open_ds1000_decision_manifest_for_diagnostics,
    prepare_ds1000_baseline_matrix_manifest,
    prepare_ds1000_gepa_decision_manifest,
    prepare_gepa_manifest_from_opened_baseline,
    prepare_gepa_v2_ds1000_manifest,
    prepare_manifests,
    prepare_scaled_ds1000_manifest,
)
from compound.proof import evaluate_ds1000_decision_test, run_ds1000_proof
from compound.telemetry import summarize_model_calls


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compound")
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate = subcommands.add_parser("validate-config")
    validate.add_argument("--config", default="compound.yaml")
    plan = subcommands.add_parser("plan")
    plan.add_argument("--config", default="compound.yaml")
    prepare = subcommands.add_parser("prepare-manifests")
    prepare.add_argument("--config", default="compound.yaml")
    prepare.add_argument("--sources-dir", default=".compound/sources")
    prepare.add_argument("--output-dir", default=None)
    proof = subcommands.add_parser("run-ds1000-proof")
    proof.add_argument("--config", default="compound.yaml")
    proof.add_argument("--max-metric-calls", type=int, default=20)
    proof.add_argument("--output-dir", default=None)
    proof.add_argument("--manifest", default=None)
    proof.add_argument("--candidate-max-tokens", type=int, default=1024)
    proof.add_argument("--reflection-max-tokens", type=int, default=1400)
    scaled = subcommands.add_parser("prepare-ds1000-scale")
    scaled.add_argument("--config", default="compound.yaml")
    scaled.add_argument("--output", default="benchmarks/manifests/ds1000_scale25.json")
    scaled.add_argument("--count", type=int, default=25)
    scaled.add_argument("--train", type=int, default=12)
    scaled.add_argument("--validation", type=int, default=6)
    scaled.add_argument("--decision", type=int, default=7)
    scaled.add_argument("--exclude-run", action="append", default=[])
    v2_prepare = subcommands.add_parser("prepare-ds1000-gepa-v2")
    v2_prepare.add_argument("--config", default="compound.yaml")
    v2_prepare.add_argument("--output", default="benchmarks/manifests/ds1000_gepa_v2.json")
    v2_prepare.add_argument("--train", type=int, default=32)
    v2_prepare.add_argument("--validation", type=int, default=16)
    v2_prepare.add_argument("--decision", type=int, default=20)
    v2_prepare.add_argument(
        "--exclude-manifest", action="append", default=[], dest="exclude_manifests"
    )
    opened_prepare = subcommands.add_parser("prepare-ds1000-gepa-opened")
    opened_prepare.add_argument(
        "--baseline-manifest", default="benchmarks/manifests/ds1000_baseline25.json"
    )
    opened_prepare.add_argument(
        "--output", default="benchmarks/manifests/ds1000_gepa_opened25.json"
    )
    opened_prepare.add_argument("--train", type=int, default=15)
    opened_prepare.add_argument("--validation", type=int, default=10)
    opened_prepare.add_argument("--config", default="compound.yaml")
    decision_prepare = subcommands.add_parser("prepare-ds1000-gepa-decision")
    decision_prepare.add_argument("--config", default="compound.yaml")
    decision_prepare.add_argument(
        "--output", default="benchmarks/manifests/ds1000_gepa_decision25.json"
    )
    decision_prepare.add_argument("--count", type=int, default=25)
    decision_prepare.add_argument(
        "--exclude-manifest", action="append", default=[], dest="exclude_manifests"
    )
    open_decision = subcommands.add_parser("open-ds1000-decision-for-diagnostics")
    open_decision.add_argument("--config", default="compound.yaml")
    open_decision.add_argument("--decision-manifest", required=True)
    open_decision.add_argument("--decision-report", required=True)
    open_decision.add_argument("--output", required=True)
    v2_run = subcommands.add_parser("run-ds1000-gepa-v2")
    v2_run.add_argument("--config", default="compound.yaml")
    v2_run.add_argument("--manifest", default="benchmarks/manifests/ds1000_gepa_v2.json")
    v2_run.add_argument("--max-metric-calls", type=int, default=120)
    v2_run.add_argument("--max-tokens", type=int, default=4096)
    v2_run.add_argument("--reasoning-effort", default=None)
    v2_run.add_argument("--validation-trials", type=int, default=2)
    v2_run.add_argument("--decision-trials", type=int, default=2)
    v2_run.add_argument("--experiment-cap-usd", type=float, default=4.0)
    v2_run.add_argument("--output-dir", default=None)
    v2_run.add_argument(
        "--candidate-model",
        default="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    )
    v2_run.add_argument(
        "--candidate-backend",
        choices=["chat_completions", "flex"],
        default="chat_completions",
    )
    v2_run.add_argument(
        "--component-policy",
        choices=["adaptive", "library_only"],
        default="adaptive",
    )
    v2_run.add_argument("--teacher-artifact", default=None)
    v2_run.add_argument(
        "--teacher-model", action="append", default=[], dest="teacher_models"
    )
    v2_probe = subcommands.add_parser("probe-ds1000-gepa-v2")
    v2_probe.add_argument("--config", default="compound.yaml")
    v2_probe.add_argument("--manifest", default="benchmarks/manifests/ds1000_gepa_v2.json")
    v2_probe.add_argument("--max-tokens", type=int, default=4096)
    v2_probe.add_argument("--reasoning-effort", default="low")
    v2_probe.add_argument("--experiment-cap-usd", type=float, default=0.50)
    v2_probe.add_argument(
        "--candidate-model",
        default="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    )
    v2_probe.add_argument(
        "--candidate-backend",
        choices=["chat_completions", "flex"],
        default="chat_completions",
    )
    v2_decision = subcommands.add_parser("evaluate-ds1000-gepa-v2-decision")
    v2_decision.add_argument("run_dir")
    v2_decision.add_argument("--config", default="compound.yaml")
    v2_decision.add_argument("--decision-manifest", default=None)
    v2_decision.add_argument(
        "--reference-model", action="append", default=[], dest="reference_models"
    )
    v2_decision.add_argument("--reference-trials", type=int, default=None)
    v2_decision.add_argument("--experiment-cap-usd", type=float, default=4.0)
    v2_decision.add_argument("--overwrite-cached", action="store_true")
    decision = subcommands.add_parser("evaluate-ds1000-decision")
    decision.add_argument("run_dir")
    decision.add_argument("--config", default="compound.yaml")
    telemetry = subcommands.add_parser("telemetry-report")
    telemetry.add_argument("--config", default="compound.yaml")
    telemetry.add_argument("--input", default=None)
    telemetry.add_argument("--output", default=None)
    migrate = subcommands.add_parser("migrate-ds1000-cache")
    migrate.add_argument("--cache-dir", default="artifacts/cache/ds1000")
    migrate.add_argument("--runs-dir", default="artifacts/optimization")
    migrate.add_argument("--reflection-cache-dir", default="artifacts/cache/reflection")
    migrate.add_argument("--legacy-evaluator-image", default="compound-ds1000-numpy:20260717")
    migrate.add_argument("--archive", action="store_true")
    migrate.add_argument("--config", default="compound.yaml")
    regrade = subcommands.add_parser("regrade-ds1000-run")
    regrade.add_argument("run_dir")
    regrade.add_argument("--cache-dir", default="artifacts/cache/ds1000")
    regrade.add_argument("--source", default=".compound/sources/ds1000/data/ds1000.jsonl.gz")
    regrade.add_argument("--evaluator-image", default="compound-ds1000-numpy:20260717-v2")
    regrade.add_argument("--output", default=None)
    regrade.add_argument("--case-id", action="append", default=None)
    regrade.add_argument("--config", default="compound.yaml")
    flex_smoke = subcommands.add_parser("run-doubleword-flex-smoke")
    flex_smoke.add_argument("--config", default="compound.yaml")
    flex_smoke.add_argument(
        "--source", default=".compound/sources/ds1000/data/ds1000.jsonl.gz"
    )
    flex_smoke.add_argument("--case-id", default="ds1000_428")
    flex_smoke.add_argument("--model", action="append", default=None, dest="models")
    flex_smoke.add_argument("--max-output-tokens", type=int, default=4096)
    flex_smoke.add_argument("--poll-interval-seconds", type=float, default=2.0)
    flex_smoke.add_argument("--timeout-seconds", type=float, default=3600.0)
    flex_smoke.add_argument(
        "--evaluator-image", default="compound-ds1000-numpy:20260720-v3"
    )
    flex_smoke.add_argument("--output", default=None)
    baseline_prepare = subcommands.add_parser("prepare-ds1000-baseline-matrix")
    baseline_prepare.add_argument("--config", default="compound.yaml")
    baseline_prepare.add_argument(
        "--output", default="benchmarks/manifests/ds1000_baseline25.json"
    )
    baseline_prepare.add_argument("--count", type=int, default=25)
    baseline_prepare.add_argument(
        "--exclude-manifest", action="append", default=[], dest="exclude_manifests"
    )
    baseline_run = subcommands.add_parser("run-ds1000-baseline-matrix")
    baseline_run.add_argument("--config", default="compound.yaml")
    baseline_run.add_argument(
        "--manifest", default="benchmarks/manifests/ds1000_baseline25.json"
    )
    baseline_run.add_argument("--model", action="append", default=None, dest="models")
    baseline_run.add_argument("--case-id", action="append", default=None, dest="case_ids")
    baseline_run.add_argument("--max-output-tokens", type=int, default=4096)
    baseline_run.add_argument("--reasoning-effort", default=None)
    baseline_run.add_argument("--trials-per-case", type=int, default=1)
    baseline_run.add_argument("--experiment-cap-usd", type=float, default=8.0)
    baseline_run.add_argument("--poll-interval-seconds", type=float, default=2.0)
    baseline_run.add_argument("--timeout-seconds", type=float, default=3600.0)
    baseline_run.add_argument(
        "--evaluator-image", default="compound-ds1000-numpy:20260720-v3"
    )
    baseline_run.add_argument("--output", default=None)
    bfcl_run = subcommands.add_parser("run-bfcl-baseline-matrix")
    bfcl_run.add_argument("--config", default="compound.yaml")
    bfcl_run.add_argument("--manifest", default="benchmarks/manifests/bfcl.json")
    bfcl_run.add_argument("--source-dir", default=".compound/sources/gorilla")
    bfcl_run.add_argument("--model", action="append", default=None, dest="models")
    bfcl_run.add_argument("--case-id", action="append", default=None, dest="case_ids")
    bfcl_run.add_argument("--max-output-tokens", type=int, default=4096)
    bfcl_run.add_argument("--reasoning-effort", default=None)
    bfcl_run.add_argument("--trials-per-case", type=int, default=1)
    bfcl_run.add_argument("--experiment-cap-usd", type=float, default=4.0)
    bfcl_run.add_argument("--poll-interval-seconds", type=float, default=2.0)
    bfcl_run.add_argument("--timeout-seconds", type=float, default=3600.0)
    bfcl_run.add_argument("--output", default=None)
    return parser


def main() -> None:
    load_env()
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "validate-config":
        print("compound.yaml is valid")
        return
    if args.command == "prepare-manifests":
        written = prepare_manifests(
            config,
            sources_dir=args.sources_dir,
            output_dir=args.output_dir or config["manifests_dir"],
        )
        for benchmark, path in written.items():
            print(f"{benchmark}: {path}")
        return
    if args.command == "run-ds1000-proof":
        path = run_ds1000_proof(
            config_path=args.config,
            max_metric_calls=args.max_metric_calls,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            candidate_max_tokens=args.candidate_max_tokens,
            reflection_max_tokens=args.reflection_max_tokens,
        )
        print(path)
        return
    if args.command == "prepare-ds1000-scale":
        excluded: set[str] = set()
        for run_dir in args.exclude_run:
            run_path = Path(run_dir)
            run_manifest = json.loads((run_path / "manifest.json").read_text())
            excluded.update(run_manifest["train_case_ids"])
            excluded.update(run_manifest["validation_case_ids"])
            decision_path = run_path / "decision-test.json"
            if decision_path.exists():
                excluded.update(json.loads(decision_path.read_text())["case_ids"])
        path = prepare_scaled_ds1000_manifest(
            source_dir=".compound/sources/ds1000",
            output_path=args.output,
            revision=config["benchmarks"]["ds1000"]["revision"],
            seed=int(config["seed"]) + 1,
            count=args.count,
            train=args.train,
            validation=args.validation,
            decision=args.decision,
            excluded_case_ids=excluded,
        )
        print(path)
        return
    if args.command == "prepare-ds1000-gepa-v2":
        from compound.catalogs import load_ds1000_catalog

        catalog = {case.case_id: case for case in load_ds1000_catalog(".compound/sources/ds1000")}
        manifests = args.exclude_manifests or sorted(
            str(path)
            for path in Path("benchmarks/manifests").glob("ds1000*.json")
            if path.resolve() != Path(args.output).resolve()
        )
        excluded_case_ids: set[str] = set()
        excluded_origin_groups: set[str] = set()
        for manifest_path in manifests:
            payload = json.loads(Path(manifest_path).read_text())
            for record in payload["cases"]:
                case_id = record["case_id"]
                excluded_case_ids.add(case_id)
                excluded_origin_groups.add(ds1000_origin_group(catalog[case_id]))
        path = prepare_gepa_v2_ds1000_manifest(
            source_dir=".compound/sources/ds1000",
            output_path=args.output,
            revision=config["benchmarks"]["ds1000"]["revision"],
            seed=int(config["seed"]) + 2,
            train=args.train,
            validation=args.validation,
            decision=args.decision,
            excluded_case_ids=excluded_case_ids,
            excluded_origin_groups=excluded_origin_groups,
        )
        print(path)
        return
    if args.command == "run-ds1000-gepa-v2":
        path = run_gepa_v2(
            manifest_path=args.manifest,
            config_path=args.config,
            max_metric_calls=args.max_metric_calls,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            validation_trials=args.validation_trials,
            decision_trials=args.decision_trials,
            experiment_cap_usd=args.experiment_cap_usd,
            output_dir=args.output_dir,
            candidate_model=args.candidate_model,
            candidate_backend=args.candidate_backend,
            component_policy=args.component_policy,
            teacher_artifact_path=args.teacher_artifact,
            teacher_models=args.teacher_models,
        )
        print(path)
        return
    if args.command == "prepare-ds1000-gepa-opened":
        path = prepare_gepa_manifest_from_opened_baseline(
            baseline_manifest_path=args.baseline_manifest,
            output_path=args.output,
            train=args.train,
            validation=args.validation,
        )
        print(path)
        return
    if args.command == "prepare-ds1000-gepa-decision":
        from compound.catalogs import load_ds1000_catalog

        catalog = {
            case.case_id: case for case in load_ds1000_catalog(".compound/sources/ds1000")
        }
        manifests = args.exclude_manifests or sorted(
            str(path)
            for path in Path("benchmarks/manifests").glob("ds1000*.json")
            if path.resolve() != Path(args.output).resolve()
        )
        excluded_case_ids: set[str] = set()
        excluded_origin_groups: set[str] = set()
        for manifest_path in manifests:
            payload = json.loads(Path(manifest_path).read_text())
            for record in payload["cases"]:
                case_id = record["case_id"]
                excluded_case_ids.add(case_id)
                excluded_origin_groups.add(ds1000_origin_group(catalog[case_id]))
        path = prepare_ds1000_gepa_decision_manifest(
            source_dir=".compound/sources/ds1000",
            output_path=args.output,
            revision=config["benchmarks"]["ds1000"]["revision"],
            seed=int(config["seed"]) + 4,
            count=args.count,
            excluded_case_ids=excluded_case_ids,
            excluded_origin_groups=excluded_origin_groups,
        )
        print(path)
        return
    if args.command == "open-ds1000-decision-for-diagnostics":
        path = open_ds1000_decision_manifest_for_diagnostics(
            decision_manifest_path=args.decision_manifest,
            decision_report_path=args.decision_report,
            output_path=args.output,
        )
        print(path)
        return
    if args.command == "probe-ds1000-gepa-v2":
        path = probe_gepa_v2(
            manifest_path=args.manifest,
            config_path=args.config,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            experiment_cap_usd=args.experiment_cap_usd,
            candidate_model=args.candidate_model,
            candidate_backend=args.candidate_backend,
        )
        print(path)
        return
    if args.command == "evaluate-ds1000-gepa-v2-decision":
        path = evaluate_gepa_v2_decision(
            args.run_dir,
            config_path=args.config,
            decision_manifest_path=args.decision_manifest,
            reference_models=args.reference_models,
            reference_trials=args.reference_trials,
            experiment_cap_usd=args.experiment_cap_usd,
            overwrite_cached=args.overwrite_cached,
        )
        print(path)
        return
    if args.command == "evaluate-ds1000-decision":
        path = evaluate_ds1000_decision_test(args.run_dir, config_path=args.config)
        print(path)
        return
    if args.command == "telemetry-report":
        source = args.input or f"{config['artifacts_dir']}/telemetry/model_calls.jsonl"
        report = summarize_model_calls(source)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(rendered)
        else:
            print(rendered, end="")
        return
    if args.command == "migrate-ds1000-cache":
        result = migrate_legacy_ds1000_cache(
            cache_dir=args.cache_dir,
            runs_dir=args.runs_dir,
            reflection_cache_dir=args.reflection_cache_dir,
            legacy_evaluator_image=args.legacy_evaluator_image,
            archive=args.archive,
        )
        print(json.dumps(asdict(result), indent=2, sort_keys=True))
        return
    if args.command == "regrade-ds1000-run":
        path = regrade_ds1000_run(
            run_dir=args.run_dir,
            cache_dir=args.cache_dir,
            source_path=args.source,
            evaluator_image=args.evaluator_image,
            output_path=args.output,
            case_ids=set(args.case_id) if args.case_id else None,
        )
        print(path)
        return
    if args.command == "run-doubleword-flex-smoke":
        path = run_doubleword_flex_smoke(
            config_path=args.config,
            source_path=args.source,
            case_id=args.case_id,
            models=args.models,
            max_output_tokens=args.max_output_tokens,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
            evaluator_image=args.evaluator_image,
            output_path=args.output,
        )
        print(path)
        return
    if args.command == "prepare-ds1000-baseline-matrix":
        from compound.catalogs import load_ds1000_catalog

        catalog = {
            case.case_id: case for case in load_ds1000_catalog(".compound/sources/ds1000")
        }
        manifests = args.exclude_manifests or sorted(
            str(path)
            for path in Path("benchmarks/manifests").glob("ds1000*.json")
            if path.resolve() != Path(args.output).resolve()
        )
        excluded_case_ids: set[str] = set()
        excluded_origin_groups: set[str] = set()
        for manifest_path in manifests:
            payload = json.loads(Path(manifest_path).read_text())
            for record in payload["cases"]:
                case_id = record["case_id"]
                excluded_case_ids.add(case_id)
                excluded_origin_groups.add(ds1000_origin_group(catalog[case_id]))
        path = prepare_ds1000_baseline_matrix_manifest(
            source_dir=".compound/sources/ds1000",
            output_path=args.output,
            revision=config["benchmarks"]["ds1000"]["revision"],
            seed=int(config["seed"]) + 3,
            count=args.count,
            excluded_case_ids=excluded_case_ids,
            excluded_origin_groups=excluded_origin_groups,
        )
        print(path)
        return
    if args.command == "run-bfcl-baseline-matrix":
        path = run_bfcl_baseline_matrix(
            manifest_path=args.manifest,
            config_path=args.config,
            source_dir=args.source_dir,
            models=args.models,
            case_ids=args.case_ids,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            trials_per_case=args.trials_per_case,
            experiment_cap_usd=args.experiment_cap_usd,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
            output_path=args.output,
        )
        print(path)
        return
    if args.command == "run-ds1000-baseline-matrix":
        path = run_ds1000_baseline_matrix(
            manifest_path=args.manifest,
            config_path=args.config,
            models=args.models,
            case_ids=args.case_ids,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            trials_per_case=args.trials_per_case,
            experiment_cap_usd=args.experiment_cap_usd,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.timeout_seconds,
            evaluator_image=args.evaluator_image,
            output_path=args.output,
        )
        print(path)
        return
    summary = {
        name: {
            "task_key": value["task_key"],
            "sample_count": value["sample_count"],
            "partitions": value["partitions"],
        }
        for name, value in config["benchmarks"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
