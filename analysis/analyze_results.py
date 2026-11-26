import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from io import BytesIO
from itertools import product
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from visual_rules.dataset import CORE_CONDITIONS
from visual_rules.models import MODEL_IDS
from visual_rules.rules import parse_rule, parse_values

MODELS = ("qwen", "gemma", "centurio", "smolvlm")
LANGUAGES = ("en", "bn")
METHODS = ("direct", "execute", "constrained")
REPLAYS = ("observed", "verified_counts", "verified_rule", "both_verified")
EXPECTED_PAIRS = {"rord": 173, "heldout": 24, "scenediff": 5}
ARCHIVE_PREFIXES = {
    "rord-main-v1": "rord",
    "visual-rule-photo-heldout": "heldout",
    "scenediff-replication-v2": "scenediff",
}
DEFAULT_TABLES = ROOT / "results" / "paper_analysis" / "tables"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dataset_name(archive_name):
    stem = Path(archive_name).name
    return next((dataset for prefix, dataset in ARCHIVE_PREFIXES.items()
                 if any(stem == f"{prefix}-{model}-results.zip" for model in MODELS)), None)


def find_runs(directory):
    for path in sorted(directory.rglob("*.zip")):
        with ZipFile(path) as archive:
            dataset = dataset_name(path.name)
            if dataset:
                yield dataset, path.relative_to(directory).as_posix(), archive
                continue
            for member in archive.namelist():
                dataset = dataset_name(member)
                if dataset:
                    with ZipFile(BytesIO(archive.read(member))) as inner:
                        yield dataset, f"{path.relative_to(directory).as_posix()}!{member}", inner


def read_member(archive, suffix):
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one file ending in {suffix}, found {len(matches)}")
    return archive.read(matches[0])


def read_metadata(archive):
    if any(name.endswith("/configuration.json") for name in archive.namelist()):
        configuration = json.loads(read_member(archive, "/configuration.json"))
        summary = json.loads(read_member(archive, "/compute_summary.json"))
        return configuration, summary

    attempts = [json.loads(archive.read(name)) for name in archive.namelist()
                if "/runs/" in name and name.endswith("/attempt.json")]
    settings_paths = [name for name in archive.namelist()
                      if "/runs/" in name and name.endswith("/settings.json")]
    if len(settings_paths) != 1 or not attempts:
        raise ValueError("Missing legacy run settings or attempt logs")
    settings = json.loads(archive.read(settings_paths[0]))
    model = settings["backend"]
    constrained = [json.loads(archive.read(name)) for name in archive.namelist()
                   if f"/constrained/{model}/compute/" in name and name.endswith(".json")]
    logs = attempts + constrained
    summary = {
        "baseline_evaluations": max(attempt["runner_summary"]["completed"] for attempt in attempts),
        "constrained_prompts": sum(attempt.get("successful_text_only_calls", 0) for attempt in constrained),
        "timed_wall_seconds": sum(attempt["command_wall_seconds"] for attempt in logs),
        "gpu_hours": sum(attempt["gpu_hours"] for attempt in logs),
    }
    return {
        **settings, "model_id": MODEL_IDS[model], "model_revision": settings["revision"],
        "git_commit": attempts[0]["git_commit"],
        "precision": attempts[0]["runner_summary"]["model_precision"],
    }, summary


def read_run(dataset, source, archive):
    manifest_raw = read_member(archive, "/data/manifest.json")
    manifest = json.loads(manifest_raw)
    fingerprint = sha256(manifest_raw).hexdigest()
    example_bytes = read_member(archive, "/data/examples.jsonl")
    if sha256(example_bytes).hexdigest() != manifest["examples_sha256"]:
        raise ValueError(f"Example checksum mismatch in {source}")
    example_rows = list(map(json.loads, example_bytes.splitlines()))
    examples = {row["id"]: row for row in example_rows}
    groups = {row["group"] for row in example_rows}
    expected_pairs = EXPECTED_PAIRS[dataset]
    keys = {(row["group"], row["language"], row["condition"]) for row in example_rows}
    if (
        len(groups) != expected_pairs
        or manifest["groups"] != expected_pairs
        or len(manifest["images"]) != 2 * expected_pairs
        or len(examples) != manifest["examples"]
        or len(example_rows) != len(keys)
        or keys != set(product(groups, LANGUAGES, CORE_CONDITIONS))
    ):
        raise ValueError(f"Incomplete or mismatched dataset in {source}")

    config, compute = read_metadata(archive)
    model = config["backend"]
    if model not in MODELS or config["dataset_sha256"] != fingerprint:
        raise ValueError(f"Run configuration mismatch in {source}")
    prediction_bytes = read_member(archive, "/scored_predictions.jsonl")
    predictions = list(map(json.loads, prediction_bytes.splitlines()))
    expected = set(product(examples, METHODS))
    actual = {(row["id"], row["method"]) for row in predictions}
    if actual != expected or len(predictions) != len(expected):
        raise ValueError(f"Incomplete or duplicate predictions in {source}")

    clusters = {}
    for row in predictions:
        example = examples[row["id"]]
        for key in ("group", "language", "condition"):
            if row[key] != example[key]:
                raise ValueError(f"Prediction {key} mismatch for {row['id']} in {source}")
        if row["gold"] is not example["answer"]:
            raise ValueError(f"Gold label mismatch for {row['id']} in {source}")
        if row["answer"] is not None and type(row["answer"]) is not bool:
            raise ValueError(f"Invalid prediction type for {row['id']} in {source}")
        if clusters.setdefault(row["group"], row["cluster"]) != row["cluster"]:
            raise ValueError(f"Inconsistent capture environment for {row['group']}")

    return {
        "dataset": dataset, "model": model, "source": source, "fingerprint": fingerprint,
        "predictions_sha256": sha256(prediction_bytes).hexdigest(), "examples": examples,
        "predictions": predictions, "clusters": clusters, "config": config, "compute": compute,
    }


def mean(values):
    return float(np.mean(values))


def confidence_interval(values, rng, repeats):
    array = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(array), size=(repeats, len(array)))
    low, high = np.quantile(array[indices].mean(axis=1), (0.025, 0.975))
    return float(low), float(high)


def capture_means(pair_scores, clusters):
    grouped = defaultdict(list)
    for group, score in sorted(pair_scores.items()):
        grouped[clusters[group]].append(score)
    return [mean(values) for values in grouped.values()]


def calculate_scores(predictions, clusters):
    grouped = defaultdict(list)
    for row in predictions:
        grouped[row["group"]].append(row["answer"] is row["gold"])
    pairs = {group: all(answers) for group, answers in grouped.items()}
    return {
        "examples": len(predictions),
        "pairs": len(pairs),
        "capture_environments": len(set(clusters.values())),
        "accuracy": mean([row["answer"] is row["gold"] for row in predictions]),
        "all_four_accuracy": mean(list(pairs.values())),
        "capture_weighted_all_four": mean(capture_means(pairs, clusters)),
    }, pairs


def replay_execution(predictions, examples):
    for row in predictions:
        example = examples[row["id"]]
        raw = row["raw"]
        estimated_rule = parse_rule(raw["rule"])
        estimated_counts = parse_values(raw["values"])
        verified_rule = parse_rule(example["rule_code"])
        verified_counts = tuple(example["values"])

        observed = estimated_rule.applies(estimated_counts) if estimated_rule and estimated_counts else None
        if observed is not row["answer"] or verified_rule is None:
            raise ValueError(f"Could not reproduce execution for {row['id']}")
        scores = (
            observed,
            estimated_rule.applies(verified_counts) if estimated_rule else None,
            verified_rule.applies(estimated_counts) if estimated_counts else None,
            verified_rule.applies(verified_counts),
        )
        if scores[-1] is not row["gold"]:
            raise ValueError(f"Verified rule and counts disagree with the gold label for {row['id']}")
        yield row, scores, estimated_counts is not None


def summarize_run(run, rng, repeats):
    tables = defaultdict(list)
    clusters = run["clusters"]
    predictions = run["predictions"]
    for language in LANGUAGES:
        identity = {"dataset": run["dataset"], "model": run["model"], "language": language}
        scores_by_method = {}
        count_outputs = {}
        for method in METHODS:
            selected = [row for row in predictions if row["language"] == language and row["method"] == method]
            case_metrics, pair_scores = calculate_scores(selected, clusters)
            low, high = confidence_interval(capture_means(pair_scores, clusters), rng, repeats)
            tables["main_results.csv"].append({
                **identity, "method": method, **case_metrics,
                "capture_weighted_ci_low": low, "capture_weighted_ci_high": high,
            })
            scores_by_method[method] = pair_scores
            if method == "direct":
                continue

            replay_rows = list(replay_execution(selected, run["examples"]))
            count_outputs[method] = {row["id"]: row["raw"]["values"] for row in selected}
            tables["diagnostics.csv"].append({
                **identity, "method": method, "examples": len(selected),
                "rule_parseable": mean([row["rule_parse_valid"] for row in selected]),
                "rule_exact": mean([row["rule_exact"] for row in selected]),
                "counts_parseable": mean([parsed for _, _, parsed in replay_rows]),
                "counts_exact": mean([row["value_exact"] for row in selected]),
            })
            for index, variant in enumerate(REPLAYS):
                updated = [{**row, "answer": answers[index]} for row, answers, _ in replay_rows]
                replay_metrics, _ = calculate_scores(updated, clusters)
                tables["component_replay.csv"].append({
                    **identity, "method": method, "substitution": variant, **replay_metrics,
                })
        if count_outputs["execute"] != count_outputs["constrained"]:
            raise ValueError(f"Execution methods used different visual counts for {identity}")
        for baseline, method in (("direct", "execute"), ("direct", "constrained"), ("execute", "constrained")):
            first, second = scores_by_method[baseline], scores_by_method[method]
            paired = {group: int(second[group]) - int(first[group]) for group in first}
            clustered = capture_means(paired, clusters)
            low, high = confidence_interval(clustered, rng, repeats)
            tables["method_differences.csv"].append({
                **identity, "baseline": baseline, "method": method,
                "pairs": len(paired), "capture_environments": len(clustered),
                "pair_weighted_difference": mean(list(paired.values())),
                "capture_weighted_difference": mean(clustered),
                "ci_low": low, "ci_high": high,
            })
    config = run["config"]
    compute = run["compute"]
    tables["compute.csv"].append({
        "dataset": run["dataset"], "model": run["model"], "model_id": config["model_id"],
        "model_revision": config["model_revision"], "code_commit": config["git_commit"],
        "precision": config["precision"], "dataset_sha256": run["fingerprint"],
        "baseline_evaluations": compute.get("baseline_evaluations"),
        "constrained_prompts": compute.get("constrained_prompts"),
        "timed_wall_seconds": compute.get("timed_wall_seconds"), "gpu_hours": compute.get("gpu_hours"),
        "peak_allocated_gib": compute.get("peak_allocated_gib"),
        "peak_reserved_gib": compute.get("peak_reserved_gib"), "usd_cost": compute.get("usd_cost"),
    })
    return tables


def write_table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    if args.bootstrap_repeats < 1:
        raise ValueError("Use at least one bootstrap repeat")
    runs = {}
    cohorts = {}
    for dataset, source, archive in find_runs(args.results):
        run = read_run(dataset, source, archive)
        key = dataset, run["model"]
        previous = runs.get(key)
        if previous:
            if any(previous[name] != run[name] for name in (
                "fingerprint", "predictions_sha256", "clusters", "config", "compute",
            )):
                raise ValueError(f"Conflicting archives for {key}: {previous['source']} and {source}")
            continue
        cohort = run["fingerprint"], run["clusters"]
        if cohorts.setdefault(dataset, cohort) != cohort:
            raise ValueError(f"Models for {dataset} use different datasets or capture environments")
        runs[key] = run
    missing = [model for model in MODELS if ("rord", model) not in runs]
    if missing:
        raise ValueError(f"Missing RORD results for {', '.join(missing)} under {args.results}")

    rng = np.random.default_rng(args.seed)
    tables = defaultdict(list)
    for key in sorted(runs):
        for name, rows in summarize_run(runs[key], rng, args.bootstrap_repeats).items():
            tables[name].extend(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_table(args.output / name, rows)
    manifest = {
        "seed": args.seed, "bootstrap_repeats": args.bootstrap_repeats,
        "confidence_level": 0.95, "resampling_unit": "capture environment",
        "runs": [{"dataset": key[0], "model": key[1], "source": runs[key]["source"],
                  "dataset_sha256": runs[key]["fingerprint"],
                  "predictions_sha256": runs[key]["predictions_sha256"]} for key in sorted(runs)],
    }
    (args.output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(tables)} tables for {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
