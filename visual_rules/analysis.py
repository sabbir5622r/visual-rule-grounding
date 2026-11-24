from collections import defaultdict
import csv
import json
import random
import sqlite3
from statistics import fmean as mean

from .dataset import CONDITIONS, CORE_CONDITIONS


def load_predictions(path, allow_partial=False):
    database = path / "run.sqlite"
    if not database.is_file():
        raise ValueError(f"No run database at {database}")
    with sqlite3.connect(database) as connection:
        settings = json.loads(connection.execute("SELECT json FROM settings WHERE id=1").fetchone()[0])
        requested = [row[0] for row in connection.execute("SELECT name FROM requested_method")]
        predictions = [
            json.loads(row[0]) for row in connection.execute(
                "SELECT json FROM prediction ORDER BY method, example"
            )
        ]
    counts = defaultdict(int)
    for row in predictions:
        counts[row["method"]] += 1
    if not predictions:
        raise ValueError(f"No predictions at {path}")
    if not allow_partial and any(counts[method] != settings["examples"] for method in requested):
        raise ValueError(f"Incomplete method at {path}. Resume it or use --allow-partial for inspection.")
    if not allow_partial and settings["backend"] == "oracle":
        raise ValueError("Oracle runs are workflow checks. Use --allow-partial to inspect their output.")
    return settings, predictions


def is_correct(row):
    return row["answer"] is not None and row["answer"] == row["gold"]


def summarize_groups(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["group"]][row["condition"]] = row
    metrics = defaultdict(dict)
    for group, conditions in grouped.items():
        metrics["accuracy"][group] = mean(is_correct(row) for row in conditions.values())
        metrics["valid_answer_rate"][group] = mean(row["answer"] is not None for row in conditions.values())
        for condition, row in conditions.items():
            metrics[f"accuracy_{condition}"][group] = is_correct(row)
        for key in ("value_exact", "rule_exact", "value_parse_valid", "rule_parse_valid"):
            available = [row[key] for row in conditions.values() if key in row]
            if available:
                metrics[key][group] = mean(available)
        if all(condition in conditions for condition in CORE_CONDITIONS):
            quartet = [conditions[condition] for condition in CORE_CONDITIONS]
            metrics["factorial_group_accuracy"][group] = all(map(is_correct, quartet))
            if all(row["answer"] is not None for row in quartet):
                gold = [int(row["gold"]) for row in quartet]
                output = [int(row["answer"]) for row in quartet]
                gold_interaction = gold[3] - gold[2] - gold[1] + gold[0]
                predicted_interaction = output[3] - output[2] - output[1] + output[0]
                orientation = 1 if gold[0] else -1
                metrics["signed_interaction_error_among_valid"][group] = (
                    orientation * (predicted_interaction - gold_interaction)
                )
                metrics["interaction_exact_among_valid"][group] = predicted_interaction == gold_interaction
            metrics["valid_factorial_group_rate"][group] = all(row["answer"] is not None for row in quartet)
        if all(condition in conditions for condition in CONDITIONS):
            metrics["all_conditions_accuracy"][group] = all(map(is_correct, conditions.values()))
        original = conditions.get("original")
        if original is None:
            continue
        for condition, name in (
            ("image_edit", "image_flip"), ("rule_edit", "rule_flip"), ("placebo", "placebo_invariance"),
            ("rule_paraphrase", "paraphrase_invariance"),
        ):
            comparison = conditions.get(condition)
            if comparison is None:
                continue
            metrics[f"{name}_correct_pair"][group] = is_correct(original) and is_correct(comparison)
            if is_correct(original):
                metrics[f"{name}_given_correct_original"][group] = is_correct(comparison)
        exact_components = [
            row for row in conditions.values() if row.get("value_exact") and row.get("rule_exact")
        ]
        if exact_components:
            metrics["execution_accuracy_given_exact_components"][group] = mean(map(is_correct, exact_components))
    return metrics


def interval(values, draws, seed):
    if not values:
        return None, None
    ordered = [values[group] for group in sorted(values)]
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(ordered, k=len(ordered))) for _ in range(draws))
    return samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]


def stat_row(dataset, backend, language, family, metric, comparison, values, draws, seed):
    low, high = interval(values, draws, seed)
    return {
        "dataset": dataset,
        "backend": backend,
        "language": language,
        "family": family,
        "metric": metric,
        "method_or_contrast": comparison,
        "groups": len(values),
        "estimate": mean(values.values()),
        "ci_low": low,
        "ci_high": high,
    }


def build_report(paths, directory, draws, seed, allow_partial=False):
    combined = defaultdict(list)
    run_settings = {}
    for path in paths:
        settings, predictions = load_predictions(path, allow_partial)
        dataset = f"{settings['split']}:{settings['dataset_sha256'][:12]}"
        backend = settings["backend"]
        processing = (settings["revision"], settings["visual_tokens"], settings["implementation"])
        previous = run_settings.setdefault((dataset, backend), processing)
        if previous != processing:
            raise ValueError(f"Conflicting model revision, image settings or code for {dataset}/{backend}.")
        for row in predictions:
            combined[(dataset, backend, row["language"], row["method"], "all")].append(row)
            combined[(dataset, backend, row["language"], row["method"], row["family"])].append(row)

    reported = []
    metrics_by_run = {}
    for (dataset, backend, language, method, family), rows in sorted(combined.items()):
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate predictions for {dataset}/{backend}/{language}/{method}.")
        metrics = summarize_groups(rows)
        metrics_by_run[dataset, backend, language, family, method] = metrics
        for metric, values in sorted(metrics.items()):
            reported.append(stat_row(dataset, backend, language, family, metric, method, values, draws, seed))

    for key, metrics in sorted(metrics_by_run.items()):
        dataset, backend, language, family, method = key
        if method != "direct":
            reference = metrics_by_run.get((dataset, backend, language, family, "direct"), {})
            reported.extend(contrast(dataset, backend, language, family, f"{method} minus direct",
                                     metrics, reference, draws, seed))
        if language == "bn":
            reference = metrics_by_run.get((dataset, backend, "en", family, method), {})
            reported.extend(contrast(dataset, backend, "bn-en", family, method, metrics, reference, draws, seed))

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(reported[0]))
        writer.writeheader()
        writer.writerows(reported)
    (directory / "metrics.json").write_text(json.dumps(reported, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(reported)} estimates to {directory / 'metrics.csv'}", flush=True)


def contrast(dataset, backend, language, family, name, first, second, draws, seed):
    metrics = ("accuracy", "factorial_group_accuracy", "rule_flip_correct_pair",
               "image_flip_correct_pair", "placebo_invariance_correct_pair", "paraphrase_invariance_correct_pair")
    for metric in metrics:
        reference = second.get(metric, {})
        values = {group: value - reference[group] for group, value in first.get(metric, {}).items()
                  if group in reference}
        if values:
            yield stat_row(dataset, backend, language, family, metric, name, values, draws, seed)
