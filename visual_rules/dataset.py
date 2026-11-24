from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import NamedTuple

from .render import draw_chart
from .rules import Clause, Rule, parse_rule


FAMILIES = ("cutoff", "range", "gap", "conjunction")
CONDITIONS = ("original", "image_edit", "rule_edit", "both_edits", "placebo", "rule_paraphrase")
CORE_CONDITIONS = CONDITIONS[:4]
LANGUAGES = ("en", "bn")
SCHEMA = 2


class Scenario(NamedTuple):
    base_values: tuple[int, int]
    edited_values: tuple[int, int]
    base_rule: Rule
    edited_rule: Rule


def random_values(rng):
    return rng.randint(15, 85), rng.randint(15, 85)


def sample_rule(rng, family, feature):
    direction = rng.choice(("GE", "LE"))
    if family == "range":
        low, high = sorted(rng.sample(range(12, 89), 2))
        return Rule((Clause(feature, "GE", low), Clause(feature, "LE", high)))
    if family == "conjunction":
        clauses = tuple(
            Clause(axis, rng.choice(("GE", "LE")), rng.randint(14, 86))
            for axis in ("A", "B")
        )
        return Rule(clauses)
    if family == "gap":
        return Rule((Clause("GAP", direction, rng.randint(-50, 50)),))
    return Rule((Clause(feature, direction, rng.randint(14, 86)),))


def sample_scenario(rng, family, original_answer, double_edit_returns):
    feature = rng.choice(("A", "B"))
    edited_index = ("A", "B").index(feature)
    expected_double = original_answer if double_edit_returns else not original_answer
    for _ in range(150):
        base = random_values(rng)
        edited = list(base)
        edited[edited_index] = rng.randint(15, 85)
        edited = tuple(edited)
        if abs(base[edited_index] - edited[edited_index]) < 12:
            continue
        original = None
        alternate = None
        for _ in range(160):
            rule = sample_rule(rng, family, feature)
            pattern = (rule.applies(base), rule.applies(edited))
            if pattern == (original_answer, not original_answer):
                original = rule
            if pattern == (not original_answer, expected_double):
                alternate = rule
            if original and alternate:
                return Scenario(base, edited, original, alternate)
    raise RuntimeError(f"Could not construct a {family} scenario")


def make_dataset(directory, groups, seed, split):
    if groups < len(FAMILIES):
        raise ValueError("Use at least four scenarios to cover every rule family.")
    directory = Path(directory)
    if (directory / "examples.jsonl").exists() or (directory / "images").exists():
        raise ValueError(f"Dataset output already exists in {directory}. Use a fresh path.")
    image_directory = directory / "images"
    image_directory.mkdir(parents=True)
    rng = random.Random(f"visual-rules-v1:{seed}:{split}")
    cells = [
        (family, layout, answer, returns)
        for layout in ("bars", "table")
        for answer in (False, True)
        for returns in (False, True)
        for family in FAMILIES
    ]
    rng.shuffle(cells)
    seen_scenarios = set()
    with (directory / "examples.jsonl").open("x", encoding="utf-8") as output:
        for index in range(groups):
            family, layout, answer, returns = cells[index % len(cells)]
            if groups < len(cells):
                family = FAMILIES[index % len(FAMILIES)]
            group = f"{split}-{index + 1:05d}"
            scenario = sample_scenario(rng, family, answer, returns)
            while scenario in seen_scenarios:
                scenario = sample_scenario(rng, family, answer, returns)
            seen_scenarios.add(scenario)
            write_group(output, directory, group, rng, family, layout, scenario, split)
    return seal_dataset(directory, kind="controlled", groups=groups, seed=seed, split=split)


def write_group(output, directory, group, rng, family, layout, scenario, split):
    entries = ["K", "L", "M"]
    rng.shuffle(entries)
    target = rng.choice(entries)
    distractor = rng.choice([entry for entry in entries if entry != target])
    rows = {entry: random_values(rng) for entry in entries}
    rows[target] = scenario.base_values
    edited_rows = {**rows, target: scenario.edited_values}
    first, second = scenario.base_values
    edited_first, edited_second = scenario.edited_values
    changes = (edited_first - first, edited_second - second)
    index = 0 if changes[0] else 1
    change = changes[index]
    matching_values = list(rows[distractor])
    matching_values[index] = rng.randint(15 - min(change, 0), 85 - max(change, 0))
    rows[distractor] = tuple(matching_values)
    edited_rows[distractor] = rows[distractor]
    matching_values[index] += change
    placebo_rows = {**rows, distractor: tuple(matching_values)}
    image_names = {}
    for name, chart_rows in (("base", rows), ("edit", edited_rows), ("placebo", placebo_rows)):
        image_names[name] = f"images/{group}-{name}.png"
        draw_chart(chart_rows, layout, directory / image_names[name])
    conditions = (
        ("original", "base", scenario.base_rule, scenario.base_values),
        ("image_edit", "edit", scenario.base_rule, scenario.edited_values),
        ("rule_edit", "base", scenario.edited_rule, scenario.base_values),
        ("both_edits", "edit", scenario.edited_rule, scenario.edited_values),
        ("placebo", "placebo", scenario.base_rule, scenario.base_values),
    )
    style = 0 if split == "pilot" else 1
    conditions += (("rule_paraphrase", "base", scenario.base_rule, scenario.base_values),)
    for language in LANGUAGES:
        for condition, image, rule, values in conditions:
            rule_style = style + 2 if condition == "rule_paraphrase" else style
            record = {
                "id": f"{group}:{language}:{condition}",
                "group": group,
                "language": language,
                "condition": condition,
                "image": image_names[image],
                "target": target,
                "rule_text": rule.as_text(language, rule_style),
                "rule_code": rule.as_code(),
                "values": list(values),
                "answer": rule.applies(values),
                "family": family,
                "layout": layout,
                "source": "controlled",
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def seal_dataset(directory, **metadata):
    directory = Path(directory)
    examples_path = directory / "examples.jsonl"
    rows = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines()]
    images = sorted({row["image"] for row in rows})
    manifest = {
        "schema": SCHEMA,
        **metadata,
        "examples": len(rows),
        "examples_sha256": sha256_file(examples_path),
        "images": {name: sha256_file(directory / name) for name in images},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_dataset(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    examples_path = directory / "examples.jsonl"
    if manifest["schema"] != SCHEMA or sha256_file(examples_path) != manifest["examples_sha256"]:
        raise ValueError("Dataset manifest and examples do not match.")
    rows = [json.loads(line) for line in examples_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["examples"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Dataset example count or IDs are invalid.")
    if set(manifest["images"]) != {row["image"] for row in rows}:
        raise ValueError("Referenced images do not match the manifest.")
    for name, checksum in manifest["images"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != checksum:
            raise ValueError(f"Dataset image path or content has changed: {name}")
    for row in rows:
        rule = parse_rule(row["rule_code"])
        if rule is None or rule.applies(row["values"]) != row["answer"]:
            raise ValueError(f"Rule and answer do not agree for {row['id']}")
    group_counts = Counter((row["group"], row["language"]) for row in rows)
    if manifest["kind"] == "controlled" and set(group_counts.values()) != {len(CONDITIONS)}:
        raise ValueError("Each controlled group must contain every condition.")
    return rows, manifest, sha256_file(manifest_path)
