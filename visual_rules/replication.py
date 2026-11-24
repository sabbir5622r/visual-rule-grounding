import json
from pathlib import Path
import shutil

from PIL import Image

from .dataset import LANGUAGES, seal_dataset, sha256_file
from .rules import parse_rule


def import_charts(review_path, directory):
    review_path = Path(review_path)
    directory = Path(directory)
    records = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("The review file contains no charts.")
    groups = set()
    images = set()
    checked = []
    for record in records:
        identifier = record["id"]
        if not identifier or identifier in groups:
            raise ValueError("Each audited chart needs a unique, nonempty ID.")
        groups.add(identifier)
        if not all(str(record[key]).strip() for key in ("source_url", "source_id", "reviewer", "target", "measurement_note")):
            raise ValueError(f"{identifier} lacks provenance or review information.")
        source = review_path.parent / record["image"]
        with Image.open(source) as image:
            image.verify()
        image_sha256 = sha256_file(source)
        if image_sha256 in images:
            raise ValueError(f"Repeated public image in {identifier}. Keep one independent group per image.")
        images.add(image_sha256)
        values = record["values"]
        if len(values) != 2 or any(type(value) is not int or not 0 <= value <= 999 for value in values):
            raise ValueError(f"{identifier} requires two nonnegative integer measurements below 1000.")
        rules = [parse_rule(record[name]["code"]) for name in ("original", "rule_edit")]
        if None in rules or rules[0].applies(values) == rules[1].applies(values):
            raise ValueError(f"{identifier} needs two valid written rules with opposite decisions.")
        checked.append((record, source, rules))
    if (directory / "examples.jsonl").exists() or (directory / "images").exists():
        raise ValueError(f"Replication output already exists in {directory}. Use a fresh path.")
    (directory / "images").mkdir(parents=True)
    with (directory / "examples.jsonl").open("x", encoding="utf-8") as output:
        for index, (record, source, rules) in enumerate(checked, 1):
            name = f"images/chart-{index:05d}{source.suffix.lower()}"
            shutil.copyfile(source, directory / name)
            group = f"external-{record['id']}"
            for language in LANGUAGES:
                for condition, rule in zip(("original", "rule_edit"), rules):
                    row = {
                        "id": f"{group}:{language}:{condition}",
                        "group": group,
                        "language": language,
                        "condition": condition,
                        "image": name,
                        "target": record["target"],
                        "measurement_note": record["measurement_note"],
                        "rule_text": rule.as_text(language, style=1),
                        "rule_code": rule.as_code(),
                        "values": record["values"],
                        "answer": rule.applies(record["values"]),
                        "family": record.get("family", "external"),
                        "layout": "original_chart",
                        "source": record["source_url"],
                        "source_id": record["source_id"],
                        "reviewer": record["reviewer"],
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return seal_dataset(directory, kind="external", groups=len(checked), split="external")
