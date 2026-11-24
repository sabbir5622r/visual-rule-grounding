import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageOps

from .dataset import CORE_CONDITIONS, LANGUAGES, seal_dataset
from .rules import parse_rule


def required_text(record, field):
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text.")
    return value.strip()


def read_counts(record, field):
    counts = record[field]
    if not isinstance(counts, list) or len(counts) != 2:
        raise ValueError(f"{field} must be a list with counts for A and B.")
    if any(type(value) is not int or not 0 <= value <= 999 for value in counts):
        raise ValueError(f"{field} must contain integers from 0 to 999.")
    return tuple(counts)


def read_photo(path):
    with Image.open(path) as original:
        if original.format not in {"JPEG", "PNG", "WEBP"} or getattr(original, "n_frames", 1) != 1:
            raise ValueError(f"Use a single JPEG, PNG, or WebP photo: {path.name}")
        image = ImageOps.exif_transpose(original).convert("RGB")
    image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    image.info.clear()
    return image


def load_pair(record, source_dir):
    identifier = required_text(record, "id")
    if ":" in identifier:
        raise ValueError("A scene ID cannot contain a colon.")
    note = required_text(record, "measurement_note")
    source = required_text(record, "source")
    base_values = read_counts(record, "original_values")
    edited_values = read_counts(record, "edited_values")
    if sum(first != second for first, second in zip(base_values, edited_values)) != 1:
        raise ValueError("The photo edit must change exactly one annotated count.")

    original_rule = parse_rule(required_text(record, "original_rule"))
    edited_rule = parse_rule(required_text(record, "edited_rule"))
    if original_rule is None or edited_rule is None:
        raise ValueError("Use the supported rule grammar. Example: A GE 3 AND B LE 1.")
    base_clauses = original_rule.clauses
    edit_clauses = edited_rule.clauses
    if (
        tuple(clause.feature for clause in base_clauses) != tuple(clause.feature for clause in edit_clauses)
        or sum(first != second for first, second in zip(base_clauses, edit_clauses)) != 1
    ):
        raise ValueError("The rule edit must change one comparison and retain the same features.")
    base_label = original_rule.applies(base_values)
    if original_rule.applies(edited_values) == base_label:
        raise ValueError("The photo change alone must flip the correct answer.")
    if edited_rule.applies(base_values) == base_label:
        raise ValueError("The rule change alone must flip the correct answer.")

    paths = []
    for field in ("original_photo", "edited_photo"):
        path = (source_dir / required_text(record, field)).resolve()
        if not path.is_relative_to(source_dir):
            raise ValueError(f"{field} must point inside the review file's folder.")
        paths.append(path)
    if len(base_clauses) == 1:
        family = "gap" if base_clauses[0].feature == "GAP" else "cutoff"
    elif len({clause.feature for clause in base_clauses}) == 1:
        family = "range"
    else:
        family = "conjunction"
    return identifier, note, source, family, paths, base_values, edited_values, original_rule, edited_rule


def import_photos(review_path, directory):
    review_path = Path(review_path).resolve()
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError(f"Photo dataset output already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    scene_ids = set()
    image_ids = set()
    with TemporaryDirectory(dir=directory.parent) as temporary:
        staging = Path(temporary) / "dataset"
        (staging / "images").mkdir(parents=True)
        with review_path.open(encoding="utf-8-sig") as review, (staging / "examples.jsonl").open("x", encoding="utf-8") as output:
            for line_number, line in enumerate(review, 1):
                if not line.strip():
                    continue
                try:
                    pair = load_pair(json.loads(line), review_path.parent)
                    identifier, note, source, family, paths, base_values, edited_values, original_rule, edited_rule = pair
                    if identifier in scene_ids:
                        raise ValueError(f"Repeated scene ID: {identifier}")
                    scene_ids.add(identifier)
                    names = []
                    for edition, path in zip(("original", "edit"), paths):
                        image = read_photo(path)
                        digest = (image.size, hashlib.sha256(image.tobytes()).digest())
                        if digest in image_ids:
                            raise ValueError(f"Repeated image across photo inputs: {path.name}")
                        image_ids.add(digest)
                        name = f"images/photo-{len(scene_ids):05d}-{edition}.png"
                        image.save(staging / name)
                        names.append(name)
                    write_photo_group(output, identifier, note, source, family, names,
                                      base_values, edited_values, original_rule, edited_rule)
                except (KeyError, TypeError, ValueError, OSError) as error:
                    raise ValueError(f"{review_path.name}, line {line_number}: {error}") from error
        if not scene_ids:
            raise ValueError("The review file contains no photo pairs.")
        manifest = seal_dataset(staging, kind="photo", groups=len(scene_ids), split="photo_pilot")
        staging.rename(directory)
    return manifest


def write_photo_group(output, identifier, note, source, family, names,
                      base_values, edited_values, original_rule, edited_rule):
    group = f"photo-{identifier}"
    inputs = (
        (names[0], base_values, original_rule),
        (names[1], edited_values, original_rule),
        (names[0], base_values, edited_rule),
        (names[1], edited_values, edited_rule),
    )
    for language in LANGUAGES:
        for condition, (image, values, rule) in zip(CORE_CONDITIONS, inputs):
            rule_text = rule.as_text(language).replace("An entry", "A scene").replace("একটি এন্ট্রি", "একটি দৃশ্য")
            row = {
                "id": f"{group}:{language}:{condition}",
                "group": group,
                "language": language,
                "condition": condition,
                "image": image,
                "target": "scene",
                "measurement_note": note,
                "rule_text": rule_text,
                "rule_code": rule.as_code(),
                "values": list(values),
                "answer": rule.applies(values),
                "family": family,
                "layout": "photo",
                "source": source,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
