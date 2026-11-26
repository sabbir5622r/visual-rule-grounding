import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
from textwrap import fill

import matplotlib.pyplot as plt
from PIL import Image

from analyze_results import find_runs

EXAMPLES = {
    "rord": (
        ("photo-I-211011_I05017_T16", "Staircase", "Cats", "Black wall fixtures"),
        ("photo-I-211014_I02008_W05", "Kitchen", "Yellow-green mugs", "Metal saucepans"),
        ("photo-I-210715_I06019_T02", "Parking garage", "Black SUVs right of M3", "M3 pillars"),
    ),
    "heldout": (
        ("photo-test_backyard_toys-pair_01", "Backyard", "White plastic chairs", "Green playhouses"),
        ("photo-test_office-pair_01", "Office", "Red chairs", "Gray tabletop plates"),
        ("photo-stairwell-pair_01", "Stairwell", "Green emergency signs", "Teal backpacks"),
    ),
    "scenediff": (
        ("photo-hallway_1_hallway_2", "Hallway", "Black backpacks", "Red fire extinguishers"),
        ("photo-bus_1_bus_2", "Bus", "White parcels", "Blue hinged panels"),
        ("photo-post_office_1_post_office_2", "Post office", "Blue letter slots", "White mailing parcels"),
    ),
}


def load_examples(results):
    selected = {}
    for dataset, source, archive in find_runs(results):
        if dataset in selected:
            continue
        manifest_path = next(name for name in archive.namelist() if name.endswith("/data/manifest.json"))
        base = PurePosixPath(manifest_path).parent
        manifest_bytes = archive.read(manifest_path)
        manifest = json.loads(manifest_bytes)
        examples = archive.read(str(base / "examples.jsonl"))
        if sha256(examples).hexdigest() != manifest["examples_sha256"]:
            raise ValueError(f"Example checksum mismatch in {source}")
        grouped = {group: {} for group, *_ in EXAMPLES[dataset]}
        for line in examples.splitlines():
            row = json.loads(line)
            if row["language"] == "en" and row["group"] in grouped:
                grouped[row["group"]][row["condition"]] = row
        selected[dataset] = []
        for group, setting, label_a, label_b in EXAMPLES[dataset]:
            conditions = grouped[group]
            pictures = []
            for condition in ("original", "image_edit"):
                path = conditions[condition]["image"]
                image_data = archive.read(str(base / path))
                if sha256(image_data).hexdigest() != manifest["images"][path]:
                    raise ValueError(f"Image checksum mismatch for {path}")
                with Image.open(BytesIO(image_data)) as image:
                    pictures.append(image.convert("RGB"))
            selected[dataset].append({
                "setting": setting, "labels": (label_a, label_b), "pictures": pictures,
                "conditions": conditions, "source": source,
                "dataset_sha256": sha256(manifest_bytes).hexdigest(),
            })
    return selected


def show_photo(axis, image, label):
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xlabel(label, fontsize=18, labelpad=8)


def counts_label(row):
    return f"A = {row['values'][0]}, B = {row['values'][1]}"


def plot_gallery(examples):
    figure, axes = plt.subplots(len(examples), 2, figsize=(11.5, 11), layout="constrained")
    for row_axes, example in zip(axes, examples, strict=True):
        label_a, label_b = example["labels"]
        for axis, image, condition, side in zip(
            row_axes, example["pictures"], ("original", "image_edit"), ("Original", "Reference"), strict=True,
        ):
            counts = counts_label(example["conditions"][condition])
            caption = f"{example['setting']} · {side}\n{counts}"
            if side == "Original":
                caption += "\n" + fill(f"A: {label_a}  ·  B: {label_b}", 47)
            show_photo(axis, image, caption)
    return figure


def plot_rule_example(example):
    figure = plt.figure(figsize=(11.5, 7.1), layout="constrained")
    layout = figure.add_gridspec(2, 2, height_ratios=(2, 1.25))
    conditions = example["conditions"]
    for col, image, condition, side in zip(
        range(2), example["pictures"], ("original", "image_edit"), ("Original image", "Reference image"), strict=True,
    ):
        show_photo(figure.add_subplot(layout[0, col]), image, f"{side}\n{counts_label(conditions[condition])}")

    axis = figure.add_subplot(layout[1, :])
    axis.axis("off")
    rule_rows = []
    for label, left, right in (
        ("Original rule", "original", "image_edit"),
        ("Edited rule", "rule_edit", "both_edits"),
    ):
        rule = conditions[left]["rule_text"].removeprefix("A scene qualifies if and only if ").rstrip(".")
        decisions = ["Qualifies" if conditions[key]["answer"] else "Does not qualify" for key in (left, right)]
        rule_rows.append([f"{label}\n{fill(rule, 37)}", *decisions])
    table = axis.table(cellText=rule_rows, colLabels=("Written rule", "Original image", "Reference image"),
                       colWidths=(0.48, 0.26, 0.26), cellLoc="center", bbox=(0.01, 0.02, 0.98, 0.78))
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C4CAD0")
        if row == 0 or col == 0:
            cell.set_facecolor("#E9EFF4")
        else:
            cell.set_facecolor("#DBEEE6" if cell.get_text().get_text() == "Qualifies" else "#F7E8DB")
    label_a, label_b = example["labels"]
    axis.text(0.5, 0.94, f"A: {label_a}  ·  B: {label_b}", ha="center", va="center", transform=axis.transAxes)
    return figure


def write_photo_manifest(directory, selected):
    fields = ("dataset", "group", "condition", "image", "source", "dataset_sha256", "measurement_note")
    with (directory / "dataset_figure_sources.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for dataset, examples in selected.items():
            for example in examples:
                for condition in ("original", "image_edit"):
                    row = example["conditions"][condition]
                    writer.writerow({**{name: row[name] for name in ("group", "condition", "image", "measurement_note")},
                                     "dataset": dataset, "source": example["source"],
                                     "dataset_sha256": example["dataset_sha256"]})
