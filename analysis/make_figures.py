import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

from analyze_results import CORE_CONDITIONS, DEFAULT_TABLES, LANGUAGES, METHODS, MODELS, REPLAYS, ROOT
from dataset_figures import load_examples, plot_gallery, plot_rule_example, write_photo_manifest

DEFAULT_FIGURES = ROOT / "results" / "paper_analysis" / "figures"
MODEL_NAMES = dict(zip(MODELS, ("Qwen", "Gemma", "Centurio Aya", "SmolVLM2"), strict=True))
LANGUAGE_NAMES = {"en": "English rules", "bn": "Bangla rules"}
METHOD_NAMES = {
    "direct": "Direct",
    "execute": "Free-form execution",
    "constrained": "Constrained execution",
}
METHOD_COLORS = {"direct": "#7F7F7F", "execute": "#0072B2", "constrained": "#009E73"}



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--font-dir", type=Path)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    return parser.parse_args()


def set_style(font_directory):
    font_name = "Times New Roman"
    if font_directory is not None:
        for path in font_directory.iterdir():
            if path.suffix.lower() in (".ttf", ".ttc", ".otf"):
                font_manager.fontManager.addfont(str(path))
    try:
        font_manager.findfont(font_manager.FontProperties(family=font_name), fallback_to_default=False)
    except ValueError as error:
        raise RuntimeError("Install Times New Roman or pass --font-dir with its font files.") from error
    plt.rcParams.update({
        "font.family": font_name,
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def read_table(folder, name):
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"Run python analysis/analyze_results.py first. Missing {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


METHOD_TICKS = ("Direct", "Free-form\nexecution", "Constrained\nexecution")
MODEL_COLORS = dict(zip(MODELS, ("#0072B2", "#E69F00", "#009E73", "#CC79A7"), strict=True))


def save_figure(figure, directory, name):
    for extension in ("pdf", "png"):
        figure.savefig(directory / f"{name}.{extension}", bbox_inches="tight", dpi=300)
    plt.close(figure)


def place_shared_legend(figure, axis, left=0.15, bottom=0.29, **settings):
    handles, labels = axis.get_legend_handles_labels()
    figure.set_layout_engine(None)
    figure.subplots_adjust(left=left, right=0.98, bottom=bottom, top=0.96, wspace=0.16)
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), frameon=False, **settings)


def select_rows(rows, dataset, *keys):
    return {tuple(row[key] for key in keys): row for row in rows if row["dataset"] == dataset}


def heatmap(axis, values, row_labels, column_labels, scale, cmap="Blues", precision=0):
    values = np.asarray(values)
    rendered = axis.imshow(values, cmap=cmap, norm=scale, aspect="auto")
    axis.set_xticks(range(len(column_labels)), column_labels)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for (row, column), value in np.ndenumerate(values):
        if np.isfinite(value):
            axis.text(column, row, f"{value:.{precision}f}", ha="center", va="center",
                      color="white" if scale(value) > 0.64 else "#222222")
    return rendered


def plot_pair_accuracy(rows, dataset, directory, filename):
    table = select_rows(rows, dataset, "model", "language", "method")
    models = [model for model in MODELS if all((model, language, method) in table
              for language in LANGUAGES for method in METHODS)]
    if not models:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.0 if len(models) > 1 else 2.3), layout="constrained")
    for axis, language in zip(axes, LANGUAGES, strict=True):
        values = [[100 * float(table[model, language, method]["all_four_accuracy"]) for method in METHODS]
                  for model in models]
        rendered = heatmap(axis, values, [MODEL_NAMES[model] for model in models], METHOD_TICKS, Normalize(0, 60),
                           precision=1)
        axis.set_xlabel(LANGUAGE_NAMES[language])
    axes[1].tick_params(labelleft=False)
    figure.colorbar(rendered, ax=axes, label="All four correct (%)", shrink=0.85)
    save_figure(figure, directory, filename)


def plot_paired_effects(rows, directory):
    table = select_rows([row for row in rows if row["baseline"] == "direct"], "rord", "model", "language", "method")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True, layout="constrained")
    positions = np.arange(len(MODELS))
    for axis, language in zip(axes, LANGUAGES, strict=True):
        axis.axvline(0, color="0.5", linewidth=0.9)
        for offset, method in zip((-0.15, 0.15), METHODS[1:], strict=True):
            selected = [table[model, language, method] for model in MODELS]
            center = 100 * np.array([float(row["capture_weighted_difference"]) for row in selected])
            low = 100 * np.array([float(row["ci_low"]) for row in selected])
            high = 100 * np.array([float(row["ci_high"]) for row in selected])
            error = np.maximum(0, np.stack((center - low, high - center)))
            axis.errorbar(center, positions + offset, xerr=error, fmt="o", capsize=4,
                          color=METHOD_COLORS[method], label=METHOD_NAMES[method])
        axis.set_xlabel(f"{LANGUAGE_NAMES[language]}\nChange vs. direct (pp)")
        axis.set_xlim(-34, 65)
        axis.set_yticks(positions, list(MODEL_NAMES.values()))
        axis.grid(axis="x", color="0.88")
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    place_shared_legend(figure, axes[-1], ncol=2)
    save_figure(figure, directory, "06_paired_method_effects")


def plot_conditions(rows, directory):
    table = select_rows(rows, "rord", "model", "language", "method", "condition")
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), layout="constrained")
    for index, (axis, model) in enumerate(zip(axes.flat, MODELS, strict=True)):
        column = index % 2
        values = [[100 * float(table[model, language, method, condition]["accuracy"]) for condition in CORE_CONDITIONS]
                  for language in LANGUAGES for method in METHODS]
        labels = ("EN · Direct", "EN · Free-form", "EN · Constrained", "BN · Direct", "BN · Free-form", "BN · Constrained")
        rendered = heatmap(axis, values, labels, ("Orig.", "Image", "Rule", "Both"), Normalize(0, 100))
        axis.axhline(2.5, color="white", linewidth=2)
        axis.set_xlabel(MODEL_NAMES[model])
        axis.tick_params(labelleft=column == 0)
    figure.colorbar(rendered, ax=axes, label="Decision accuracy (%)", shrink=0.85)
    save_figure(figure, directory, "07_accuracy_by_intervention")


def plot_pair_outcomes(rows, directory):
    table = select_rows(rows, "rord", "model", "language", "method", "correct_conditions")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 6.6), sharey=True, layout="constrained")
    order = [(model, method) for model in MODELS for method in METHODS]
    positions = np.array([index + 0.5 * (index // 3) for index in range(len(order))])
    labels = [f"{MODEL_NAMES[model]} · {('Direct', 'Free', 'Constr.')[METHODS.index(method)]}" for model, method in order]
    colors = ("#E7E7E7", "#CCDDDD", "#83B9B3", "#418E7A", "#00664E")
    for axis, language in zip(axes, LANGUAGES, strict=True):
        left = np.zeros(len(order))
        for correct, color in enumerate(colors):
            values = np.array([100 * float(table[model, language, method, str(correct)]["fraction"])
                               for model, method in order])
            axis.barh(positions, values, left=left, height=0.76, color=color, label=str(correct), edgecolor="white")
            left += values
        axis.set_xlim(0, 100)
        axis.set_yticks(positions, labels)
        axis.set_xlabel(f"{LANGUAGE_NAMES[language]}\nShare of image pairs (%)")
    axes[0].invert_yaxis()
    place_shared_legend(figure, axes[-1], left=0.24, bottom=0.28, ncol=5,
                        title="Number of conditions answered correctly", title_fontsize=18)
    save_figure(figure, directory, "08_pair_outcome_distribution")


def plot_rule_quality(rows, directory):
    table = select_rows(rows, "rord", "model", "language", "method")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharey=True, layout="constrained")
    positions = np.arange(len(MODELS))
    for axis, language in zip(axes, LANGUAGES, strict=True):
        for offset, method in zip((-0.15, 0.15), METHODS[1:], strict=True):
            selected = [table[model, language, method] for model in MODELS]
            exact = [100 * float(row["rule_exact"]) for row in selected]
            valid = [100 * float(row["rule_parseable"]) for row in selected]
            axis.hlines(positions + offset, exact, valid, color=METHOD_COLORS[method], linewidth=2, alpha=0.65)
            axis.plot(exact, positions + offset, "o", color=METHOD_COLORS[method], label=METHOD_NAMES[method])
            axis.plot(valid, positions + offset, "|", color=METHOD_COLORS[method],
                      markersize=10, markeredgewidth=2)
        axis.set_yticks(positions, list(MODEL_NAMES.values()))
        axis.set_xlim(-5, 105)
        axis.set_xlabel(f"{LANGUAGE_NAMES[language]}\nExact (dot) to valid (tick), %")
        axis.grid(axis="x", color="0.9")
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    place_shared_legend(figure, axes[-1], ncol=2)
    save_figure(figure, directory, "09_rule_validity_and_exact_match")


def plot_replay(rows, directory):
    table = select_rows([row for row in rows if row["method"] == "execute"], "rord", "model", "language", "substitution")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), layout="constrained")
    for axis, language in zip(axes, LANGUAGES, strict=True):
        values = [[100 * float(table[model, language, replay]["all_four_accuracy"]) for replay in REPLAYS]
                  for model in MODELS]
        rendered = heatmap(axis, values, list(MODEL_NAMES.values()),
                           ("Model\noutputs", "Verified\ncounts", "Verified\nrule", "Both\nverified"),
                           Normalize(0, 100), cmap="Greens", precision=1)
        axis.set_xlabel(LANGUAGE_NAMES[language])
    axes[1].tick_params(labelleft=False)
    figure.colorbar(rendered, ax=axes, label="All four correct (%)", shrink=0.85)
    save_figure(figure, directory, "10_component_substitutions")


def plot_decision_changes(rows, directory):
    table = select_rows(rows, "rord", "model", "language", "method", "condition", "expected_flip")
    conditions = (("image_edit", "True"), ("rule_edit", "True"), ("both_edits", "True"), ("both_edits", "False"))
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), layout="constrained")
    for index, (axis, model) in enumerate(zip(axes.flat, MODELS, strict=True)):
        values = [[100 * float(table[model, language, method, condition, flip]["valid_consistent_rate"])
                   for condition, flip in conditions] for language in LANGUAGES for method in METHODS]
        labels = ("EN · Direct", "EN · Free-form", "EN · Constrained", "BN · Direct", "BN · Free-form", "BN · Constrained")
        rendered = heatmap(axis, values, labels, ("Image\nflip", "Rule\nflip", "Both\nflip", "Both\nkeep"),
                           Normalize(0, 100), cmap="Purples")
        axis.axhline(2.5, color="white", linewidth=2)
        axis.set_xlabel(MODEL_NAMES[model])
        axis.tick_params(labelleft=index % 2 == 0)
    figure.colorbar(rendered, ax=axes, label="Expected decision change (%)", shrink=0.85)
    save_figure(figure, directory, "11_decision_change_consistency")


def plot_accuracy_gap(rows, directory):
    table = select_rows(rows, "rord", "model", "language", "method")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.9), sharex=True, sharey=True, layout="constrained")
    for axis, language in zip(axes, LANGUAGES, strict=True):
        axis.plot([0, 100], [0, 100], linestyle=":", color="0.55", linewidth=1)
        for model in MODELS:
            for method, marker in zip(METHODS, ("o", "s", "^"), strict=True):
                row = table[model, language, method]
                axis.scatter(100 * float(row["accuracy"]), 100 * float(row["all_four_accuracy"]),
                             marker=marker, color=MODEL_COLORS[model], s=68)
        axis.set_xlabel(f"{LANGUAGE_NAMES[language]}\nIndividual decision accuracy (%)")
        axis.set_xlim(-4, 103)
        axis.set_xticks(range(0, 101, 20))
        axis.set_ylim(-4, 63)
    axes[0].set_ylabel("All four correct (%)")
    model_handles = [
        Line2D([], [], color=MODEL_COLORS[model], marker="o", linestyle="none", label=MODEL_NAMES[model])
        for model in MODELS
    ]
    method_handles = [
        Line2D([], [], color="0.3", marker=marker, linestyle="none", label=METHOD_NAMES[method])
        for method, marker in zip(METHODS, ("o", "s", "^"), strict=True)
    ]
    figure.set_layout_engine(None)
    figure.subplots_adjust(left=0.1, right=0.98, bottom=0.28, top=0.85, wspace=0.16)
    figure.legend(handles=model_handles, loc="lower center", ncol=4, frameon=False)
    figure.legend(handles=method_handles, loc="upper center", ncol=3, frameon=False)
    save_figure(figure, directory, "12_decision_accuracy_vs_pair_accuracy")


def plot_compute(rows, directory):
    table = select_rows(rows, "rord", "model")
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True, layout="constrained")
    positions = np.arange(len(MODELS))
    for axis, metric, label in zip(
        axes, ("gpu_hours", "peak_reserved_gib"), ("Logged evaluation GPU-hours", "Peak reserved GPU memory (GiB)"),
        strict=True,
    ):
        values = [float(table[model,][metric]) for model in MODELS]
        axis.hlines(positions, 0, values, color="0.8", linewidth=2)
        axis.scatter(values, positions, s=75, color=list(MODEL_COLORS.values()))
        for value, position in zip(values, positions, strict=True):
            axis.annotate(f"{value:.2f}", (value, position), xytext=(7, 0), textcoords="offset points", va="center")
        axis.set_xlim(0, max(values) * 1.23)
        axis.set_ylim(-0.55, len(MODELS) - 0.45)
        axis.set_yticks(positions, list(MODEL_NAMES.values()))
        axis.set_xlabel(label)
    axes[0].invert_yaxis()
    save_figure(figure, directory, "13_compute_and_memory")


def plot_count_quality(rows, directory):
    table = select_rows([row for row in rows if row["language"] == "en" and row["method"] == "execute"],
                        "rord", "model")
    figure, axis = plt.subplots(figsize=(8.8, 3.9), layout="constrained")
    positions = np.arange(len(MODELS))
    exact = np.array([100 * float(table[model,]["counts_exact"]) for model in MODELS])
    valid = np.array([100 * float(table[model,]["counts_parseable"]) for model in MODELS])
    axis.barh(positions, exact, label="Exact counts", color="#009E73")
    axis.barh(positions, valid - exact, left=exact, label="Valid but wrong", color="#E69F00")
    axis.barh(positions, 100 - valid, left=valid, label="Unparseable", color="#7F7F7F")
    axis.set_yticks(positions, list(MODEL_NAMES.values()))
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Visual count outputs (%)")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)
    save_figure(figure, directory, "14_visual_count_errors")


def main():
    args = parse_arguments()
    set_style(args.font_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    tables = {name: read_table(args.tables, f"{name}.csv") for name in (
        "main_results", "method_differences", "condition_results", "pair_outcomes",
        "diagnostics", "component_replay", "decision_flips", "compute",
    )}
    examples = load_examples(args.results)
    if "rord" in examples:
        save_figure(plot_rule_example(examples["rord"][0]), args.output, "00_written_rule_and_image_pair")
    for dataset, stem in (("rord", "01_rord_photo_gallery"), ("heldout", "02_heldout_photo_gallery"),
                          ("scenediff", "03_scenediff_photo_gallery")):
        if dataset in examples:
            save_figure(plot_gallery(examples[dataset]), args.output, stem)
    if examples:
        write_photo_manifest(args.output, examples)
    for dataset, stem in (("rord", "04_rord_pair_accuracy"), ("heldout", "05_heldout_pair_accuracy"),
                          ("scenediff", "15_scenediff_pair_accuracy")):
        plot_pair_accuracy(tables["main_results"], dataset, args.output, stem)
    for plot, table in (
        (plot_paired_effects, "method_differences"), (plot_conditions, "condition_results"),
        (plot_pair_outcomes, "pair_outcomes"), (plot_rule_quality, "diagnostics"),
        (plot_replay, "component_replay"), (plot_decision_changes, "decision_flips"),
        (plot_accuracy_gap, "main_results"), (plot_compute, "compute"),
        (plot_count_quality, "diagnostics"),
    ):
        plot(tables[table], args.output)
    print(f"Saved PDF and 300-DPI PNG figures to {args.output}")


if __name__ == "__main__":
    main()
