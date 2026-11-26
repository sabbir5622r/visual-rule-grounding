import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np

from analyze_results import DEFAULT_TABLES, LANGUAGES, MODELS, REPLAYS, ROOT

DEFAULT_FIGURES = ROOT / "results" / "paper_analysis" / "figures"
MODEL_NAMES = ("Qwen", "Gemma", "Centurio\nAya", "SmolVLM2")
LANGUAGE_NAMES = {"en": "English rules", "bn": "Bangla rules"}
METHOD_NAMES = {
    "direct": "Direct",
    "execute": "Free-form execution",
    "constrained": "Constrained execution",
}
METHOD_COLORS = {"direct": "#7F7F7F", "execute": "#0072B2", "constrained": "#009E73"}
REPLAY_NAMES = {
    "observed": "Model outputs",
    "verified_counts": "Verified counts",
    "verified_rule": "Verified rule",
    "both_verified": "Both verified",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--font-dir", type=Path)
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


def select_table(rows, dataset, series_column):
    return {(row["model"], row["language"], row[series_column]): row
            for row in rows if row["dataset"] == dataset}


def create_panels(y_label, lower, upper):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True)
    axes[0].set_ylabel(y_label)
    for axis, language in zip(axes, LANGUAGES, strict=True):
        axis.set_xticks(range(len(MODELS)), MODEL_NAMES)
        axis.set_xlabel(LANGUAGE_NAMES[language])
        axis.set_ylim(lower, upper)
        axis.set_xlim(-0.52, len(MODELS) - 0.48)
        axis.yaxis.set_major_locator(MultipleLocator(20))
        axis.grid(axis="y", color="0.88", linewidth=0.7)
        axis.set_axisbelow(True)
    return figure, axes


def save_figure(figure, axes, directory, name):
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                  ncol=len(labels), frameon=False, columnspacing=1.0)
    figure.subplots_adjust(left=0.11, right=0.99, bottom=0.37, top=0.97, wspace=0.13)
    figure.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(directory / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(figure)


def plot_bars(table, values, names, colors, y_label, limits, directory, filename):
    figure, axes = create_panels(y_label, *limits)
    positions = np.arange(len(MODELS))
    width = 0.76 / len(names)
    for axis, language in zip(axes, LANGUAGES, strict=True):
        for index, (series, label) in enumerate(names.items()):
            offset = (index - (len(names) - 1) / 2) * width
            heights = [100 * float(table[(model, language, series)][values[series]]) for model in MODELS]
            axis.bar(positions + offset, heights, width=width, label=label, color=colors[series],
                     edgecolor="white", linewidth=0.5)
    save_figure(figure, axes, directory, filename)


def plot_pair_results(rows, dataset, directory, filename):
    table = select_table(rows, dataset, "method")
    expected = {(model, language, method) for model in MODELS for language in LANGUAGES for method in METHOD_NAMES}
    if expected <= table.keys():
        plot_bars(table, dict.fromkeys(METHOD_NAMES, "all_four_accuracy"), METHOD_NAMES, METHOD_COLORS,
                  "All four correct (%)", (0, 60), directory, filename)


def plot_differences(rows, directory):
    figure, axes = create_panels("Change vs. direct (pp)", -40, 70)
    positions = np.arange(len(MODELS))
    table = select_table([row for row in rows if row["baseline"] == "direct"], "rord", "method")
    for axis, language in zip(axes, LANGUAGES, strict=True):
        axis.axhline(0, color="0.38", linewidth=0.9)
        for offset, method in zip((-0.14, 0.14), ("execute", "constrained"), strict=True):
            selected = [table[model, language, method] for model in MODELS]
            midpoint = 100 * np.array([float(row["capture_weighted_difference"]) for row in selected])
            low = 100 * np.array([float(row["ci_low"]) for row in selected])
            high = 100 * np.array([float(row["ci_high"]) for row in selected])
            errors = np.maximum(0, np.stack((midpoint - low, high - midpoint)))
            axis.errorbar(positions + offset, midpoint, yerr=errors, fmt="o",
                          color=METHOD_COLORS[method], capsize=4, markersize=6,
                          label=METHOD_NAMES[method])
    save_figure(figure, axes, directory, "03_paired_method_differences")


def plot_diagnostics(rows, directory):
    table = select_table(rows, "rord", "method")
    diagnostics = {
        (model, language, f"{method}_{metric}"): table[model, language, method]
        for model in MODELS for language in LANGUAGES
        for method in ("execute", "constrained") for metric in ("rule_parseable", "rule_exact")
    }
    names = {
        "execute_rule_parseable": "Free-form valid",
        "execute_rule_exact": "Free-form exact",
        "constrained_rule_parseable": "Constrained valid",
        "constrained_rule_exact": "Constrained exact",
    }
    colors = dict(zip(names, ("#56B4E9", "#0072B2", "#E69F00", "#009E73"), strict=True))
    columns = {key: key.split("_", 1)[1] for key in names}
    plot_bars(diagnostics, columns, names, colors, "Rule outputs (%)", (0, 105), directory,
              "04_rule_format_and_meaning")


def plot_component_replay(rows, directory):
    table = select_table([row for row in rows if row["method"] == "execute"], "rord", "substitution")
    colors = dict(zip(REPLAYS, ("#7F7F7F", "#0072B2", "#E69F00", "#009E73"), strict=True))
    plot_bars(table, dict.fromkeys(REPLAYS, "all_four_accuracy"), REPLAY_NAMES, colors,
              "All four correct (%)", (0, 105), directory, "05_component_replay")


def main():
    args = parse_arguments()
    set_style(args.font_dir)
    main_results = read_table(args.tables, "main_results.csv")
    differences = read_table(args.tables, "method_differences.csv")
    diagnostics = read_table(args.tables, "diagnostics.csv")
    replay = read_table(args.tables, "component_replay.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    plot_pair_results(main_results, "rord", args.output, "01_rord_all_four_correct")
    plot_pair_results(main_results, "heldout", args.output, "02_heldout_all_four_correct")
    plot_differences(differences, args.output)
    plot_diagnostics(diagnostics, args.output)
    plot_component_replay(replay, args.output)
    print(f"Saved PDF and 300-DPI PNG figures to {args.output}")


if __name__ == "__main__":
    main()
