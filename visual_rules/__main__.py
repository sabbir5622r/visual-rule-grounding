import argparse
import json
from pathlib import Path

from .analysis import build_report
from .dataset import LANGUAGES, load_dataset, make_dataset
from .methods import METHODS, PRIMARY_METHODS
from .models import MODEL_IDS
from .replication import import_charts
from .runner import run_backend


def positive_integer(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return value


def main():
    parser = argparse.ArgumentParser(prog="python -m visual_rules")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Generate the controlled chart study.")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--split", choices=("pilot", "test"), default="pilot")
    prepare.add_argument("--groups", type=positive_integer)
    prepare.add_argument("--seed", type=int, default=2027)

    replication = commands.add_parser("import-charts", help="Import audited, unmodified public charts.")
    replication.add_argument("--review", type=Path, required=True)
    replication.add_argument("--output", type=Path, required=True)

    validate = commands.add_parser("validate", help="Check a sealed dataset and its images.")
    validate.add_argument("--data", type=Path, required=True)

    run = commands.add_parser("run", help="Evaluate one model and save each completed evaluation.")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--backend", choices=(*MODEL_IDS, "oracle"), default="qwen")
    run.add_argument("--revision")
    run.add_argument("--languages", nargs="+", choices=LANGUAGES, default=LANGUAGES)
    run.add_argument("--methods", nargs="+", choices=METHODS, default=PRIMARY_METHODS)
    run.add_argument("--take", type=positive_integer)
    run.add_argument("--max-new", type=positive_integer)
    run.add_argument("--seconds", type=positive_integer)
    run.add_argument("--visual-tokens", type=positive_integer, default=512)
    run.add_argument("--log-every", type=positive_integer, default=25)

    analyze = commands.add_parser("analyze", help="Report group-level metrics and paired differences.")
    analyze.add_argument("--runs", type=Path, nargs="+", required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--bootstrap", type=positive_integer, default=2000)
    analyze.add_argument("--seed", type=int, default=117)
    analyze.add_argument("--allow-partial", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "prepare":
            groups = args.groups or {"pilot": 32, "test": 384}[args.split]
            manifest = make_dataset(args.output, groups, args.seed, args.split)
            print(f"Created {manifest['examples']} examples across {groups} scenarios.")
        elif args.command == "import-charts":
            manifest = import_charts(args.review, args.output)
            print(f"Imported {manifest['groups']} reviewed external charts.")
        elif args.command == "validate":
            rows, manifest, digest = load_dataset(args.data)
            print(json.dumps({"examples": len(rows), "kind": manifest["kind"], "sha256": digest}, indent=2))
        elif args.command == "run":
            if args.visual_tokens < 128:
                parser.error("--visual-tokens must be at least 128.")
            run_backend(args)
        else:
            build_report(args.runs, args.output, args.bootstrap, args.seed, args.allow_partial)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
