from collections import Counter
from functools import partial
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from .dataset import load_dataset, sha256_file
from .methods import run_method
from .models import MODEL_IDS, MODEL_REVISIONS, VisionLanguageModel


def connect(path):
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS requested_method (name TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS prediction (
            method TEXT NOT NULL, example TEXT NOT NULL, json TEXT NOT NULL,
            PRIMARY KEY (method, example)
        );
        CREATE TABLE IF NOT EXISTS response (key TEXT PRIMARY KEY, text TEXT NOT NULL);
        """
    )
    return connection


def select_examples(rows, limits, languages):
    selected_groups = list(dict.fromkeys(row["group"] for row in rows))
    selected_groups = set(selected_groups[:limits] if limits else selected_groups)
    return [row for row in rows if row["group"] in selected_groups and row["language"] in languages]


def run_backend(args):
    started = time.monotonic()
    rows, manifest, dataset_hash = load_dataset(args.data)
    rows = select_examples(rows, args.take, args.languages)
    if not rows:
        raise ValueError("The selected dataset, languages and group limit have no examples.")
    methods = tuple(dict.fromkeys(args.methods))
    revision = args.revision or MODEL_REVISIONS.get(args.backend)
    if args.backend != "oracle":
        from huggingface_hub import HfApi

        if not revision or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            revision = HfApi().model_info(MODEL_IDS[args.backend], revision=revision).sha
    settings = {
        "dataset_sha256": dataset_hash,
        "dataset_kind": manifest["kind"],
        "split": manifest["split"],
        "backend": args.backend,
        "revision": revision,
        "visual_tokens": args.visual_tokens,
        "selection": {"languages": sorted(set(args.languages)), "take": args.take},
        "examples": len(rows),
        "implementation": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("runner.py", "methods.py", "models.py", "rules.py")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with connect(args.output / "run.sqlite") as connection:
        payload = json.dumps(settings, sort_keys=True)
        with connection:
            connection.execute("INSERT OR IGNORE INTO settings VALUES (1, ?)", (payload,))
        if connection.execute("SELECT json FROM settings WHERE id=1").fetchone()[0] != payload:
            raise ValueError("The output belongs to a different dataset, selection, model, revision or code.")
        with connection:
            connection.executemany("INSERT OR IGNORE INTO requested_method VALUES (?)", [(method,) for method in methods])
        (args.output / "settings.json").write_text(payload + "\n", encoding="utf-8")
        progress = infer_rows(args, connection, rows, methods, manifest, revision, started)
        temporary_path = args.output / "predictions.jsonl.tmp"
        with temporary_path.open("w", encoding="utf-8") as output:
            for (prediction,) in connection.execute("SELECT json FROM prediction ORDER BY method, example"):
                output.write(prediction + "\n")
        temporary_path.replace(args.output / "predictions.jsonl")
        print(json.dumps(progress, indent=2), flush=True)


def infer_rows(args, connection, rows, methods, manifest, revision, started):
    completed = set(connection.execute("SELECT method, example FROM prediction"))
    expected = len(rows) * len(methods)
    remaining = [(method, row) for row in rows for method in methods if (method, row["id"]) not in completed]
    print(f"{expected - len(remaining)}/{expected} selected evaluations already saved.", flush=True)
    if not remaining:
        return {"expected": expected, "completed": expected, "new": 0}
    model = None if args.backend == "oracle" else VisionLanguageModel(args.backend, revision, args.visual_tokens)
    inference_started = time.monotonic()
    calls = Counter()
    added = 0
    for method, row in remaining:
        if args.max_new and added == args.max_new:
            break
        if args.seconds and time.monotonic() - started >= args.seconds:
            break
        prior_calls = calls.copy()
        respond = partial(cached_response, connection, model, args.data, manifest["images"], calls)
        evaluation_started = time.monotonic()
        result = run_method(method, row, respond)
        metadata = {key: row[key] for key in ("id", "group", "language", "condition", "family", "layout", "source")}
        prediction = {
            **metadata,
            **result,
            "method": method,
            "backend": args.backend,
            "gold": row["answer"],
            "seconds": round(time.monotonic() - evaluation_started, 5),
            "model_calls": calls["model"] - prior_calls["model"],
            "cache_hits": calls["cached"] - prior_calls["cached"],
        }
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO prediction VALUES (?, ?, ?)",
                (method, row["id"], json.dumps(prediction, ensure_ascii=False)),
            )
        added += 1
        if added in (1, 5, 20) or added % args.log_every == 0:
            elapsed = time.monotonic() - inference_started
            print(
                f"{expected - len(remaining) + added}/{expected} | "
                f"{elapsed / added:.2f} s per new evaluation | "
                f"{calls['model']} uncached model calls | {calls['cached']} cache hits",
                flush=True,
            )
    gpu_peak = round(model.torch.cuda.max_memory_allocated(0) / (1024 ** 3), 3) if model else None
    return {
        "expected": expected,
        "completed": expected - len(remaining) + added,
        "new": added,
        "wall_seconds": round(time.monotonic() - started, 2),
        "prediction_seconds": round(time.monotonic() - inference_started, 2),
        "uncached_model_calls": calls["model"],
        "cache_hits": calls["cached"],
        "peak_allocated_gib": gpu_peak,
    }


def cached_response(connection, model, directory, images, calls, prompt, image, output_tokens, oracle):
    if model is None:
        return oracle
    identity = json.dumps([prompt, images[image] if image else None, output_tokens], ensure_ascii=False)
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    saved = connection.execute("SELECT text FROM response WHERE key=?", (key,)).fetchone()
    if saved is not None:
        calls["cached"] += 1
        return saved[0]
    answer = model.respond(prompt, directory / image if image else None, output_tokens)
    with connection:
        connection.execute("INSERT OR IGNORE INTO response VALUES (?, ?)", (key, answer))
    calls["model"] += 1
    return answer
