import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from uuid import uuid4


REPO = Path(__file__).resolve().parent
PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "Pillow", "huggingface-hub")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def capture(*command):
    try:
        result = subprocess.run(
            command, cwd=REPO, capture_output=True, text=True, check=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def package_versions():
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def save_record(path, record):
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def read_summary(path):
    with path.open("rb") as log:
        log.seek(max(0, path.stat().st_size - 65536))
        tail = log.read().decode("utf-8", errors="replace")
    index = tail.rfind("\n{\n")
    if index < 0:
        return {}
    try:
        summary, _ = json.JSONDecoder().raw_decode(tail[index + 1:])
    except json.JSONDecodeError:
        return {}
    return summary if isinstance(summary, dict) and {"completed", "expected"} <= summary.keys() else {}


def run_logged(arguments, directory):
    log_path = directory / "run.log"
    record_path = directory / "attempt.json"
    command = [sys.executable, "-u", "-m", "visual_rules", "run", *arguments]
    record = {
        "status": "started",
        "command": command,
        "git_commit": capture("git", "rev-parse", "HEAD"),
        "git_status": capture("git", "status", "--porcelain", "--untracked-files=no"),
        "python": platform.python_version(),
        "packages": package_versions(),
        "visible_gpus": capture(
            "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"
        ),
        "started_utc": utc_now(),
    }
    save_record(record_path, record)
    started = time.monotonic()
    exit_code = None
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                command, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            try:
                with process.stdout:
                    for line in process.stdout:
                        log.write(line)
                        print(line, end="", flush=True)
                exit_code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        record["status"] = "finished" if exit_code == 0 else "failed"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        exit_code = 130
    except OSError as error:
        record["status"] = "failed"
        record["error"] = str(error)
        exit_code = 1
    finally:
        seconds = time.monotonic() - started
        summary = read_summary(log_path) if log_path.is_file() else {}
        count = summary.get("gpus_used")
        record.update(
            finished_utc=utc_now(),
            command_wall_seconds=round(seconds, 3),
            exit_code=exit_code,
            runner_summary=summary,
            run_complete=bool(summary) and summary["completed"] == summary["expected"],
            gpu_hours=round(seconds * count / 3600, 6) if type(count) is int else None,
        )
        save_record(record_path, record)
        print(f"Compute record: {record_path}", flush=True)
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Run Visual Rule Grounding with a live log and a compute record.",
        allow_abbrev=False,
    )
    parser.add_argument("--output", required=True, type=Path, help="The visual_rules output directory.")
    args, run_arguments = parser.parse_known_args()
    output = args.output.resolve()
    attempt = output / "compute" / uuid4().hex[:12]
    attempt.mkdir(parents=True)
    return run_logged(["--output", str(output), *run_arguments], attempt)


if __name__ == "__main__":
    raise SystemExit(main())
