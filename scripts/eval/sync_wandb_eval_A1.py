#!/usr/bin/env python
"""
Synchronize all offline W&B runs of one Task A.1 eval_id.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Add the project root to sys.path so that 'src' can be imported
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.part1_ST.task_A1_gene_predict.src_A1.common import resolve_eval_task_root


MARKER_NAME = ".histost_wandb_sync.json"
FAILED_MARKER_NAME = ".histost_wandb_sync_failed.json"
LOCK_NAME = ".histost_wandb_sync.lock"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def resolve_eval_root() -> Path:

    try:
        return Path(resolve_eval_task_root()).expanduser().resolve()

    except Exception as e:
        raise RuntimeError(
            "Could not resolve eval root automatically from "
            "`src_A1.common.resolve_eval_task_root()`.\n"
            "Either run this script from inside the HistoST repo environment, "
            "or pass:\n"
            "  --eval_root /path/to/eval/A-1_eval_st_predictions\n"
            f"Original error: {repr(e)}"
        ) from e


def file_signature(path: Path, eval_dir: Path) -> Dict[str, object]:
    st = path.stat()
    return {
        "name": path.name,
        "relative_path": str(path.relative_to(eval_dir)),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def current_wandb_file_signatures(run_dir: Path, eval_dir: Path) -> List[Dict[str, object]]:
    wandb_files = sorted(
        p
        for p in run_dir.glob("*.wandb")
        if p.is_file()
        and not p.name.startswith("._")  # ignore macOS AppleDouble metadata files
    )
    return [file_signature(p, eval_dir) for p in wandb_files]


def marker_matches_current_files(run_dir: Path, eval_dir: Path) -> bool:
    marker_path = run_dir / MARKER_NAME

    if not marker_path.exists():
        return False

    try:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if payload.get("status") != "synced":
        return False

    old_signatures = payload.get("wandb_files", [])
    new_signatures = current_wandb_file_signatures(run_dir, eval_dir)

    return old_signatures == new_signatures and len(new_signatures) > 0


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def discover_wandb_run_dirs(eval_dir: Path) -> List[Path]:
    """Find every folder below eval_dir that directly contains .wandb files (offline run folders)."""
    wandb_files = sorted(
        p
        for p in eval_dir.rglob("*.wandb")
        if p.is_file()
        and not p.name.startswith("._")  # ignore macOS AppleDouble metadata files
        and LOCK_NAME not in p.parts
        and MARKER_NAME not in p.parts
    )

    return sorted({p.parent for p in wandb_files})


def acquire_lock(run_dir: Path, force: bool = False) -> Tuple[bool, Path]:
    """
    Use atomic directory creation as a local lock.

    This prevents two instances of this script from syncing the same W&B run folder
    at the same time.
    """
    lock_dir = run_dir / LOCK_NAME

    if force and lock_dir.exists():
        shutil.rmtree(lock_dir, ignore_errors=True)

    try:
        lock_dir.mkdir(exist_ok=False)
    except FileExistsError:
        return False, lock_dir

    owner_payload = {
        "pid": os.getpid(),
        "created_utc": utc_now(),
        "host": os.uname().nodename if hasattr(os, "uname") else None,
    }
    write_json_atomic(lock_dir / "owner.json", owner_payload)

    return True, lock_dir


def release_lock(lock_dir: Path) -> None:
    shutil.rmtree(lock_dir, ignore_errors=True)


def sync_one_run_dir(
    *,
    run_dir: Path,
    eval_dir: Path,
    eval_id: str,
    repo_root: Path | None,
    wandb_extra_args: List[str],
    force: bool,
    dry_run: bool,
    verbose: bool,
) -> Tuple[str, str]:
    rel_run_dir = str(run_dir.relative_to(eval_dir))
    signatures = current_wandb_file_signatures(run_dir, eval_dir)
    file_names = ", ".join(str(s["name"]) for s in signatures)

    if not signatures:
        return "skipped", f"[SKIP] {rel_run_dir}: no .wandb file found."

    if not force and marker_matches_current_files(run_dir, eval_dir):
        return "already", f"[OK already synced] {rel_run_dir}: {file_names}"

    if dry_run:
        return "would_sync", f"[DRY RUN would sync] {rel_run_dir}: {file_names}"

    locked, lock_dir = acquire_lock(run_dir, force=force)
    if not locked:
        return (
            "locked",
            f"[SKIP locked] {rel_run_dir}: another sync process appears to be working here.",
        )

    try:
        # Re-check after lock acquisition, in case another process synced it first.
        if not force and marker_matches_current_files(run_dir, eval_dir):
            return "already", f"[OK already synced] {rel_run_dir}: {file_names}"

        real_wandb_files = sorted(
            p
            for p in run_dir.glob("*.wandb")
            if p.is_file()
            and not p.name.startswith("._")
        )

        if len(real_wandb_files) != 1:
            return (
                "failed",
                f"[FAILED] {rel_run_dir}: expected exactly one real .wandb file, "
                f"found {len(real_wandb_files)}: {[p.name for p in real_wandb_files]}",
            )

        wandb_file = real_wandb_files[0]

        cmd = ["wandb", "sync", "--include-synced", "--include-offline", *wandb_extra_args, str(wandb_file)]

        env = os.environ.copy()
        env["WANDB_MODE"] = "online"

        proc = subprocess.run(
            cmd,
            cwd=str(repo_root if repo_root is not None else Path.cwd()),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        command_text = " ".join(shlex.quote(x) for x in cmd)

        stdout = proc.stdout or ""
        lower_stdout = stdout.lower()

        actually_uploaded = (
            proc.returncode == 0
            and "skipping directory" not in lower_stdout
            and "skipping file" not in lower_stdout
            and (
                "syncing:" in lower_stdout
                or "wandb.ai/" in lower_stdout
                or "view run at" in lower_stdout
                or "done" in lower_stdout
            )
        )

        if actually_uploaded:
            payload = {
                "status": "synced",
                "synced_at_utc": utc_now(),
                "eval_id": eval_id,
                "run_dir": str(run_dir),
                "wandb_files": current_wandb_file_signatures(run_dir, eval_dir),
                "sync_command": command_text,
                "wandb_stdout_tail": stdout[-8000:],
            }
            write_json_atomic(run_dir / MARKER_NAME, payload)

            failed_marker = run_dir / FAILED_MARKER_NAME
            if failed_marker.exists():
                failed_marker.unlink()

            msg = f"[SYNCED uploaded] {rel_run_dir}: {file_names}"
            if verbose and proc.stdout.strip():
                msg += "\n" + proc.stdout.strip()

            return "synced", msg

        failure_payload = {
            "status": "failed",
            "failed_at_utc": utc_now(),
            "eval_id": eval_id,
            "run_dir": str(run_dir),
            "wandb_files": current_wandb_file_signatures(run_dir, eval_dir),
            "sync_command": command_text,
            "exit_code": int(proc.returncode),
            "output_tail": stdout[-8000:],
        }
        write_json_atomic(run_dir / FAILED_MARKER_NAME, failure_payload)

        msg = (
            f"[FAILED or skipped by wandb] {rel_run_dir}: {file_names}\n"
            f"Command: {command_text}\n"
            f"Exit code: {proc.returncode}\n"
            f"{stdout.strip()}"
        )
        return "failed", msg

    finally:
        release_lock(lock_dir)



def main(args) -> int:

    args.jobs = max(1, int(args.jobs))

    repo_root = project_root
    eval_root = resolve_eval_root()
    eval_dir = eval_root / args.eval_id

    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    if not args.dry_run and shutil.which("wandb") is None:
        raise RuntimeError(
            "`wandb` CLI was not found. Install wandb and run `wandb login` first."
        )

    wandb_extra_args = shlex.split(args.wandb_args)
    run_dirs = discover_wandb_run_dirs(eval_dir)

    print(f"Eval directory: {eval_dir}")
    print(f"Discovered W&B run folders: {len(run_dirs)}")
    print(f"Jobs: {args.jobs}")
    print("")

    if not run_dirs:
        print("No .wandb files found under this eval_id.")
        return 0

    counts: Dict[str, int] = {
        "already": 0,
        "synced": 0,
        "would_sync": 0,
        "failed": 0,
        "locked": 0,
        "skipped": 0,
    }

    if args.jobs == 1:
        for run_dir in run_dirs:
            status, msg = sync_one_run_dir(
                run_dir=run_dir,
                eval_dir=eval_dir,
                eval_id=args.eval_id,
                repo_root=repo_root,
                wandb_extra_args=wandb_extra_args,
                force=args.force,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
            counts[status] = counts.get(status, 0) + 1
            print(msg)

    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [
                executor.submit(
                    sync_one_run_dir,
                    run_dir=run_dir,
                    eval_dir=eval_dir,
                    eval_id=args.eval_id,
                    repo_root=repo_root,
                    wandb_extra_args=wandb_extra_args,
                    force=args.force,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )
                for run_dir in run_dirs
            ]

            for future in as_completed(futures):
                status, msg = future.result()
                counts[status] = counts.get(status, 0) + 1
                print(msg)

    print("")
    print("Summary")
    print("-------")
    for key in ["already", "synced", "would_sync", "failed", "locked", "skipped"]:
        print(f"{key}: {counts.get(key, 0)}")

    return 1 if counts.get("failed", 0) > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Synchronize every offline W&B run folder under one eval/A-1_eval_st_predictions/<eval_id> directory."))

    parser.add_argument("--eval_id", type=str,
                        required=True,
                        help="Evaluation ID, e.g. 'eval_1'.")

    parser.add_argument("--jobs", type=int,
                        default=1,
                        help=("Number of concurrent `wandb sync` processes. Default is 1. Use 2-4 carefully."))

    parser.add_argument("--force", action="store_true",
                        help="Ignore existing local sync markers and run `wandb sync` again.")

    parser.add_argument("--dry_run",
                        action="store_true",
                        help="Print what would be synced, but do not run `wandb sync`.")

    parser.add_argument("--wandb_args", type=str,
                        default="",
                        help=("Extra arguments passed to `wandb sync`. "
                              "Example: --wandb_args \"--entity my_team\""))

    parser.add_argument("--verbose", 
                        action="store_true",
                        help="Print full `wandb sync` output for successful syncs.")

    args = parser.parse_args()
    try:
        raise SystemExit(main(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)