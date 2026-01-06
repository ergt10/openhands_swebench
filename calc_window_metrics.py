#!/usr/bin/env python3
"""
Compute task/min and step/min for a time window using minisweagent logs and timings files.
- Counts completed instances by "Saved trajectory" lines in the log during the window.
- Counts served steps by summing steps whose query start_ts falls in the window across all timings.json files.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable

LOG_TIMEFMT = "%Y-%m-%d %H:%M:%S,%f"

def parse_args():
    p = argparse.ArgumentParser(description="Compute metrics over a time window")
    p.add_argument("start", help="Window start time (e.g. '2026-01-05 15:34:00')")
    p.add_argument("end", help="Window end time (e.g. '2026-01-05 16:00:00')")
    p.add_argument("--log", default=None, help="Path to minisweagent log; defaults to <data-dir>/minisweagent.log")
    p.add_argument("--data-dir", default="continuum_32", help="Directory containing instance folders")
    return p.parse_args()


def parse_time(s: str) -> dt.datetime:
    """
    Parse time allowing single-digit hours; normalize to zero-padded ISO string first.
    """
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        pass

    # Normalize "YYYY-MM-DD H:MM:SS[.fff]" -> zero-padded hour
    if " " in s:
        date, time = s.split(" ", 1)
        parts = time.split(":")
        if len(parts) >= 2:
            h = parts[0].zfill(2)
            norm = f"{date} {h}:{':'.join(parts[1:])}"
            try:
                return dt.datetime.fromisoformat(norm)
            except ValueError:
                pass
    raise ValueError(f"Invalid time format: {s}. Try zero-padding hour, e.g. 03:43:00")


def parse_log_completed(log_path: Path, data_dir: Path, start: dt.datetime, end: dt.datetime) -> set[str]:
    completed = set()
    token = f"{data_dir.name}/"
    with log_path.open() as f:
        for line in f:
            if len(line) < 24 or line[4] != "-":
                continue
            try:
                ts = dt.datetime.strptime(line[:23], LOG_TIMEFMT)
            except ValueError:
                continue
            if not (start <= ts <= end):
                continue
            if "Saved trajectory to" not in line:
                continue
            # Expected format: ...Saved trajectory to '.../<data_dir>/<id>/<id>.traj.json'
            parts = line.split(token)
            if len(parts) < 2:
                continue
            tail = parts[1]
            inst = tail.split("/")[0]
            if inst:
                completed.add(inst)
    return completed


def count_steps(data_dir: Path, start: dt.datetime, end: dt.datetime) -> int:
    step_count = 0
    for timings_path in data_dir.rglob("*.timings.json"):
        try:
            data = json.loads(timings_path.read_text())
        except Exception:
            continue
        for step in data.get("steps", []):
            ts = step.get("query", {}).get("start_ts")
            if ts is None:
                continue
            step_time = dt.datetime.fromtimestamp(ts)
            if start <= step_time <= end:
                step_count += 1
    return step_count


def main():
    args = parse_args()
    start = parse_time(args.start)
    end = parse_time(args.end)
    if start >= end:
        raise SystemExit("start must be before end")

    data_dir = Path(args.data_dir)
    log_path = Path(args.log) if args.log else data_dir / "minisweagent.log"

    completed = parse_log_completed(log_path, data_dir, start, end)
    steps = count_steps(data_dir, start, end)

    minutes = (end - start).total_seconds() / 60.0
    tasks_per_min = len(completed) / minutes if minutes else 0.0
    steps_per_min = steps / minutes if minutes else 0.0

    print(f"Window: {start} -> {end} ({minutes:.2f} min)")
    print(f"Completed instances: {len(completed)}")
    if completed:
        print(f"  IDs: {', '.join(sorted(completed))}")
    print(f"Total steps served (by query start_ts): {steps}")
    print(f"Tasks/min: {tasks_per_min:.3f}")
    print(f"Steps/min: {steps_per_min:.3f}")


if __name__ == "__main__":
    main()
