#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path


TIME_KEYS = ("_start", "_end")


def parse_duration(value: str) -> float:
    """Parse duration like 30min, 2h, 1800s into seconds (float)."""
    raw = value.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]*)", raw)
    if not match:
        raise ValueError(f"Invalid duration: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return amount
    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return amount * 60.0
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return amount * 3600.0
    if unit in {"d", "day", "days"}:
        return amount * 86400.0
    raise ValueError(f"Unknown duration unit: {unit!r}")


def iter_timing_files(root: Path):
    return sorted(root.rglob("timing_*.json"))


def find_global_min_timestamp(files):
    min_ts = None
    min_meta = None
    for path in files:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key, value in obj.items():
                    if not isinstance(value, (int, float)):
                        continue
                    if key.endswith(TIME_KEYS) or key == "llm_first_chunk":
                        if min_ts is None or value < min_ts:
                            min_ts = float(value)
                            min_meta = (path, obj.get("instance_id"), key)
    return min_ts, min_meta


def count_llm_end_in_window(files, window_start, window_end):
    count = 0
    for path in files:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                llm_end = obj.get("llm_end")
                if isinstance(llm_end, (int, float)) and window_start <= llm_end <= window_end:
                    count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Count LLM calls whose llm_end falls within a time window "
            "starting from the earliest timestamp across all timing files."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Target working directory to search (e.g. TRnew_120).",
    )
    parser.add_argument(
        "--window",
        required=True,
        help="Time window length (e.g. 30min, 2h, 1800s).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    files = iter_timing_files(root)
    if not files:
        raise SystemExit(f"No timing_*.json found under {root}")

    window_seconds = parse_duration(args.window)
    min_ts, min_meta = find_global_min_timestamp(files)
    if min_ts is None:
        raise SystemExit(f"No timestamps found in timing files under {root}")

    window_end = min_ts + window_seconds
    count = count_llm_end_in_window(files, min_ts, window_end)

    min_dt = datetime.fromtimestamp(min_ts)
    end_dt = datetime.fromtimestamp(window_end)
    origin_info = (
        f"{min_meta[0].name} (instance={min_meta[1]}, key={min_meta[2]})"
        if min_meta
        else "unknown"
    )

    print(f"root: {root}")
    print(f"timing_files: {len(files)}")
    print(f"window_seconds: {window_seconds}")
    print(f"window_start: {min_ts} ({min_dt})")
    print(f"window_end:   {window_end} ({end_dt})")
    print(f"window_origin: {origin_info}")
    print(f"llm_end_count: {count}")


if __name__ == "__main__":
    main()
