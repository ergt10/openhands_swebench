#!/usr/bin/env python3
"""
Run OpenHands SWE-bench inference with a hard timeout, then stop & remove all Docker containers.

Requested command:
  python -m evaluation.benchmarks.swe_bench.run_infer \
    --config-file OpenHands/config.toml \
    --llm-config vllm_local \
    --agent-cls CodeActAgent \
    --dataset princeton-nlp/SWE-bench_Lite \
    --split test \
    --max-iterations 50 \
    --eval-num-workers 144 \
    --eval-output-dir sglanggate_144_openhands \
    --num-rollouts 20

Timeout: 3.5 hours (12600 seconds).

Cleanup:
  docker stop $(docker ps -aq)
  docker rm $(docker ps -aq)
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


TIMEOUT_SECONDS = int(3.5 * 60 * 60)  # 3.5 hours


def _kill_process_tree(proc: subprocess.Popen, *, grace_seconds: int = 15) -> None:
    if proc.poll() is not None:
        return

    if os.name != "posix":
        proc.kill()
        return

    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run(cmd: list[str], *, check: bool = False) -> int:
    completed = subprocess.run(cmd, check=False)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return completed.returncode


def _docker_cleanup_all_containers() -> None:
    # Equivalent to:
    #   docker stop $(docker ps -aq)
    #   docker rm $(docker ps -aq)
    # but safe when there are no containers.
    _run(["bash", "-lc", "docker ps -aq | xargs -r docker stop"], check=False)
    _run(["bash", "-lc", "docker ps -aq | xargs -r docker rm"], check=False)


def main() -> int:
    repo_root = Path(__file__).resolve().parent

    cmd = [
        sys.executable,
        "-m",
        "evaluation.benchmarks.swe_bench.run_infer",
        "--config-file",
        str(repo_root / "OpenHands" / "config.toml"),
        "--llm-config",
        "vllm_local",
        "--agent-cls",
        "CodeActAgent",
        "--dataset",
        "princeton-nlp/SWE-bench_Lite",
        "--split",
        "test",
        "--max-iterations",
        "50",
        "--eval-num-workers",
        "144",
        "--eval-output-dir",
        "sglanggate_144_openhands",
        "--num-rollouts",
        "20",
    ]

    print(f"[run] timeout={TIMEOUT_SECONDS}s cmd={' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        start_new_session=(os.name == "posix"),
    )
    timed_out = False
    try:
        proc.wait(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        print(f"[timeout] exceeded {TIMEOUT_SECONDS}s, killing process tree...")
        _kill_process_tree(proc)
    except KeyboardInterrupt:
        print("[interrupt] Ctrl+C received, killing process tree...")
        _kill_process_tree(proc)
        raise
    finally:
        print("[cleanup] docker stop/rm all containers...")
        _docker_cleanup_all_containers()

    if timed_out:
        return 124
    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())

