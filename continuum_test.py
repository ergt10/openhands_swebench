#!/usr/bin/env python3
"""Run swe_bench eval sequentially with multiple worker counts."""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict

WORKER_COUNTS = [24]
VLLM_PORT = 8100
RUN_TIMEOUT_SECONDS = 70 * 60 
METRIC_NAMES = [
    'vllm:prefix_cache_hits_total',
    'vllm:prefix_cache_queries_total',
    'vllm:request_decode_time_seconds_sum',
    'vllm:request_prefill_time_seconds_sum',
    'vllm:request_decode_time_seconds_count',
]


def kill_process_tree(proc: subprocess.Popen, *, grace_seconds: int = 10) -> None:
    """Best-effort: terminate a process and all of its children."""
    if proc.poll() is not None:
        return

    if os.name != 'posix':
        proc.kill()
        return

    pgid = proc.pid

    def process_group_alive() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def wait_process_group_exit(timeout_seconds: float) -> bool:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not process_group_alive():
                return True
            time.sleep(0.2)
        return not process_group_alive()

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if wait_process_group_exit(grace_seconds):
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        return

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    wait_process_group_exit(grace_seconds)
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_with_timeout(cmd: list[str], *, env: dict[str, str], timeout_seconds: int) -> None:
    proc = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=(os.name == 'posix'),
    )
    try:
        returncode = proc.wait(timeout=timeout_seconds)
    except KeyboardInterrupt:
        kill_process_tree(proc)
        raise
    except subprocess.TimeoutExpired as e:
        kill_process_tree(proc)
        raise e

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    base_env = os.environ.copy()
    base_env['EVAL_SKIP_MAXIMUM_RETRIES_EXCEEDED'] = 'true'
    try:
        for workers in WORKER_COUNTS:
            # 写入仓库根目录，避免落在 OpenHands 子目录影响源码哈希
            output_dir = repo_root / f'Continuum_{workers}'
            before_metrics = fetch_vllm_metrics()

            cmd = [
                sys.executable,
                '-m',
                'evaluation.benchmarks.swe_bench.run_infer',
                '--config-file',
                str(repo_root / 'OpenHands' / 'config.toml'),
                '--llm-config',
                'vllm_local',
                '--agent-cls',
                'CodeActAgent',
                '--dataset',
                'princeton-nlp/SWE-bench_Lite',
                '--split',
                'test',
                '--max-iterations',
                '50',
                '--eval-num-workers',
                str(workers),
                '--eval-output-dir',
                str(output_dir),
            ]
            print(f'Running eval with {workers} workers -> {output_dir}')
            timed_out = False
            interrupted = False
            try:
                run_with_timeout(cmd, env=base_env, timeout_seconds=RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                print(
                    f'Run with {workers} workers exceeded {RUN_TIMEOUT_SECONDS}s, killing and cleaning containers...'
                )
            except KeyboardInterrupt:
                interrupted = True
                print(
                    'Interrupted (Ctrl+C). Collecting metrics and cleaning containers...'
                )

            after_metrics: Dict[str, float] = {}
            diff: Dict[str, float] = {}
            computed: Dict[str, float | None] = {}
            metrics_error: str | None = None
            try:
                after_metrics = fetch_vllm_metrics()
                diff, computed = summarize_metrics(before_metrics, after_metrics)
            except Exception as e:
                metrics_error = str(e)
                # Still write a JSON artifact for bookkeeping/debugging.
                after_metrics = {}
                diff = {}
                computed = {}

            metrics_path = output_dir / 'vllm_metrics.json'
            with metrics_path.open('w') as f:
                json.dump(
                    {
                        'workers': workers,
                        'timed_out': timed_out,
                        'interrupted': interrupted,
                        'run_timeout_seconds': RUN_TIMEOUT_SECONDS,
                        'before': before_metrics,
                        'after': after_metrics,
                        'diff': diff,
                        'computed': computed,
                        'metrics_error': metrics_error,
                    },
                    f,
                    indent=2,
                )
            print(f'Wrote metrics to {metrics_path}')
            if timed_out:
                cleanup_openhands_containers()
                cleanup_openhands_runtime_images()
                continue
            if interrupted:
                cleanup_openhands_containers()
                cleanup_openhands_runtime_images()
                return 130
    except KeyboardInterrupt:
        print('Interrupted (Ctrl+C). Cleaning containers...')
        cleanup_openhands_containers()
        cleanup_openhands_runtime_images()
        return 130
    return 0


def cleanup_openhands_containers() -> None:
    cmds = [
        "docker ps --format '{{.Names}}' | grep '^openhands-' | xargs -r docker stop",
        "docker ps -a --format '{{.ID}} {{.Names}}' | grep 'openhands-runtime-' | awk '{print $1}' | xargs -r docker rm",
    ]
    for cmd in cmds:
        subprocess.run(['bash', '-lc', cmd], check=False)


def cleanup_openhands_runtime_images() -> None:
    cmd = "docker images ghcr.io/openhands/runtime -q | sort -u | xargs -r docker rmi -f"
    subprocess.run(['bash', '-lc', cmd], check=False)


def fetch_vllm_metrics() -> Dict[str, float]:
    url = f'http://127.0.0.1:{VLLM_PORT}/metrics'
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode('utf-8')
    except Exception as e:
        raise RuntimeError(f'Failed to fetch vLLM metrics from {url}: {e}') from e

    values: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        first = line.split()[0]
        base_name = first.split('{', 1)[0]
        if base_name not in METRIC_NAMES:
            continue
        try:
            values[base_name] = float(line.rsplit(None, 1)[-1])
        except ValueError:
            continue
    # Ensure all expected metrics exist (default to 0.0 if missing)
    for name in METRIC_NAMES:
        values.setdefault(name, 0.0)
    return values


def summarize_metrics(before: Dict[str, float], after: Dict[str, float]):
    diff = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in METRIC_NAMES}

    queries = diff['vllm:prefix_cache_queries_total']
    hits = diff['vllm:prefix_cache_hits_total']
    decode_sum = diff['vllm:request_decode_time_seconds_sum']
    prefill_sum = diff['vllm:request_prefill_time_seconds_sum']
    decode_count = diff['vllm:request_decode_time_seconds_count']

    hit_rate = hits / queries if queries > 0 else None
    decode_time_per_req = decode_sum / decode_count if decode_count > 0 else None
    prefill_time_per_req = prefill_sum / decode_count if decode_count > 0 else None

    computed = {
        'prefix_cache_hit_rate': hit_rate,
        'decode_time_per_request': decode_time_per_req,
        'prefill_time_per_request': prefill_time_per_req,
    }
    return diff, computed


if __name__ == '__main__':
    raise SystemExit(main())
