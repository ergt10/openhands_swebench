#!/usr/bin/env python3
"""
Pre-pull (and optionally pre-build runtime images for) all instances used by
SWE-bench Lite test split so the eval run doesn't spend time downloading/building.

Usage:
    python prefetch_swebench_lite_images.py \
        --dataset princeton-nlp/SWE-bench_Lite \
        --split test \
        --platform linux/amd64 \
        --build-runtime   # optional: also build OH runtime images for each base
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
import concurrent.futures
import threading

import docker
from datasets import load_dataset


# Allow imports from the OpenHands repo checked out in ./OpenHands
REPO_ROOT = Path(__file__).resolve().parent
OPENHANDS_DIR = REPO_ROOT / 'OpenHands'
if str(OPENHANDS_DIR) not in sys.path:
    sys.path.insert(0, str(OPENHANDS_DIR))

# Import after sys.path update
from evaluation.benchmarks.swe_bench.run_infer import (  # noqa: E402
    get_instance_docker_image,
    set_dataset_type,
)
from openhands.runtime.builder import DockerRuntimeBuilder  # noqa: E402
from openhands.runtime.utils.runtime_build import (  # noqa: E402
    build_runtime_image,
)


def iter_unique_base_images(dataset: str, split: str) -> list[str]:
    """Load the dataset and return unique base images needed for the split."""
    set_dataset_type(dataset)
    ds = load_dataset(dataset, split=split)
    images: set[str] = set()
    for row in ds:
        instance_id = row['instance_id']
        images.add(
            get_instance_docker_image(
                instance_id=instance_id, swebench_official_image=True
            )
        )
    return sorted(images)


def docker_pull(image: str, platform: str | None) -> None:
    cmd = ['docker', 'pull']
    if platform:
        cmd += ['--platform', platform]
    cmd.append(image)
    subprocess.run(cmd, check=True)


def build_runtime(image: str, platform: str | None, force_rebuild: bool) -> str:
    client = docker.from_env()
    builder = DockerRuntimeBuilder(client)
    # Browsing is disabled in run_infer; mirror that for consistency.
    return build_runtime_image(
        base_image=image,
        runtime_builder=builder,
        platform=platform,
        enable_browser=False,
        force_rebuild=force_rebuild,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Pre-pull SWE-bench Lite images and optionally build runtime images.'
    )
    parser.add_argument(
        '--dataset',
        default='princeton-nlp/SWE-bench_Lite',
        help='Dataset name to pull images for (default: %(default)s)',
    )
    parser.add_argument(
        '--split', default='test', help='Dataset split to use (default: %(default)s)'
    )
    parser.add_argument(
        '--platform',
        default='linux/amd64',
        help='Platform to use for docker pull/build (default: %(default)s)',
    )
    parser.add_argument(
        '--build-runtime',
        action='store_true',
        help='Also build OpenHands runtime images after pulling base images.',
    )
    parser.add_argument(
        '--force-rebuild',
        action='store_true',
        help='Force rebuild runtime images (passed to build_runtime_image).',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only list images; do not pull/build.',
    )
    parser.add_argument(
        '--build-workers',
        type=int,
        default=20,
        help='Concurrent runtime build workers (default: %(default)s)',
    )
    args = parser.parse_args()

    images = iter_unique_base_images(args.dataset, args.split)
    print(f'Found {len(images)} base images for {args.dataset}[{args.split}].')

    if args.dry_run:
        for img in images:
            print(img)
        return 0

    MAX_RETRIES = 5
    SLEEP_SECS = 10
    print_lock = threading.Lock()

    def log(msg: str) -> None:
        with print_lock:
            print(msg, flush=True)

    def build_with_retry(image: str, idx: int, total: int) -> None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                runtime_image = build_runtime(
                    image, args.platform, args.force_rebuild
                )
                log(f'[{idx}/{total}] Built runtime image: {runtime_image}')
                return
            except Exception as e:
                if attempt == MAX_RETRIES:
                    log(
                        f'[{idx}/{total}] Build failed after {MAX_RETRIES} attempts: {e}'
                    )
                    raise
                log(
                    f'[{idx}/{total}] Build failed (attempt {attempt}/{MAX_RETRIES}): {e}. '
                    f'Retrying in {SLEEP_SECS}s...'
                )
                time.sleep(SLEEP_SECS)

    if args.build_runtime:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.build_workers
        ) as executor:
            futures: list[concurrent.futures.Future[None]] = []
            for idx, image in enumerate(images, start=1):
                log(f'[{idx}/{len(images)}] Pulling base image: {image}')
                docker_pull(image, args.platform)
                futures.append(
                    executor.submit(build_with_retry, image, idx, len(images))
                )
            for f in concurrent.futures.as_completed(futures):
                f.result()
    else:
        for idx, image in enumerate(images, start=1):
            log(f'[{idx}/{len(images)}] Pulling base image: {image}')
            docker_pull(image, args.platform)

    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
