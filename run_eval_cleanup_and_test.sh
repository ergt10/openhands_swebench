#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python run_swebench_eval_multi.py

docker images ghcr.io/openhands/runtime -q \
  | sort -u \
  | xargs -r docker rmi -f

python - <<'PY'
from pathlib import Path

path = Path("OpenHands/config.toml")
text = path.read_text()
old = "cleanup_runtime_image = false"
new = "cleanup_runtime_image = true"

if old in text:
    text = text.replace(old, new)
elif new in text:
    pass
else:
    raise SystemExit("cleanup_runtime_image setting not found")

path.write_text(text)
PY

python vllm_test.py
