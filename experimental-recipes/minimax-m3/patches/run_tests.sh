#!/bin/bash
# Run the _llm_nom test suite inside the sglang container, with the local
# patched source mounted over the in-image source.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
docker run --rm \
  -v "$HERE/_llm_nom":/usr/local/lib/python3.12/dist-packages/sglang/srt/function_call/_llm_nom \
  -v "$HERE/tests":/tests \
  --entrypoint bash \
  scitrera/dgx-spark-sglang-mm:v0 \
  -c 'cd /tests && python3 -m pytest -v "$@" 2>&1 | grep -vE "UserWarning: Triton|warnings.warn"' -- "$@"
