#!/bin/bash
# Copyright (c) 2026 Little Cedar Group
set -euo pipefail
# Adapted preparation entrypoint; upstream/littlecedar is reference-only.
# -S avoids sitecustomize/adapters during preparation. They remain enabled for serving.
exec python3 -S "$(dirname "${BASH_SOURCE[0]}")/prepare.py"
