# DeepSeek V4.1 native-launch preparation

This mod adapts Little Cedar Group's `dsv41-sglang-overlay` for native sparkrun
SGLang launches. It prepares the container and optional packed Engram artifacts;
it never launches or supervises the serving engine. The recipe supplies resolved
model/cache paths and rank context through interpolated `env` values and `SPARKRUN_*` hook metadata, and calls SGLang
directly. No hook-produced environment file or checkpoint symlink is needed.

## Vendored source and attribution

[`upstream/littlecedar/`](upstream/littlecedar/) contains **byte-for-byte copies**
from [Little Cedar's registry](https://github.com/littlecedar/sparkrun-recipe-registry/tree/fdd1003ba2a1e766a0129d26ef769ae4025c8c30/mods/dsv41-sglang-overlay)
at commit `fdd1003ba2a1e766a0129d26ef769ae4025c8c30`:

- `run.sh`: original compatibility/cache preparation hook, including its original
  notices.
- `launcher.py`: original policy/packing/`boot.py` launcher, retained for source
  provenance and comparison.
- `README.md`: original explanation and attribution, including its license notice
  and hardware observations about the original lane.

[`PROVENANCE.json`](PROVENANCE.json) records each source path, git blob ID and
SHA-256, and identifies the active adapted files separately. These vendored files
are **reference-only**: neither `upstream/littlecedar/run.sh` nor
`upstream/littlecedar/launcher.py` is invoked or imported by the active mod.
Original source observations are not validation of this adaptation.

The active top-level `run.sh` and `prepare.py` adapt Little Cedar's compatibility,
cache-ownership and detached rank-local packing responsibilities. This
adaptation adds resolved hook/environment inputs, topology checks, immutable
revision namespaces, cache receipts and packed-header validation, and removes
all serving-launch logic. The original `RECIPE_ENV` policy moves to the recipe's
ordinary environment/templates; upstream `boot.py run` is bypassed.

Little Cedar's image and launcher build on [knapcio's TP4 profile](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/58f232155917d388b6053eca079c617fd306c33c)
of [Mia / MiaAI-Lab's DeepSeek deployment](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks).
The image's Engram packer remains `/opt/dsv41/scripts/pack_engram.py`; no kernel,
model weights or image sources are copied into this mod. The pack format is
verified against knapcio's pinned packer. See the upstream profile's
[NOTICE](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/blob/58f232155917d388b6053eca079c617fd306c33c/NOTICE)
for its original 0xSero MIT skeleton and SGLang/b12x/sparkring component credits.

Vendored files retain their exact original notices; see
[Little Cedar's repository](https://github.com/littlecedar/sparkrun-recipe-registry/tree/fdd1003ba2a1e766a0129d26ef769ae4025c8c30)
for upstream license terms. Image components and model weights retain their own
licenses.

## Inputs and execution

`run.sh` executes only top-level `prepare.py`, using Python `-S` to avoid importing
image adapters during preparation. It requires a `pre_exec` SGLang hook on
exactly four nodes, rank 0–3, `TP_SIZE=4`, `EP_SIZE=1`, `DATA_PARALLEL=1`,
`PIPELINE_PARALLEL=1`, `NNODES=4`, and `CONTEXT_LENGTH` in 4096–1048576. GPU probing
checks exactly one visible CUDA device. Keep the serving command's context value
consistent with `CONTEXT_LENGTH` when authoring or modifying a recipe.

`DSV41_SOURCE` must name an existing container checkpoint. The hook-provided
`SPARKRUN_MODEL_REVISION` must be a concrete 40-character commit matching the
recipe's pin. The helper checks the
Engram tensors in the safetensors metadata without reading the entire model.
`SPARKRUN_RUNTIME_CACHE_ENABLED=1` and the resolved `SPARKRUN_RUNTIME_CACHE_DIR`
identify the actual persistent cache. Derived environment paths must match:

| Variable | Path relative to that cache |
| --- | --- |
| `DSV41_PACKED_DIR` | `engram/v1/<resolved-model-commit>` |
| `STATE_PATH` | `state` |
| `B12X_COMPILE_CACHE_DIR` | `b12x-compile` |
| `B12X_ROCE_CACHE_DIR` | `b12x-roce` |
| `B12X_NEXT_COMPILE_CACHE_DIR` (optional; adapter derives it from `STATE_PATH`) | `state/b12x-next-compile` |

The serving process and hooks share this backing cache **on each node**. Each
node retains independent model and runtime caches; there is no new network share.
The helper validates fixed source-policy settings so adapter inputs cannot
silently diverge from the recipe's corresponding engine arguments.

Packed-cache receipts bind the checkpoint commit, config/index hashes, image
and packer identity, TP degree and row counts. Existing packed files need matching
headers, row ranges and byte lengths; unidentified or incompatible artifacts fail
explicitly and are never deleted automatically. Revision namespacing isolates new
checkpoints from older running engines and requires disk headroom for both.

Missing shards start the image's original packer with explicit model, rank, TP and
output path. It runs detached, as the runtime-cache owner, with low CPU/I/O
priority and redirected stdio. A nonblocking file lock spans validation and the
child; logs stay in `STATE_PATH`. It remains a child inside the container, so
container teardown stops it. Optional packer failures preserve the checkpoint
reader fallback. There is no inter-node wait or synchronous full-model packing.

## Use and validation

The mod is adjacent to the recipe. Use `mods: [dsv41-native-prepare]` for both
file-based and registry-based recipes. Adjacent lookup wins before registry mod
directories. Delegated transfer stages the locally resolved adjacent directory
to the head; publishing a separately registered mod is unnecessary.

Offline tests live in `experimental-recipes/deepseek4/tests`. They exercise
metadata, ownership/spawn settings, cache identities, locks, negative controls,
and the commandless native recipe profile. Core runtime tests cover generated
SGLang arguments and shell environment propagation. No hardware launch, image pull, acceptance
request or benchmark was performed for this adaptation. Removing `boot.py run`
also removes its smoke checks, warmup and failure supervision; the experimental
recipe documentation describes the hardware acceptance work still required.
