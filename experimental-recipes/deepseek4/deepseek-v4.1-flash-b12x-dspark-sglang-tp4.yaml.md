# DeepSeek V4.1 Flash, native SGLang launch (experimental)

`deepseek-v4.1-flash-b12x-dspark-sglang-tp4.yaml` is a candidate for four DGX
Sparks, one visible GPU per node, TP4/DP1/PP1/EP1. It declares
`min_sparkrun_version: "0.4.0"` for namespaced interpolation in `env` and the
`SPARKRUN_*` hook environment. Earlier releases cannot run it correctly and
versions predating the minimum-version feature cannot enforce that declaration. The preparation mod is vendored beside the recipe in
`dsv41-native-prepare/`; the simple-name reference works for both a file path and
a registry-loaded recipe. Keep that adjacent directory when copying the recipe.
No Little Cedar registry subscription is required. Normal transfer modes,
including `--transfer-mode delegated`, are supported; delegated transfer stages
the adjacent mod to the head before distribution.

The recipe retains Little Cedar's exact image digest, adapter settings, memory
fraction 0.80, 9.4M-token KV cap, DSpark block size 5, and 1M context. The available
KV budget can still clamp that cap. Fabric device names remain sparkrun-detected.
The adapter's default-disabled fast loader and `expandable_segments:False`
preserve the source policy.
The image adapters also read chunk, graph, and DSpark sizing from environment;
those values are retained alongside the matching engine flags and validated.
The source's hardware measurements do not validate this native-launch candidate.

## Launch and cache behavior

The recipe pins the [checkpoint commit](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/fb2764a5cf321eaa5070ca8f9e892818f477c16d)
used by the pinned upstream `boot.py`: `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.
Sparkrun prepares that exact snapshot and renders its in-container path into
`DSV41_SOURCE`. It renders adapter settings, port and runtime-cache paths before
creating each container. The hook obtains rank and revision from `SPARKRUN_*`. Preparation, serving, and the image's Docker healthcheck therefore
inherit those values. There is no generated environment file or checkpoint
symlink. There is no `command:` block: sparkrun generates `sglang serve` from
`defaults` and appends native rank and rendezvous arguments. The engine receives the
original model ID and the expanded `defaults.revision: "{model_revision}"`, so
Hugging Face loads the same pinned commit as distribution and the adapter's
`DSV41_SOURCE`. No shell environment handoff or duplicate revision is needed.
`model_type: llm` selects the modern CLI's LLM backend without auto-detecting a
different model revision. The public model name defaults to the original ID;
no repeated `served_model_name` is needed. `PYTHONUNBUFFERED=1` replaces Python's
previous `-u` option. Docker's current 32 GB shared-memory default is inherited.
`entrypoint: ""` remains necessary: the image's Dockerfile declares
`ENTRYPOINT ["python3", "-u", "/opt/dsv41/boot.py"]`, which must be cleared.

The [pinned SGLang CLI source](https://github.com/sgl-project/sglang/blob/f80c91a4b97e52e17fbcff088aa6b47b6270f6b9/python/sglang/cli/serve.py)
dispatches the LLM backend to the same `sglang.launch_server.run_server` entry.
The [model arguments](https://github.com/sgl-project/sglang/blob/f80c91a4b97e52e17fbcff088aa6b47b6270f6b9/python/sglang/srt/arg_groups/fields/model.py)
support exact commit revisions. Actual installed console-script/image behavior
remains part of the hardware trial.

The preparation mod owns topology/context checks; there is no duplicate
recipe `pre_exec` test block. It checks CUDA visibility, cache availability,
required image components, and Engram tensor metadata. The two Engram layers
(1 and 14) must use the expected FP8 weight/scale format. Invalid configuration or
an incompatible checkpoint fails before serving. The helper runs Python with
`-S` to avoid activating import-time image adapters during preparation; serving
keeps the image's normal Python startup and `PYTHONPATH`.

Each node uses its own persistent runtime-cache mount. **Shared means the hook,
packer and engine on a node see the same backing directory; the four nodes do
not share a network cache.** Kernels continue using sparkrun's managed compiler
cache variables. The extra b12x roots are derived from that same mount; b12x and
b12x_next remain separate.

Engram artifacts live under `engram/v1/<resolved-model-commit>/`. An identity
receipt binds them to the image digest, packer source hash, checkpoint
config/index hashes, TP4 and row counts. Existing shards must have the exact
4096-byte format, rank row range and byte length. Unidentified, corrupt or
mismatched artifacts cause an explicit failure: the helper never deletes or
rewrites shards that another process may use. Preserve that directory and choose
a separate runtime cache when resolving such a failure. Old unversioned shards
from the source launcher are intentionally not adopted without proof of identity.
Revision separation needs disk headroom for both versions; no automatic cleanup
is performed.

Missing shards start the image's original `pack_engram.py` with explicit
`--model`, `--rank`, `--tp 4` and `--out`, detached with low CPU/I/O priority.
A nonblocking node-local file lock spans preparation and that child. The child
runs as the runtime-cache owner, with redirected input/output and a persistent
`state/engram-pack-<revision>-r<rank>.log`. It belongs to the container, so stopping
the container also stops packing. The original packer's atomic `.partial` to
`.bin` publication is retained. Missing tools, space, or a pack failure leave the
exact checkpoint reader available; a cold launch can serve without packed shards
and use completed shards on a later boot. The hook does not wait for packing or
other ranks.

## Environment audit

The recipe has one `env` mapping, with 58 active entries (15 interpolated).
The 40 pinned upstream adapter/launcher/overlay files were checked for consumers;
the remaining engine, Python and NCCL settings retain their runtime meaning.
See [the audit](ENVIRONMENT.md) for the retained settings and removed duplicates.

Four command-construction-only variables were removed: `MEM_FRACTION_STATIC`,
`MAX_RUNNING_REQUESTS`, `MAX_TOTAL_TOKENS`, and `SPEC_ALGO`. Their settings now
exist only in defaults. The adapter still reads chunk size, graph batch size and
DSpark block size, so those three env values remain tied to defaults. TP and node
count are also active adapter inputs. EP/DP/PP/context env values enforce the
preparation mod's supported profile.

The mod reads rank and pinned revision from the hook environment and uses
`DSV41_SOURCE` directly, removing `NODE_RANK`, `MODEL_PATH` and
`DSV41_CHECKPOINT_REVISION`. Default-valued offload mode, statistics interval and
fast-loader switch were removed; the hook still rejects unsupported overrides.
The b12x_next cache derives from `STATE_PATH`; its separate explicit env entry is
unnecessary. Python warning suppression and the LL128 buffer setting (LL128 is
excluded by `NCCL_PROTO`) were also removed. NCCL debug settings remain as comments
for optional debugging.

## Acceptance differences and required hardware trial

This recipe bypasses both the registry shim and `boot.py run`. It consequently
**does not run boot.py's arithmetic, JSON-schema, tool-call and vision smoke
checks, nor its prompt/batch warmup**, and does not reproduce that supervisor's
smoke-failure engine termination, launch receipt, API-key-file handling or log
redaction. Existing generic post-hooks do not guarantee that acceptance policy
across CLI/API/benchmark paths, so this candidate makes no acceptance claim.
Sparkrun owns process lifecycle and runs port, HTTP-health, and streaming
inference readiness checks (inference readiness is enabled by default). The inference probe confirms
that a request yields output; it is not the upstream arithmetic/schema/tool/vision
acceptance suite or its warmup. A failed probe is reported as a readiness
failure. The image can still use `boot.py health` independently, with the
creation-time `SERVER_PORT` available.

The readiness budgets are deliberate but provisional: `port_timeout_s: 7200`
(two hours) and `health_timeout_s: 3600` (one hour) are inherited from Little
Cedar. Sparkrun defaults are 1800 and 900 seconds (30 and 15 minutes). These are
maximum waits for separate stages, not fixed delays; success advances immediately.
No native-launch startup measurement currently establishes that the larger values
are necessary. Record cold and warm startup times during the hardware trial,
including first-run compilation/loading and readiness inference. Prefer removing
this block and inheriting the defaults if they provide sufficient headroom;
otherwise reduce only the stage budgets that still need an override. Streaming
inference keeps its default 120-second budget.

Before promoting this recipe, run cold/warm four-node trials using the pinned
image and a recorded model commit. Verify rank-specific packing, final engine
argv/env and cache reuse, nondefault port healthchecks, startup/stop behavior,
all four smoke checks, warmup, representative inference and performance against
the source lane. Confirm adapter behavior against the actual image digest: the
registry documents a local healthcheck patch beyond the inspected upstream SHA.
No GPU launch, image pull, or benchmark was performed for this change.

Offline checks (from this directory):

```sh
python3 -B -m unittest discover -s tests -v
bash -n dsv41-native-prepare/run.sh
sparkrun recipe validate deepseek-v4.1-flash-b12x-dspark-sglang-tp4.yaml
```

The tests include failing controls for unsupported topology/cache settings,
changed checkpoint identity, missing/truncated checkpoint shards, invalid packed
headers, and native argument/environment propagation. They do not replace the
hardware trial.

## Attribution and licenses

Adapted from [Little Cedar Group's source recipe](https://github.com/littlecedar/sparkrun-recipe-registry/blob/fdd1003ba2a1e766a0129d26ef769ae4025c8c30/recipes/ds4/deepseek-v4.1-flash-knapcio-tp4-1m-sglang.yaml)
at commit `fdd1003ba2a1e766a0129d26ef769ae4025c8c30`. That source and its accompanying
records provide the upstream attribution chain.

The registry's [MIT license](../../LICENSE)
is the recipe-level license. Vendored upstream mod files retain their original
notices under [the mod directory](dsv41-native-prepare/README.md),
with exact source revision and hashes in
[PROVENANCE.json](dsv41-native-prepare/PROVENANCE.json).
Image components and model weights retain their upstream terms. Source
authors are credited, not represented as maintainers or endorsers of this
experimental adaptation.
