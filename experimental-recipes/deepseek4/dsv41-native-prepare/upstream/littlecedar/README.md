# dsv41-sglang-overlay

Compatibility gate and launcher shim for the **knapcio DSV41 SGLang image**
(vendored as `littlecedar/dgx-spark-dsv41:canary-roce`, natively
`dsv41-4x-spark:canary-roce`), used by
`recipes/ds4/deepseek-v4.1-flash-knapcio-tp4-1m-sglang.yaml`.

| | |
|:--|:--|
| Kind | Pre-exec mod; **modifies no image file**. |
| License | AGPL-3.0-or-later (this repo). |
| Verified | Booted live on 4 Sparks (`.32`–`.35`) 2026-10-02; ~12 healthy boots. Those boots ran the fast loader, which has since been turned **off** in the launcher (`DSV41_FAST_LOAD=0`, below); the gated boot is otherwise unchanged. Portability rewrite (no host mounts; `/cache/runtime` Engram) booted 2026-10-02. See `recipes/ds4/AGENTS.md`. |

The image is **vendored**: `littlecedar/dgx-spark-dsv41:canary-roce` on Docker Hub
(digest-pinned), so sparkrun pulls it and no node builds it. sparkrun distributes
the image by default (its `distribution_config` default is `containers.enabled:
true`), so the recipe carries no `distribution_config:` block.

## Why it exists

The image is built from
[`knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4`](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4)
via its `Dockerfile.canary-roce`. It bakes the full adapter overlay at
`/opt/dsv41/adapter` (on `PYTHONPATH` via the image `ENV`), the `b12x` /
`b12x_next` kernels, and `/opt/dsv41/boot.py`. Its entrypoint is
`python3 -u /opt/dsv41/boot.py run`, and **boot.py builds the entire
`sglang.launch_server` argv from environment variables**.

sparkrun's SGLang runtime instead assumes the command is `sglang serve …` and
appends the per-node rendezvous flags `--dist-init-addr HOST:PORT --nnodes N
--node-rank R`. `launcher.py` bridges the two: it consumes those three flags,
maps them onto boot.py's `DIST_INIT_ADDR` / `NNODES` / `NODE_RANK`, and execs
`boot.py`. The recipe's `command:` is a one-line call to it; serve parameters
ride either in the command (`--model {model}`, `--tp {tensor_parallel}`,
`--context-length {max_model_len}`, `--port {port}`) or in the mod's
`RECIPE_ENV` (everything else).

## Mechanics (fail-closed)

`run.sh` is a gate, not a patch. It refuses to proceed unless the container
really is the overlay build:

- `/opt/dsv41/boot.py` exists,
- `/opt/dsv41/adapter/sitecustomize.py` exists and contains `EngramLoader`,
- `/opt/b12x_next/b12x_next` exists,
- the engine's `engram.py` exists.

It then logs the resolved paths and creates + `reown`s the cacheable-object
directories under the sparkrun-managed runtime cache (`/cache/runtime/engram`,
`/cache/runtime/state`, `/cache/runtime/b12x-compile`, `/cache/runtime/b12x-roce`)
so the non-root serve user can write them. Nothing in the image is modified —
the upstream repo's own in-image test suite gates the overlay instead.

## Portability (the whole point of the `/cache/runtime` wiring)

The recipe carries **no host bind mounts**. Everything that is not the model and
not image content — the two per-rank Engram shards, boot.py's `launch.json` /
`api-key`, and the b12x JIT caches — lives under sparkrun's managed runtime
cache: `<host> ~/.cache/sparkrun/runtime-cache/sglang/<model_dir>/`, mounted at
`/cache/runtime`. sparkrun creates and chowns that leaf per node; the recipe
never spells the host path and the container path is constant.

The two pieces that make this work:

- **`run.sh`** creates and `reown`s the subdirectories (it runs before the serve
  command), so the non-root user can write them.
- **`launcher.py`** is the only component that learns the node's rank (sparkrun
  appends `--node-rank` to the serve command on the head *and* every worker; the
  container env carries no rank). It materializes *this rank's*
  `engram-l{1,14}-r{rank}of{tp}.bin` under `/cache/runtime/engram`:
  - present → no-op;
  - missing → it starts the image's `/opt/dsv41/scripts/pack_engram.py`
    **detached** (`flock`-guarded, one per node) and boots immediately on the
    checkpoint Engram fallback. The packed shards are used from the next boot.
    Synchronicity would blow sparkrun's ~150 s head-rendezvous wait on a cold
    node; backgrounding keeps every boot inside budget and correctness is
    unaffected (`adapter/engram_backend.py` treats the packed shard as optional
    acceleration; `row_store.cpp` fails closed on a range mismatch).

It also **locates the checkpoint**. The recipe passes the repo id
(`--model {model}`); `launcher.py:resolve_checkpoint()` finds it in the fixed
in-container HF cache (`/cache/huggingface/hub/models--<org>--<name>`) and sets
`MODEL_PATH` / `DSV41_SOURCE` — preferring `refs/main`, else the newest snapshot
carrying both `config.json` and `model.safetensors.index.json`. So the recipe
spells **no snapshot path** (a hardcoded `snapshots/<hash>` is not portable and
reads as a path a user must maintain), and a cache that re-resolves to a
new hash cannot break the boot (each node's cache is independent now — there is
no shared mount). A container bind supplied as `MODEL_PATH` is left
alone (it is read-only, which is all the `SKIP_PREPARE` existence check needs).

It also **maps the recipe's command flags onto boot.py's environment**, for the
parameters boot.py reads from env rather than argv:

- `--tp {tensor_parallel}` → `TP_SIZE` (sparkrun does *not* emit `--tp-size` when
  a `command:` template is present, so without this boot.py would fall back to
  its default TP=3 on a 4-node cluster).
- `--context-length {max_model_len}` → `CONTEXT_LENGTH`, applied **after**
  `RECIPE_ENV` so the recipe's value (e.g. `-o max_model_len=…`) wins over the
  launcher default. Without the mapping the flag would be silently dropped and
  the server would advertise the launcher's default context.
- `--port {port}` → `SERVER_PORT` (the launcher's own process env; it also
  persists the port to `/tmp/sparkrun_health_port` for the image's dynamic
  healthcheck — see below).

## Docker's HEALTHCHECK and the served port

The image bakes `HEALTHCHECK CMD ["python3","-S","/opt/dsv41/boot.py","health"]`.
Originally `boot.py health` probed `http://127.0.0.1:$SERVER_PORT/health`,
defaulting to the image build port 8888 when `SERVER_PORT` was unset. The probe
reads the **container's creation env** (`docker inspect .Config.Env`), which
Docker freezes at `docker run` — so a value set at runtime by the launcher
(`os.execve`) or a mod (even writing `/proc/1/environ`) is *below Docker's view*
and could not influence it (VERIFIED live), and sparkrun never puts `SERVER_PORT`
there. A server on any other port therefore reported `Up … (unhealthy)`.

**The 2026-10-03 image rebuild (digest `4e5002ab…`) makes the healthcheck
discover the port dynamically.** `boot.py health` now resolves the probe port at
runtime: `HEALTH_PORT`/`SERVER_PORT` env → the port the launcher persists to
`/tmp/sparkrun_health_port` → the running engine's own `--port` (from
`/proc/<pid>/cmdline`) → any LISTEN port from `/proc/net/tcp{,6}` → the image
default. Any served port now probes correctly, and the recipe needs **no `env:`
block**.

`launcher.py` writes the rendezvous file as belt-and-suspenders. The path is a
literal (`/tmp`), not `$STATE_PATH`, because the image bakes `STATE_PATH=/state`
while the launcher relocates it under `/cache/runtime` — the health probe runs
from that baked env, so only a fixed path round-trips. Even without the file the
probe would find the engine's `--port`.

## PyTorch 2.13 collective-rename warning (silenced)

The image's torch is **2.13.0+cu130**. PyTorch 2.13 renamed the distributed
collectives `all_gather_into_tensor` → `all_gather_single` and
`reduce_scatter_tensor` → `reduce_scatter_single`, keeping the old names as
aliases behind a `FutureWarning`. The image's SGLang tree and its vendored
kernels still call the old names, so a boot logs, from the `_exception_logger`
wrapper at `torch/distributed/c10d_logger.py:83`:

```
FutureWarning: `torch.distributed.all_gather_into_tensor` is deprecated. Please use `torch.distributed.all_gather_single` instead.
```

(and the `reduce_scatter_tensor` twin; both seen in
`.scratch/ds4/knapcio/logs/boot9-head-serve.log`). It is a benign
once-per-callsite notice about a rename, not a change in behaviour we depend on,
and the callers are in the image, so it cannot be patched out here. `launcher.py`
silences it with a **module-scoped** filter in `RECIPE_ENV`:

```
PYTHONWARNINGS=ignore::FutureWarning:torch.distributed.c10d_logger
```

Only the c10d_logger re-emission of these two torch `FutureWarning`s is
suppressed; other warnings and the c10d debug logger are untouched. The filter is
module-scoped (not a blanket `ignore::FutureWarning`) and empirically verified
against the pinned image — a bare `message='is deprecated'` filter does **not**
take, because `warnings.filterwarnings()` anchors the message with `re.match` and
the leading backtick defeats it, whereas a module filter is robust to torch
rewording the message. Rollback is deleting the one line.

## The production env lives here, not in the recipe

`launcher.py`'s `RECIPE_ENV` carries the whole production configuration —
locations under `/cache/runtime`, the boot.py read flags, the Engram reader
tuning, the engine/serving flags, the overlay-adapter enablement, RoCEnante and
the device-free NCCL transport tuning (`mods/dsv41-sglang-overlay/launcher.py`,
inventoried in `.scratch/ds4/knapcio/ENV-MIGRATION.md`). It is applied with
`env.update(RECIPE_ENV)`, i.e. it **overrides** the launcher's process env —
necessary because the image bakes `STATE_PATH=/state`, `DSV41_CACHE_GIB=16`,
`PYTORCH_CUDA_ALLOC_CONF=…True`, `SGLANG_RUST_BUILD_MODE` and `OFFLOAD_MODE`, and
a `setdefault` would let those win (silently redirecting boot.py state to an
unwritable `/state`, and re-arming the allocator mode that NaNs above 64 prefill
query tokens). The TP degree and context length are *not* here: they arrive as
`--tp {tensor_parallel}` / `--context-length {max_model_len}` from the recipe's
`command:` and are mapped in `main()` (see above).

The recipe's own `env:` is therefore **empty** — the TP degree, context length
and port all arrive via the `command:` template from `defaults`, and the image's
dynamic healthcheck needs no container-level env (see above). `RECIPE_ENV` values
that the runtime already defaults to (SKIP_SMOKE,
WARMUP, HOST, SERVED_MODEL_NAME, the prefill thresholds, the fast-load
slice/inflight defaults) are **not** set anywhere — with one deliberate
exception: `DSV41_FAST_LOAD=0` is also the runtime default, but it is stated to
record a production decision (below), not as a no-op, so it is not to be removed
as redundant.

One production switch is deliberately **off**: `DSV41_FAST_LOAD=0` selects
SGLang's stock loader over upstream's eager loader. The eager path cuts engine
start from 343 s to ~124 s but leaves the KV pool 3–13% smaller (it sizes from the
head's `MemAvailable` right after the loads), and this lane prefers the pool.
Setting the key is how the decision is recorded — the image does not set it, and
the loader's own gate defaults false. Rationale, the upstream numbers, and the
boot-log A/B live in `recipes/ds4/AGENTS.md` §7.1.

`/cache/runtime` sits on node-local NVMe (`/dev/nvme0n1p2` here), so the packed
Engram reads never traverse the network — the same property the old
`/home/red/dsv41-engram` bind mount gave, without the machine-specific path.

## Not in this mod (and why)

- The image itself is **vendored** (`littlecedar/dgx-spark-dsv41:canary-roce`,
  digest-pinned) and pulled by sparkrun under its default distribution config
  (`containers.enabled: true`). To build locally instead, retag the result
  `dsv41-4x-spark:canary-roce` and add
  `distribution_config: { containers: { enabled: false } }` to the recipe — the
  recipe deliberately omits the block otherwise, so there is no dev-only config
  to drift.
- The checkpoint is **distributed to every node** by sparkrun (the HF cache is
  per-node, not shared — there is no NFS export), also under the default config
  (`models.enabled: true`). The image itself is vendored (pulled, not built).
