#!/usr/bin/env python3
"""sparkrun -> boot.py launcher shim for the knapcio DSV41 SGLang stack.

The `dsv41-4x-spark:canary-roce` image is built from knapcio's
DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4 repo. Its normal entrypoint is
`python3 -u /opt/dsv41/boot.py run`, which reads every serve parameter from
environment variables (MODEL_PATH, NNODES, NODE_RANK, DIST_INIT_ADDR, ...) and
builds the full `sglang.launch_server` argv itself.

sparkrun's SGLang runtime appends the per-node rendezvous flags
`--dist-init-addr HOST:PORT --nnodes N --node-rank R` to the recipe's serve
command (it assumes the command is `sglang serve ...`). This shim consumes those
flags, maps them onto boot.py's environment, and execs boot.py unchanged -- so
the recipe stays a normal sparkrun recipe and the vendored overlay is untouched.

Usage (rendered by the recipe's `command:` template, with sparkrun's node args
appended):

    python3 /workspace/mods/dsv41-sglang-overlay/launcher.py --boot --model M \
        --port 8888 --tp 4 --served-model-name M --context-length 1048576 \
        [--dist-init-addr H:P] [--nnodes N] [--node-rank R]

`--health` forwards to boot.py's health probe (used for a manual check).

The recipe's `--context-length {max_model_len}` maps onto boot.py's
CONTEXT_LENGTH. This is load-bearing: `boot.py` builds the whole
`sglang.launch_server` argv from the environment, so a `--context-length` it
never parses would leave CONTEXT_LENGTH at its default and the server would
advertise the wrong context. CONTEXT_LENGTH is applied AFTER `apply_recipe_env`
so the recipe's value wins over the launcher's own default (see main()).

Engram shards. The overlay serves the two 203 GB Engram tables from node-local
packed shards, one per rank, under `$DSV41_PACKED_DIR` (the recipe points this at
`/cache/runtime/engram`, the sparkrun-managed runtime cache). Those shards are a
cacheable artifact -- ~47 GiB/rank, derived from the checkpoint -- and must not
live in a user's home directory. They are produced by the image's
`/opt/dsv41/scripts/pack_engram.py`. This shim materializes *this rank's* shards
before booting:

  * rank from `--node-rank` (sparkrun passes it to the head and every worker),
  * if both shards already exist -> nothing to do,
  * otherwise launch the packer **detached** (flock-guarded, one per node) and
    boot immediately. boot.py then reads Engram straight from the checkpoint for
    that boot; the packed shards are picked up on the next boot.

The pack is deliberately not synchronous: a cold pack is minutes long, and
sparkrun only waits ~150 s for the head to open its rendezvous port before it
declares the launch dead. Backgrounding keeps every boot inside that budget;
correctness is unaffected because the checkpoint fallback is exact (see
`adapter/engram_backend.py`: the packed shard is an optional acceleration, and
`row_store.cpp` fails closed if a shard does not match the rank's row range).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

BOOT = "/opt/dsv41/boot.py"
PACKER = "/opt/dsv41/scripts/pack_engram.py"
ENGRAM_LAYERS = (1, 14)
# The HF cache is bind-mounted at a fixed path inside the container (sparkrun's
# orchestration/primitives.py), so the launcher can resolve the checkpoint here
# instead of the recipe spelling a snapshot hash in env:.
HF_HUB = "/cache/huggingface/hub"

# ---------------------------------------------------------------------------
# Recipe-owned environment.
#
# The recipe's `env:` block is deliberately thin: it carries only the external
# checkpoint path and the TP degree. Everything else the engine needs is the
# production configuration for this stack, and it lives here so the recipe does
# not have to spell engine internals (and so the values travel with the mod
# rather than with each recipe that uses it). These OVERRIDE (not setdefault):
# the image bakes several of them into its ENV (STATE_PATH=/state,
# DSV41_CACHE_GIB=16, PYTORCH_CUDA_ALLOC_CONF=...True, SGLANG_RUST_BUILD_MODE,
# OFFLOAD_MODE), and those live in the launcher's process env, so a setdefault
# would let the image value win and silently undo the production config
# (unwritable /state, the NaN-logits allocator mode). The keys here are disjoint
# from the recipe-owned MODEL_PATH / DSV41_SOURCE / TP_SIZE, so those pass through.
#
# REDUNDANT vars (recipe value == runtime default) are NOT here and are NOT in
# the recipe: SKIP_SMOKE(0), WARMUP(1), HOST(0.0.0.0), SERVED_MODEL_NAME,
# SPARK_PREFILL_TP_MIN_CONTEXT(32768), SPARK_PREFILL_TP_MIN_ROWS(1024),
# DSV41_FAST_LOAD_TP_SLICE(auto), DSV41_FAST_LOAD_INFLIGHT_GB(6).
#
# The ONE deliberate exception to that rule is DSV41_FAST_LOAD("0") below: it is
# also the runtime default, but it is stated to record a decision, not a no-op --
# upstream ships it on, and "0" is the production choice here. Do not "clean it
# up" as redundant. (Its N_EXPERTS companion is gone: it only served the fast
# path and defaults from config.json's n_routed_experts when unset.)
#
# NCCL_NET / NCCL_IB_HCA / NCCL_IB_GID_INDEX / NCCL_IB_DISABLE / NCCL_CROSS_NIC
# and B12X_ROCE_HCA are also absent on purpose: sparkrun's InfiniBand probe fills
# them per cluster, and b12x falls back to the detected NCCL_IB_HCA. Naming a
# host device here would break portability.
# ---------------------------------------------------------------------------
_CACHE = "/cache/runtime"
# Rendezvous for the image's baked HEALTHCHECK: a fixed path the health probe
# reads its port from. A literal (not $STATE_PATH) because the image bakes
# STATE_PATH=/state while the launcher relocates it under /cache/runtime.
HEALTH_PORT_FILE = "/tmp/sparkrun_health_port"
RECIPE_ENV: dict[str, str] = {
    # --- locations (all under the sparkrun-managed runtime cache) ----------
    "STATE_PATH": f"{_CACHE}/state",
    "DSV41_PACKED_DIR": f"{_CACHE}/engram",
    "B12X_ROCE_CACHE_DIR": f"{_CACHE}/b12x-roce",
    "B12X_COMPILE_CACHE_DIR": f"{_CACHE}/b12x-compile",
    # --- boot.py -----------------------------------------------------------
    "SKIP_PREPARE": "1",       # checkpoint distributed to this node by sparkrun
    "SKIP_VERIFY": "1",
    "READY_TIMEOUT_S": "3600",
    "OFFLOAD_MODE": "nvme",    # Engram on NVMe (host = the GPU's unified memory)
    # --- Engram reader (adapter/engram_backend.py, row_store.cpp) ----------
    "DSV41_CACHE_GIB": "4",
    "DSV41_CACHE_WAYS": "16",
    "DSV41_IO_THREADS": "96",
    "DSV41_RESIDENT_SCALES": "0",
    "DSV41_STATS_SECONDS": "60",
    "DSV41_ENGRAM_PREFETCH": "1",
    # --- parallelism / serving (boot.py reads these) -----------------------
    "EP_SIZE": "1",            # b12x_next production line; EP>1 dies at decode
    "CONTEXT_LENGTH": "1048576",
    "MEM_FRACTION_STATIC": "0.80",
    "MAX_RUNNING_REQUESTS": "16",
    "CHUNKED_PREFILL_SIZE": "4096",
    "CUDA_GRAPH_MAX_BS_DECODE": "16",
    # KV pool pin, measured on this fleet 2026-10-06 (recipes/ds4/AGENTS.md 7.1):
    # at MEM_FRACTION_STATIC=0.80 the engine's own budget line reports
    # full_token=9,493,504..9,589,504 (bytes_per_full_token=1670.75, ~15.9 GB
    # available_bytes); a pin ABOVE that is CLAMPED to the budget, not refused. 9.4M
    # is ~98% of the budget and was verified to boot and serve on all four nodes.
    # The previous 4000000 left ~58% of the ceiling unused.
    # READ full_token from the boot log after any change to weights, TP size or NCCL
    # buffers -- it moves with the head's MemAvailable (upstream saw 6.71-7.82M
    # across their stack; ours is a stock-loader EP1 pair at 0.80).
    "MAX_TOTAL_TOKENS": "9400000",
    "SPEC_ALGO": "DSPARK",
    "DSPARK_BLOCK_SIZE": "5",
    # --- overlay adapters (the upstream production line) -------------------
    "DSV41_INDEXER_CHUNKED": "1",
    "SGLANG_DSPARK_FOLDED_SAMPLING": "2",
    "SGLANG_RUST_BUILD_MODE": "never",
    "SPARK_PREFILL_TP_SPLIT": "1",
    # Fast weight loading is OFF BY CHOICE. adapter/fast_load.py cuts engine
    # start from 343 s to ~124 s, but SGLang sizes the KV pool from the head's
    # MemAvailable right after the loads and the eager pinned reads leave
    # 0.8-1.5 GB less of it visible then, so the pool comes out 3-13 % smaller
    # (upstream's EP2 measurement; see recipes/ds4/AGENTS.md 7.1 and
    # docs/fast-load.md in the image's source). We prefer the pool and pay
    # ~220 s per boot. The image does not set this var, so "0" is the stock
    # loader; it is explicit to record the decision, and "1" is the rollback.
    # Evidence labels: the 3-13 % and the 343 s -> 124 s are upstream's
    # (VERIFIED in its docs); the EP1 delta on this lane is UNMEASURED.
    "DSV41_FAST_LOAD": "0",
    "DSV41_SHARED_PAD_K": "1",
    "DSV41_WO_A_W8": "1",
    "DSV41_WO_A_W8_MID": "1",
    "DSV41_WO_A_W8_DROP": "1",
    "DSV41_DRAFT_TAU": "0.7",
    "DSV41_DRAFT_HEAD_FP8": "1",
    "DSV41_BLOCK_VERIFY": "1",
    "DSV41_FOLDED_FENCE": "1",
    "DSV41_VERIFY_CAP": "conf:0.1",
    "DSV41_AUTOTUNE_KEEP": "1",
    "DSV41_REPLICATED_SPLIT": "wqkv_a,engram.wkv",
    "DSV41_DRAFT_MAIN_PROJ_SPLIT": "1",
    "DSV41_SPLIT_COMPACT_GATHER": "1",
    "DSV41_ROUTER_LIVE": "1",
    "DSV41_MOE_B12X_NEXT": "1",
    "DSV41_MOE_B12X_NEXT_DETERMINISTIC": "1",
    "DSV41_HC_FUSED": "1",
    "DSV41_PREFILL_SP": "1",
    "DSV41_PREFILL_SP_FP8": "1",
    "DSV41_L2_PREFETCH": "1",
    "DSV41_L2_PREFETCH_WOA": "1",
    "DSV41_SPEC_SYNC_FREE": "all",
    "DSV41_EAGER_GLUE": "all",
    # NOTE: DSPARK_SPS_TABLE / DSPARK_STS_TABLE are deliberately NOT set --
    # upstream lists the SPS table under "Not used: crashes the Engram path".
    # --- RoCEnante + NCCL transport (device names come from detection) -----
    "SGLANG_ROCE_ALLREDUCE": "1",
    "SGLANG_ROCE_MAX_SIZE": "2097152",
    "DSV41_ROCE_GATHER": "2097152",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_BUFFSIZE": "1048576",
    "NCCL_LL128_BUFFSIZE": "262144",
    "NCCL_PROTO": "^LL128",
    "NCCL_MAX_NCHANNELS": "8",
    "NCCL_DEBUG": "WARN",
    "NCCL_DEBUG_SUBSYS": "INIT,ENV",
    # expandable_segments False: the V4.1 SGLang lane reports NaN logits above 64
    # prefill query tokens with it on (ds4 AGENTS.md §6.4). boot.py's default too.
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    # The image's torch is 2.13.0+cu130, which renamed the collectives
    # all_gather_into_tensor -> all_gather_single and reduce_scatter_tensor ->
    # reduce_scatter_single (PyTorch 2.13 release; the old names remain as
    # aliases behind a FutureWarning). The overlay and the engine still call the
    # old names, so every such call logs, via the _exception_logger wrapper at
    # torch/distributed/c10d_logger.py:83:
    #   FutureWarning: `torch.distributed.all_gather_into_tensor` is deprecated.
    #   Please use `torch.distributed.all_gather_single` instead.
    # (and the reduce_scatter twin). It is a benign once-per-callsite notice
    # about a rename, not a deprecation of behaviour we rely on, and we cannot
    # patch it out -- the callers are in the image's sglang tree and its vendored
    # kernels. Silence ONLY the c10d_logger re-emission of these two torch
    # FutureWarnings. Module-scoped and empirically verified against the pinned
    # image (a bare `message='is deprecated'` filter does NOT take: warnings
    # filterwarnings() anchors the message with re.match, so the leading
    # backtick defeats it; a module filter is robust to the message rewording).
    # `::` with an empty message = any message. Other FutureWarnings, and the
    # c10d debug logger, are untouched. Rollback is deleting this line.
    "PYTHONWARNINGS": ("ignore::FutureWarning:torch.distributed.c10d_logger"),
}


def apply_recipe_env(env: dict[str, str]) -> None:
    """Apply the production config, overriding the image's baked ENV.

    Override, not setdefault: the image bakes several of these (STATE_PATH,
    DSV41_CACHE_GIB, PYTORCH_CUDA_ALLOC_CONF, ...) and those values are in the
    launcher's process env, so setdefault would let the image win. The keys here
    never include the recipe-owned MODEL_PATH / DSV41_SOURCE / TP_SIZE.
    """
    env.update(RECIPE_ENV)


def _pack_env(env: dict[str, str]) -> dict[str, str]:
    """Environment for the background packer (inherit, but pin HOME)."""
    return dict(env, HOME=env.get("HOME") or "/tmp")


def ensure_engram_shards(env: dict[str, str], rank: int | None) -> None:
    """Materialize this rank's packed Engram shards, detached if absent.

    Never raises: a shard it cannot build costs the checkpoint fallback (correct,
    slower), while an exception here would cost the launch.
    """
    packed_dir = env.get("DSV41_PACKED_DIR", "").strip()
    if not packed_dir:
        return
    if rank is None:
        print("launcher: no --node-rank/NODE_RANK; skipping Engram pack "
              "(boot.py will read Engram from the checkpoint)", flush=True)
        return
    try:
        tp = int(env.get("TP_SIZE", "4"))
    except ValueError:
        tp = 4

    missing = [
        layer for layer in ENGRAM_LAYERS
        if not _shard_complete(os.path.join(packed_dir, "engram-l%d-r%dof%d.bin" % (layer, rank, tp)))
    ]
    if not missing:
        print("launcher: Engram shards present for rank %d/%d at %s" % (rank, tp, packed_dir), flush=True)
        return
    if not os.path.isfile(PACKER):
        print("launcher: packer %s missing; booting without packed Engram" % PACKER, flush=True)
        return

    os.makedirs(packed_dir, exist_ok=True)
    lock_path = os.path.join(packed_dir, ".pack.lock")
    log_path = os.path.join(env.get("STATE_PATH", packed_dir), "engram-pack.log")
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log = open(log_path, "ab", buffering=0)
    except OSError:
        log = subprocess.DEVNULL

    # flock -n: if another pack is already running on this node, leave it alone.
    # nice/ionice: the pack is background work; keep it behind the cold boot's
    # weight load (both compete for the same NVMe and the unified memory pool).
    cmd = [
        "flock", "-n", lock_path,
        "nice", "-n", "10", "ionice", "-c", "3",
        sys.executable, PACKER,
        "--rank", str(rank), "--tp", str(tp), "--out", packed_dir,
    ]
    try:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_pack_env(env),
            cwd="/opt/dsv41",
        )
        print("launcher: Engram shards missing for rank %d/%d (layers %s); "
              "packing in the background -> %s (log: %s). Booting on the "
              "checkpoint fallback for this boot." % (rank, tp, missing, packed_dir, log_path), flush=True)
    except OSError as exc:
        print("launcher: could not start Engram packer (%r); booting on the "
              "checkpoint fallback" % (exc,), flush=True)


def _shard_complete(path: str) -> bool:
    """Cheap completeness test: the packer renames the finished file into place,
    so a non-empty target with no `.partial` sibling is the done state."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
        return not os.path.exists(path.replace(".bin", ".partial"))
    except OSError:
        return False


def _repo_cache_dir(model: str) -> str:
    """`deepseek-ai/DeepSeek-V4.1-Flash` -> its dir under the HF hub cache."""
    return os.path.join(HF_HUB, "models--" + model.replace("/", "--"))


def _resolved_models_dir(model: str) -> str | None:
    """The local:local dir to serve *model* from, or ``None`` if not resolvable.

    The recipe names the model (``{model}``) and the launcher locates it in the
    fixed in-container HF cache, so no snapshot hash is spelled in `env:` (where
    it would read as a path the user must maintain). Robust to the cache
    re-resolving: prefer `refs/main`, else any snapshot carrying both `config.json`
    and `model.safetensors.index.json`. ``None`` means "not found", and the caller
    leaves MODEL_PATH unset so the engine's own error is the one seen.
    """
    import glob

    repo = _repo_cache_dir(model)
    if not os.path.isdir(repo):
        return None

    candidates: list[str] = []
    try:
        with open(os.path.join(repo, "refs", "main")) as handle:
            revision = handle.read().strip()
        if revision:
            candidates.append(os.path.join(repo, "snapshots", revision))
    except OSError:
        pass
    candidates += sorted(glob.glob(os.path.join(repo, "snapshots", "*")), reverse=True)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(os.path.join(candidate, "config.json")) and \
           os.path.isfile(os.path.join(candidate, "model.safetensors.index.json")):
            return candidate
    return None


def resolve_checkpoint(env: dict[str, str]) -> None:
    """Set MODEL_PATH / DSV41_SOURCE from the model id in the fixed HF cache.

    Runs before any checkpoint validation. If the caller supplied a valid path
    (a container bind that is not the HF hub cache — read-only, but boot.py's
    SKIP_PREPARE only needs it to exist), it is left alone. Otherwise the path is
    derived from `MODEL_ID` (sparkrun's `{model}`) and the hub cache, so the
    recipe never spells a snapshot hash.
    """
    model = env.get("MODEL_ID", "").strip()
    supplied = env.get("MODEL_PATH", "").strip()
    if supplied and os.path.isfile(os.path.join(supplied, "config.json")):
        env.setdefault("DSV41_SOURCE", supplied)
        return
    if not model:
        return
    resolved = _resolved_models_dir(model)
    if resolved:
        print(f"launcher: resolved checkpoint {model} -> {resolved}", flush=True)
        env["MODEL_PATH"] = resolved
        env["DSV41_SOURCE"] = resolved
    else:
        print(f"launcher: checkpoint for {model} not found under {HF_HUB}; "
              "leaving MODEL_PATH unset", flush=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--boot", action="store_true", help="start the engine (default)")
    parser.add_argument("--health", action="store_true", help="run boot.py health probe")
    parser.add_argument("--dist-init-addr", default=None)
    parser.add_argument("--nnodes", default=None)
    parser.add_argument("--node-rank", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--tp", default=None,
                        help="TP degree (sparkrun's {tensor_parallel}); mapped to "
                             "boot.py's TP_SIZE")
    parser.add_argument("--context-length", default=None,
                        help="served context length (sparkrun's {max_model_len}); "
                             "mapped to boot.py's CONTEXT_LENGTH")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model", default=None,
                        help="HF repo id (sparkrun's {model}); the checkpoint is "
                             "located in the fixed HF hub cache from this")
    # Everything else (unused sglang flags, the recipe's own args) is ignored.
    args, _unknown = parser.parse_known_args(argv)

    env = dict(os.environ)
    if args.dist_init_addr:
        env["DIST_INIT_ADDR"] = args.dist_init_addr
    if args.nnodes:
        env["NNODES"] = args.nnodes
    if args.node_rank is not None:
        env["NODE_RANK"] = args.node_rank
    if args.port:
        env["SERVER_PORT"] = args.port
    if args.tp:
        # boot.py reads TP_SIZE from the env; the recipe passes the TP degree as
        # `--tp {tensor_parallel}` (sparkrun renders the template but does NOT
        # emit --tp-size when one is present, so this mapping is load-bearing --
        # without it boot.py would fall back to its default TP=3).
        env["TP_SIZE"] = args.tp
    if args.served_model_name:
        env["SERVED_MODEL_NAME"] = args.served_model_name
    if args.model_path:
        env["MODEL_PATH"] = args.model_path
    if args.model:
        env["MODEL_ID"] = args.model

    if not os.path.isfile(BOOT):
        print(f"launcher: boot.py not found at {BOOT}; is the image dsv41-4x-spark:canary-roce?", file=sys.stderr)
        return 2

    # Fill the production env (locations, engine flags, adapter enablement,
    # RoCEnante, NCCL tuning). The launcher owns these and overrides the image.
    apply_recipe_env(env)

    # `--context-length {max_model_len}` is the recipe's context knob. boot.py
    # builds the whole `sglang.launch_server` argv from the environment and
    # refuses CONTEXT_LENGTH outside 4096..1048576, so an unparsed flag would
    # leave the launcher's default in place and the server would advertise the
    # wrong context. Applied AFTER apply_recipe_env so the recipe's value wins
    # over the launcher's own default (e.g. `-o max_model_len=…`).
    if args.context_length is not None:
        env["CONTEXT_LENGTH"] = args.context_length

    # Locate the checkpoint in the fixed HF cache (MODEL_ID from `{model}`).
    # After apply_recipe_env, which is where /cache/runtime paths are set.
    resolve_checkpoint(env)

    if args.health:
        os.execve(sys.executable, [sys.executable, BOOT, "health"], env)

    rank = env.get("NODE_RANK")
    ensure_engram_shards(env, int(rank) if rank is not None and str(rank).lstrip("-").isdigit() else None)

    # Belt-and-suspenders for the image's baked HEALTHCHECK: persist the served
    # port to the fixed path boot.py's dynamic health probe reads. The healthcheck
    # runs from the container's *creation* env, so it cannot see SERVER_PORT that
    # this launcher sets in-process -- but it CAN read this file. The path is a
    # literal (not $STATE_PATH) because the image bakes STATE_PATH=/state while the
    # launcher relocates it under /cache/runtime; the health probe only needs a
    # fixed rendezvous. boot.py's own --port/cmdline/proc-net discovery would also
    # find the port; this just makes the answer deterministic from the first probe.
    if env.get("SERVER_PORT"):
        try:
            with open(HEALTH_PORT_FILE, "w") as handle:
                handle.write(str(env["SERVER_PORT"]) + "\n")
        except OSError as exc:
            print(f"launcher: could not write {HEALTH_PORT_FILE} ({exc!r}); "
                  "health probe will discover the port from the process", flush=True)

    print(
        "launcher: boot.py run "
        f"NNODES={env.get('NNODES')} NODE_RANK={env.get('NODE_RANK')} "
        f"DIST_INIT_ADDR={env.get('DIST_INIT_ADDR')} TP={env.get('TP_SIZE')} "
        f"EP={env.get('EP_SIZE')} PORT={env.get('SERVER_PORT')} "
        f"CONTEXT={env.get('CONTEXT_LENGTH')}",
        flush=True,
    )
    os.execve(sys.executable, [sys.executable, "-u", BOOT, "run"], env)
    return 1  # unreachable


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
