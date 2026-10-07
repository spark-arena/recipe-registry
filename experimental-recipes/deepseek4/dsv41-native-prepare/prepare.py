# Copyright (c) 2026 Little Cedar Group
"""Preparation only: validate this TP4 lane and optionally pack this node's rows.

Policy adapted from Little Cedar's dsv41-sglang-overlay and knapcio's pinned
DSV41 stack; see README.md and PROVENANCE.json. No server is launched and no environment is saved.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

IMAGE_DIGEST = "4e5002ab58b5cca670e2624c6a0d0799642914f04487ea130c71e7f525fa9c05"
PACKER = Path("/opt/dsv41/scripts/pack_engram.py")
MAGIC, HEADER_BYTES, ROW_BYTES = 0x31344E4531565344, 4096, 264
LAYERS = (1, 14)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def log(message: str) -> None:
    print("dsv41 preparation: " + message, flush=True)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_environment(env: dict[str, str]) -> tuple[Path, Path, int, str]:
    for key, expected in {
        "SPARKRUN_HOOK": "pre_exec",
        "SPARKRUN_RUNTIME": "sglang",
        "SPARKRUN_NUM_NODES": "4",
        "NNODES": "4",
        "TP_SIZE": "4",
        "DATA_PARALLEL": "1",
        "PIPELINE_PARALLEL": "1",
        "EP_SIZE": "1",
        "CHUNKED_PREFILL_SIZE": "4096",
        "CUDA_GRAPH_MAX_BS_DECODE": "16",
        "DSPARK_BLOCK_SIZE": "5",
        "SPARKRUN_RUNTIME_CACHE_ENABLED": "1",
    }.items():
        require(
            env.get(key) == expected,
            f"{key} must be {expected!r}, got {env.get(key)!r}",
        )
    require(env.get("OFFLOAD_MODE", "nvme") == "nvme", "OFFLOAD_MODE must be nvme")
    context = int(env.get("CONTEXT_LENGTH", "0"))
    require(4096 <= context <= 1048576, "CONTEXT_LENGTH must be in [4096,1048576]")
    rank = int(env.get("SPARKRUN_NODE_RANK", "-1"))
    require(0 <= rank < 4, "SPARKRUN_NODE_RANK must be in [0,4)")
    revision = env.get("SPARKRUN_MODEL_REVISION", "")
    require(
        bool(re.fullmatch(r"[0-9a-f]{40}", revision)),
        "a concrete HF checkpoint commit is required",
    )
    root_text = env.get("SPARKRUN_RUNTIME_CACHE_DIR", "")
    require(
        bool(root_text) and Path(root_text).is_absolute(),
        "missing absolute runtime-cache mount",
    )
    cache = Path(root_text).resolve()
    require(cache.is_dir() and cache != Path("/"), "runtime-cache mount must exist")
    model_text = env.get("DSV41_SOURCE", "")
    require(
        bool(model_text) and Path(model_text).is_absolute(),
        "DSV41_SOURCE must be a resolved container path",
    )
    model = Path(model_text).resolve()
    require(model.is_dir(), f"checkpoint is missing: {model}")
    expected_dirs = {
        "DSV41_PACKED_DIR": cache / "engram" / "v1" / revision,
        "STATE_PATH": cache / "state",
        "B12X_COMPILE_CACHE_DIR": cache / "b12x-compile",
        "B12X_ROCE_CACHE_DIR": cache / "b12x-roce",
        "B12X_NEXT_COMPILE_CACHE_DIR": cache / "state" / "b12x-next-compile",
    }
    for key, expected in expected_dirs.items():
        # The adapter derives this cache from STATE_PATH when no override is set.
        default = str(expected) if key == "B12X_NEXT_COMPILE_CACHE_DIR" else ""
        actual = Path(env.get(key, default)).resolve()
        require(
            actual == expected and actual.is_relative_to(cache),
            f"{key} must be {expected}",
        )
    require(
        env.get("SERVER_PORT") == env.get("SPARKRUN_PORT"),
        "SERVER_PORT must match the serving port",
    )
    require(
        env.get("DSV41_FAST_LOAD", "0") == "0",
        "this experimental baseline requires DSV41_FAST_LOAD=0",
    )
    require(
        env.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:False",
        "expandable CUDA segments are unsupported",
    )
    return cache, model, rank, revision


def checkpoint_rows(model: Path) -> dict[int, int]:
    """Validate only metadata and byte bounds, without reading 203 GB of rows."""
    config = json.loads((model / "config.json").read_text())
    require(isinstance(config, dict), "config.json must be a JSON object")
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    rows = {}
    for layer in LAYERS:
        prefix = f"layers.{layer}.engram.embed."
        name = index[prefix + "weight"]
        require(
            name == index[prefix + "scale"],
            "Engram weight and scale must share a shard",
        )
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "invalid checkpoint shard path",
        )
        shard = model / relative
        # HF snapshot symlinks to blobs are expected; no checkpoint file is modified.
        with shard.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            require(0 < size <= 16 * 1024 * 1024, "invalid safetensors header length")
            header = json.loads(handle.read(size))
        row_count = None
        for suffix, dtype, width in (
            ("weight", "F8_E4M3", 256),
            ("scale", "F8_E8M0", 8),
        ):
            tensor = header[prefix + suffix]
            shape = tensor["shape"]
            require(
                tensor["dtype"] == dtype and len(shape) == 2 and shape[1] == width,
                "unsupported Engram tensor format",
            )
            count = shape[0]
            require(isinstance(count, int) and count > 0, "invalid Engram row count")
            require(
                row_count is None or row_count == count,
                "Engram weight/scale row counts differ",
            )
            row_count = count
            start, stop = tensor["data_offsets"]
            require(
                0 <= start < stop and stop - start == count * width,
                "invalid Engram tensor offsets",
            )
            require(8 + size + stop <= shard.stat().st_size, "checkpoint shard is truncated")
        rows[layer] = row_count
    return rows


def shard_complete(path: Path, layer: int, rows: int, rank: int) -> bool:
    if not path.exists():
        return False
    require(not path.is_symlink(), f"packed shard cannot be a symlink: {path}")
    lo, hi = rows * rank // 4, rows * (rank + 1) // 4
    expected = (MAGIC, layer, lo, hi, rows, ROW_BYTES)
    with path.open("rb") as handle:
        header = handle.read(48)
    require(
        len(header) == 48 and struct.unpack("<6Q", header) == expected,
        f"invalid packed header at {path}; isolate this cache before retrying",
    )
    require(
        path.stat().st_size == HEADER_BYTES + (hi - lo) * ROW_BYTES,
        f"truncated packed shard at {path}; isolate this cache before retrying",
    )
    return True


def validate_image() -> None:
    for path in (
        Path("/opt/dsv41/adapter/sitecustomize.py"),
        Path("/opt/dsv41/adapter/librow_store.so"),
        Path("/sgl-workspace/sglang/python/sglang/srt/layers/engram.py"),
    ):
        require(path.is_file(), f"required image component missing: {path}")
    require(
        "EngramLoader" in Path("/opt/dsv41/adapter/sitecustomize.py").read_text(),
        "wrong adapter overlay",
    )
    require(Path("/opt/b12x_next/b12x_next").is_dir(), "b12x_next kernels are missing")
    # CUDA visibility, rather than the physical nvidia-smi inventory, determines TP placement.
    probe_env = dict(os.environ)
    probe_env.pop("DSV41_SOURCE", None)
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        env=probe_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    require(
        result.stdout.strip().splitlines()[-1:] == ["1"],
        "this recipe requires exactly one visible CUDA GPU per node",
    )


def reown(path: Path, owner: tuple[int, int]) -> None:
    if os.geteuid() == 0:
        os.chown(path, *owner)


def prepare(env: dict[str, str]) -> None:
    cache, model, rank, revision = validate_environment(env)
    validate_image()
    rows = checkpoint_rows(model)
    owner = (cache.stat().st_uid, cache.stat().st_gid)
    for key in (
        "STATE_PATH",
        "B12X_COMPILE_CACHE_DIR",
        "B12X_ROCE_CACHE_DIR",
        "B12X_NEXT_COMPILE_CACHE_DIR",
        "DSV41_PACKED_DIR",
    ):
        path = Path(env[key]) if key in env else cache / "state" / "b12x-next-compile"
        path.mkdir(parents=True, exist_ok=True)
        for ancestor in (path, *path.parents):
            if ancestor == cache:
                break
            reown(ancestor, owner)
    packed = Path(env["DSV41_PACKED_DIR"])
    # This lock spans metadata validation AND the detached child. Never block other nodes.
    lock_path = packed / ".pack.lock"
    with lock_path.open("a+b") as lock:
        reown(lock_path, owner)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # A previous validated packer owns the same immutable revision namespace.
            validate_receipt(packed, model, revision, rows)
            log(f"rank {rank}: pack already active; using available shards or checkpoint fallback")
            return
        validate_receipt(packed, model, revision, rows, create=True, owner=owner)
        missing = [layer for layer in LAYERS if not shard_complete(packed / f"engram-l{layer}-r{rank}of4.bin", layer, rows[layer], rank)]
        if not missing:
            log(f"rank {rank}: packed Engram shards validated at {packed}")
            return
        if not PACKER.is_file():
            log("image packer absent; checkpoint fallback remains available")
            return
        log_path = Path(env["STATE_PATH"]) / f"engram-pack-{revision}-r{rank}.log"
        child_env = dict(env)
        child_env.pop("DSV41_SOURCE", None)  # packer receives --model, needs no adapters
        child_env["HOME"] = str(Path(env["STATE_PATH"]))
        credentials = {"user": owner[0], "group": owner[1], "extra_groups": []} if os.geteuid() == 0 else {}
        try:
            with log_path.open("ab", buffering=0) as output:
                reown(log_path, owner)
                subprocess.Popen(
                    [
                        sys.executable,
                        "-S",
                        str(Path(__file__).resolve()),
                        "--pack-worker",
                        str(model),
                        str(packed),
                        str(rank),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    start_new_session=True,
                    pass_fds=(lock.fileno(),),
                    cwd="/opt/dsv41",
                    **credentials,
                )
            log(f"rank {rank}: packing layers {missing} in background; log {log_path}; checkpoint fallback available")
        except OSError as exc:
            log(f"could not spawn optional packer ({exc}); using checkpoint fallback")


def validate_receipt(
    packed: Path,
    model: Path,
    revision: str,
    rows: dict[int, int],
    *,
    create: bool = False,
    owner: tuple[int, int] = (0, 0),
) -> None:
    receipt = {
        "format": "DSV41EN1",
        "tp": 4,
        "revision": revision,
        "image_sha256": IMAGE_DIGEST,
        "config_sha256": digest(model / "config.json"),
        "index_sha256": digest(model / "model.safetensors.index.json"),
        "packer_sha256": digest(PACKER) if PACKER.is_file() else None,
        "rows": {str(k): v for k, v in rows.items()},
    }
    path = packed / "identity.json"
    if path.exists():
        require(
            json.loads(path.read_text()) == receipt,
            f"packed-cache identity mismatch at {path}; use a separate cache",
        )
    else:
        require(
            create and not any(packed.glob("*.bin")) and not any(packed.glob("*.partial")),
            f"unidentified packed artifacts at {packed}; preserve them and use a separate cache",
        )
        temp = packed / ".identity.tmp"
        temp.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        reown(temp, owner)
        temp.replace(path)


def pack_worker(model: str, packed: str, rank: str) -> int:
    """Hold the inherited lock until the original image packer exits."""
    command = [
        "nice",
        "-n",
        "10",
        "ionice",
        "-c",
        "3",
        sys.executable,
        str(PACKER),
        "--model",
        model,
        "--rank",
        rank,
        "--tp",
        "4",
        "--out",
        packed,
    ]
    try:
        result = subprocess.run(command, check=False)
        if result.returncode:
            log(f"optional packer failed with exit {result.returncode}; checkpoint fallback remains available")
        return result.returncode
    except OSError as exc:
        log(f"optional packer unavailable ({exc}); checkpoint fallback remains available")
        return 1


if __name__ == "__main__":
    try:
        if len(sys.argv) == 5 and sys.argv[1] == "--pack-worker":
            raise SystemExit(pack_worker(*sys.argv[2:]))
        require(len(sys.argv) == 1, "unexpected preparation arguments")
        prepare(dict(os.environ))
    except (
        RuntimeError,
        OSError,
        ValueError,
        KeyError,
        struct.error,
        subprocess.SubprocessError,
    ) as error:
        print(f"dsv41 preparation FATAL: {error}", file=sys.stderr)
        raise SystemExit(1)
