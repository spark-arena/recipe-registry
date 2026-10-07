"""Offline artifact tests; no torch, GPU, Docker, or source checkout required."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "deepseek-v4.1-flash-b12x-dspark-sglang-tp4.yaml"
MOD = ROOT / "dsv41-native-prepare"
SPEC = importlib.util.spec_from_file_location("native_prepare", MOD / "prepare.py")
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)
REVISION = "a" * 40


def section(name):
    text = RECIPE.read_text().splitlines()
    start = next(i for i, line in enumerate(text) if line.startswith(name + ":"))
    lines = []
    for line in text[start + 1 :]:
        if line and not line.startswith((" ", "#")):
            break
        if line.startswith("  ") and not line.lstrip().startswith("#"):
            lines.append(line[2:])
    if not lines:
        raise AssertionError("missing recipe section " + name)
    return lines


def mapping(name):
    return {key: json.loads(value) for key, value in (line.split(": ", 1) for line in section(name))}


def defaults():
    """Read this recipe's simple scalar defaults without a YAML dependency."""
    result = {}
    for line in section("defaults"):
        key, value = line.split(": ", 1)
        if value.startswith("'") and value.endswith("'"):
            result[key] = value[1:-1].replace("''", "'")
        else:
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                result[key] = value
    return result


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "runtime cache"
        self.cache.mkdir()
        self.model = self.root / "snapshot"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"deepseek_v4"}')
        self.packer = self.root / "packer.py"
        self.packer.write_text("# test packer")
        self.write_checkpoint()
        launch = SimpleNamespace(
            model_path=str(self.model), model_revision=REVISION, runtime_cache_dir=str(self.cache), num_nodes=4, node_rank=2
        )
        rendered = {
            key: value.format(config=SimpleNamespace(**defaults()), launch=launch) for key, value in mapping("env").items()
        }
        self.env = {
            **mapping("env"),
            **rendered,
            "SPARKRUN_HOOK": "pre_exec",
            "SPARKRUN_RUNTIME": "sglang",
            "SPARKRUN_NUM_NODES": "4",
            "NNODES": "4",
            "TP_SIZE": "4",
            "EP_SIZE": "1",
            "SPARKRUN_RUNTIME_CACHE_ENABLED": "1",
            "SPARKRUN_NODE_RANK": "2",
            "SPARKRUN_PORT": "9123",
            "SERVER_PORT": "9123",
            "SPARKRUN_MODEL_REVISION": REVISION,
            "SPARKRUN_RUNTIME_CACHE_DIR": str(self.cache),
            "DSV41_SOURCE": str(self.model),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        }
        for key, suffix in {
            "STATE_PATH": "state",
            "B12X_ROCE_CACHE_DIR": "b12x-roce",
            "B12X_COMPILE_CACHE_DIR": "b12x-compile",
            "DSV41_PACKED_DIR": "engram/v1/" + REVISION,
        }.items():
            self.env[key] = str(self.cache / suffix)

    def write_checkpoint(self):
        index = {}
        for layer in prep.LAYERS:
            prefix = f"layers.{layer}.engram.embed."
            name = f"layer-{layer}.safetensors"
            index.update({prefix + "weight": name, prefix + "scale": name})
            header = json.dumps(
                {
                    prefix + "weight": {
                        "dtype": "F8_E4M3",
                        "shape": [8, 256],
                        "data_offsets": [0, 2048],
                    },
                    prefix + "scale": {
                        "dtype": "F8_E8M0",
                        "shape": [8, 8],
                        "data_offsets": [2048, 2112],
                    },
                }
            ).encode()
            (self.model / name).write_bytes(struct.pack("<Q", len(header)) + header + bytes(2112))
        (self.model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))

    def packed_shard(self, layer=1, rank=2):
        path = Path(self.env["DSV41_PACKED_DIR"])
        path.mkdir(parents=True, exist_ok=True)
        output = path / f"engram-l{layer}-r{rank}of4.bin"
        header = struct.pack("<6Q", prep.MAGIC, layer, rank * 2, rank * 2 + 2, 8, 264)
        output.write_bytes(header + bytes(4096 - len(header) + 528))
        return output

    def test_environment_rejects_unsupported_shapes_and_cache(self):
        self.assertEqual(prep.validate_environment(self.env)[2:], (2, REVISION))
        for key, value in {
            "SPARKRUN_NODE_RANK": "4",
            "SPARKRUN_NUM_NODES": "3",
            "TP_SIZE": "2",
            "NNODES": "2",
            "EP_SIZE": "2",
            "DSPARK_BLOCK_SIZE": "3",
            "CUDA_GRAPH_MAX_BS_DECODE": "8",
            "OFFLOAD_MODE": "ram",
            "SPARKRUN_RUNTIME_CACHE_ENABLED": "0",
            "SPARKRUN_RUNTIME_CACHE_DIR": "",
            "SPARKRUN_MODEL_REVISION": "main",
            "DSV41_PACKED_DIR": "/tmp/other",
            "SERVER_PORT": "8000",
            "DSV41_FAST_LOAD": "1",
            "DSV41_SOURCE": "/wrong",
            "SPARKRUN_HOOK": "post_exec",
        }.items():
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                prep.validate_environment(dict(self.env, **{key: value}))

    def test_checkpoint_metadata_and_real_truncation_control(self):
        self.assertEqual(prep.checkpoint_rows(self.model), {1: 8, 14: 8})
        path = self.model / "layer-1.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            prep.checkpoint_rows(self.model)
        self.write_checkpoint()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            prep.checkpoint_rows(self.model)

    def test_packed_header_and_length_controls(self):
        path = self.packed_shard()
        self.assertTrue(prep.shard_complete(path, 1, 8, 2))
        with self.assertRaisesRegex(RuntimeError, "header"):
            prep.shard_complete(path, 1, 8, 1)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            prep.shard_complete(path, 1, 8, 2)
        self.assertFalse(prep.shard_complete(path.with_name("absent.bin"), 1, 8, 2))

    def test_receipt_refuses_unidentified_shards_and_changed_checkpoint(self):
        path = self.packed_shard().parent
        with patch.object(prep, "PACKER", self.packer):
            with self.assertRaisesRegex(RuntimeError, "unidentified"):
                prep.validate_receipt(path, self.model, REVISION, {1: 8, 14: 8}, create=True)
            next(path.glob("*.bin")).unlink()
            prep.validate_receipt(path, self.model, REVISION, {1: 8, 14: 8}, create=True)
            prep.validate_receipt(path, self.model, REVISION, {1: 8, 14: 8})
            (self.model / "config.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                prep.validate_receipt(path, self.model, REVISION, {1: 8, 14: 8})

    def test_detached_rank_specific_packer_and_warm_reuse(self):
        with (
            patch.object(prep, "validate_image"),
            patch.object(prep, "PACKER", self.packer),
            patch.object(prep.subprocess, "Popen") as spawn,
        ):
            prep.prepare(self.env)
            args, kwargs = spawn.call_args
            self.assertEqual(
                args[0][-4:],
                ["--pack-worker", str(self.model), self.env["DSV41_PACKED_DIR"], "2"],
            )
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(len(kwargs["pass_fds"]), 1)
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertNotIn("DSV41_SOURCE", kwargs["env"])
            if os.geteuid() == 0:
                self.assertEqual(kwargs["user"], self.cache.stat().st_uid)
                self.assertEqual(kwargs["group"], self.cache.stat().st_gid)
                self.assertEqual(kwargs["extra_groups"], [])
            for layer in prep.LAYERS:
                self.packed_shard(layer)
            spawn.reset_mock()
            prep.prepare(self.env)
            spawn.assert_not_called()

    def test_busy_pack_lock_avoids_second_child_and_checks_identity(self):
        with (
            patch.object(prep, "validate_image"),
            patch.object(prep, "PACKER", self.packer),
            patch.object(prep.subprocess, "Popen") as spawn,
        ):
            prep.prepare(self.env)
            lock_path = Path(self.env["DSV41_PACKED_DIR"]) / ".pack.lock"
            with lock_path.open("a+b") as lock:
                prep.fcntl.flock(lock, prep.fcntl.LOCK_EX | prep.fcntl.LOCK_NB)
                spawn.reset_mock()
                prep.prepare(self.env)
                spawn.assert_not_called()
                (self.model / "config.json").write_text('{"changed":true}')
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    prep.prepare(self.env)

    def test_actual_image_probe_requires_one_visible_gpu(self):
        with (
            patch.object(prep.Path, "is_file", return_value=True),
            patch.object(prep.Path, "is_dir", return_value=True),
            patch.object(prep.Path, "read_text", return_value="EngramLoader"),
            patch.object(prep.subprocess, "run") as probe,
        ):
            probe.return_value.stdout = "1\n"
            prep.validate_image()
            self.assertNotIn("DSV41_SOURCE", probe.call_args.kwargs["env"])
            for count in (0, 2):
                probe.return_value.stdout = str(count) + "\n"
                with self.assertRaisesRegex(RuntimeError, "one visible"):
                    prep.validate_image()

    def test_optional_spawn_failure_preserves_fallback(self):
        with (
            patch.object(prep, "validate_image"),
            patch.object(prep, "PACKER", self.packer),
            patch.object(prep.subprocess, "Popen", side_effect=OSError("no space")),
            patch.object(prep, "log") as log,
        ):
            prep.prepare(self.env)
            self.assertIn("using checkpoint fallback", log.call_args.args[0])

    def test_worker_uses_image_packer_explicit_model_rank_and_low_priority(self):
        with (
            patch.object(prep, "PACKER", self.packer),
            patch.object(prep.subprocess, "run") as run,
        ):
            run.return_value.returncode = 1
            self.assertEqual(prep.pack_worker(str(self.model), "packed", "3"), 1)
            command = run.call_args.args[0]
            self.assertEqual(command[:6], ["nice", "-n", "10", "ionice", "-c", "3"])
            self.assertEqual(
                command[-8:],
                [
                    "--model",
                    str(self.model),
                    "--rank",
                    "3",
                    "--tp",
                    "4",
                    "--out",
                    "packed",
                ],
            )

    def test_recipe_uses_defaults_and_resolvable_vendored_mod(self):
        text = RECIPE.read_text()
        for removed in ("command:", "pre_exec:", "  shm_size:", "  served_model_name:", "# SPDX-License-Identifier:", "env_templates:"):
            self.assertNotIn(removed, text)
        config = defaults()
        self.assertEqual(config["expert_parallel"], 1)
        self.assertEqual(config["max_total_tokens"], 9400000)
        self.assertEqual(config["speculative_dspark_block_size"], 5)
        self.assertEqual(config["gpu_memory_utilization"], 0.8)
        self.assertEqual(config["model_type"], "llm")
        self.assertEqual(config["revision"], "{model_revision}")
        self.assertTrue(config["enable_decoder_swa_bounded_replay"])
        self.assertNotIn("inference", mapping("readiness"))
        self.assertIn('min_sparkrun_version: "0.4.0"', text)
        self.assertIn("model_revision: fb2764a5cf321eaa5070ca8f9e892818f477c16d", text)
        ref = section("mods")[0].removeprefix("- ")
        self.assertEqual((ROOT / ref).resolve(), MOD.resolve())
        self.assertTrue((MOD / "run.sh").is_file())
        provenance = json.loads((MOD / "PROVENANCE.json").read_text())
        self.assertEqual(len(provenance["files"]), 3)
        for entry in provenance["files"]:
            data = (MOD / entry["vendored_path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
        self.assertNotIn("launcher.py", (MOD / "run.sh").read_text())

    def test_mod_topology_guards_have_negative_controls(self):
        for key, bad in (
            ("TP_SIZE", "2"),
            ("DATA_PARALLEL", "2"),
            ("PIPELINE_PARALLEL", "2"),
            ("SPARKRUN_NUM_NODES", "3"),
            ("CONTEXT_LENGTH", "4095"),
            ("CONTEXT_LENGTH", "1048577"),
        ):
            with self.subTest(key=key, value=bad), self.assertRaises(RuntimeError):
                prep.validate_environment({**self.env, key: bad})
        for context in ("4096", "1048576"):
            prep.validate_environment({**self.env, "CONTEXT_LENGTH": context})

    def test_template_contract_and_attributions(self):
        templates = mapping("env")
        self.assertEqual(templates["DSV41_SOURCE"], "{launch.model_path}")
        self.assertIn("{launch.model_revision}", templates["DSV41_PACKED_DIR"])
        env = mapping("env")
        for name in ("NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "B12X_ROCE_HCA"):
            self.assertNotIn(name, env)
        for key in ("MODEL_PATH", "NODE_RANK", "DSV41_CHECKPOINT_REVISION", "MEM_FRACTION_STATIC", "MAX_RUNNING_REQUESTS", "MAX_TOTAL_TOKENS", "SPEC_ALGO", "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS"):
            self.assertNotIn(key, env)
        text = RECIPE.read_text()
        self.assertIn("author: Little Cedar Group", text)
        self.assertEqual(text.count("    - author:"), 1)
        self.assertIn("/blob/fdd1003ba2a1e766a0129d26ef769ae4025c8c30/recipes/ds4/", text)
        self.assertIn("Offline-validated only", text)
        self.assertIn("@sha256:" + prep.IMAGE_DIGEST, text)


if __name__ == "__main__":
    unittest.main()
