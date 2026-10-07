# Native recipe environment audit

Audit date: 2026-10-06. The recipe has **58 active env values**, including **15
interpolated values**. It previously had 73 active entries. Defaults, hook context
and the pinned image now supply the removed duplicates. NCCL debugging settings
remain commented out for opt-in use.

Sources: the image's pinned knapcio tree
[`58f2321`](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4/tree/58f232155917d388b6053eca079c617fd306c33c)
(40 adapter/launcher/overlay files), its b12x/b12x_next compiler sources, and
[SGLang's pinned environment definitions](https://github.com/sgl-project/sglang/blob/f80c91a4b97e52e17fbcff088aa6b47b6270f6b9/python/sglang/srt/environ.py).
`boot.py run` is bypassed, but `boot.py health` remains the image healthcheck.
This is a source audit; no container/GPU trial or inspection of the pulled image
was performed. The image's separately documented health-port patch remains a
hardware-trial verification item.

## Removed entries

| Variables | Replacement/reason |
| --- | --- |
| `MEM_FRACTION_STATIC`, `MAX_RUNNING_REQUESTS`, `MAX_TOTAL_TOKENS`, `SPEC_ALGO` | Only needed by the bypassed launcher to construct engine flags or warmup. Defaults generate those flags directly. Removed duplicate hook checks of these launcher inputs. |
| `MODEL_PATH` | The adapter/preparation reads `DSV41_SOURCE`. The image's inherited MODEL_PATH is unused by serving and its health-only command. |
| `NODE_RANK` | The mod reads `SPARKRUN_NODE_RANK`; Sparkrun generates engine rank flags. |
| `DSV41_CHECKPOINT_REVISION` | The mod reads pinned `SPARKRUN_MODEL_REVISION`; engine defaults reference top-level `model_revision`. |
| `OFFLOAD_MODE` | The pinned Dockerfile already sets `nvme`; the hook rejects RAM mode. |
| `DSV41_STATS_SECONDS` | Both active consumers default to 60. |
| `DSV41_FAST_LOAD` | The adapter defaults to disabled; the hook still rejects enabling it for this baseline. |
| `B12X_NEXT_COMPILE_CACHE_DIR` | The active MoE adapter derives it from `STATE_PATH/b12x-next-compile`. The mod creates/validates that same directory; it remains separate from b12x. |
| `NCCL_LL128_BUFFSIZE` | This lane excludes LL128 with `NCCL_PROTO=^LL128`. |
| `PYTHONWARNINGS` | Optional warning suppression, unnecessary for serving behavior. |
| `NCCL_DEBUG`, `NCCL_DEBUG_SUBSYS` | Commented examples (`INFO`, `INIT,ENV`) for debugging, not active overrides. |

## Retained entries

The adapter's default graph batch size and DSpark block size happen to match this
recipe, but retaining their interpolated values keeps engine flags and adapter
inputs aligned when the defaults change. The EP/DP/PP/context values are explicit
preparation guards, rather than controls consumed by SGLang itself.

| Variable | Consumer / reason |
| --- | --- |
| `PYTHONUNBUFFERED` | Python process setting; preserves unbuffered logs after removing `python -u`. |
| `DSV41_CACHE_GIB` | `adapter/engram_backend.py:126`: active adapter setting/gate. |
| `DSV41_CACHE_WAYS` | `adapter/row_store.cpp:383`: active adapter setting/gate. |
| `DSV41_IO_THREADS` | `adapter/row_store.cpp:265`: active adapter setting/gate. |
| `DSV41_RESIDENT_SCALES` | `adapter/row_store.cpp:458`: active adapter setting/gate. |
| `DSV41_ENGRAM_PREFETCH` | `adapter/engram_prefetch.py:36`: active adapter setting/gate. |
| `DSV41_INDEXER_CHUNKED` | `adapter/indexer_chunked.py:214`: active adapter setting/gate. |
| `SGLANG_DSPARK_FOLDED_SAMPLING` | SGLang DSpark sampling mode; required by the enabled adapter stack. |
| `SGLANG_RUST_BUILD_MODE` | SGLang Rust extension loader; `never` overrides its `auto` default. |
| `SPARK_PREFILL_TP_SPLIT` | `adapter/indexer_chunked_v3.py:364`: active adapter setting/gate. |
| `DSV41_SHARED_PAD_K` | `adapter/shared_pad_k.py:81`: active adapter setting/gate. |
| `DSV41_WO_A_W8` | `adapter/sitecustomize.py:77`: active adapter setting/gate. |
| `DSV41_WO_A_W8_MID` | `adapter/wo_a_w8.py:79`: active adapter setting/gate. |
| `DSV41_WO_A_W8_DROP` | `adapter/wo_a_w8.py:126`: active adapter setting/gate. |
| `DSV41_DRAFT_TAU` | `adapter/draft_tau.py:17`: active adapter setting/gate. |
| `DSV41_DRAFT_HEAD_FP8` | `adapter/draft_head_fp8.py:14`: active adapter setting/gate. |
| `DSV41_BLOCK_VERIFY` | `adapter/block_verify.py:25`: active adapter setting/gate. |
| `DSV41_FOLDED_FENCE` | `adapter/folded_result_fence.py:14`: active adapter setting/gate. |
| `DSV41_VERIFY_CAP` | `adapter/block_verify.py:94`: active adapter setting/gate. |
| `DSV41_AUTOTUNE_KEEP` | `adapter/sitecustomize.py:189`: active adapter setting/gate. |
| `DSV41_REPLICATED_SPLIT` | `adapter/replicated_split.py:44`: active adapter setting/gate. |
| `DSV41_DRAFT_MAIN_PROJ_SPLIT` | `adapter/draft_main_proj.py:21`: active adapter setting/gate. |
| `DSV41_SPLIT_COMPACT_GATHER` | `adapter/replicated_split.py:47`: active adapter setting/gate. |
| `DSV41_ROUTER_LIVE` | `adapter/router_live.py:16`: active adapter setting/gate. |
| `DSV41_MOE_B12X_NEXT` | `adapter/moe_b12x_next.py:81`: active adapter setting/gate. |
| `DSV41_MOE_B12X_NEXT_DETERMINISTIC` | `adapter/moe_b12x_next.py:100`: active adapter setting/gate. |
| `DSV41_HC_FUSED` | `adapter/hc_fused.py:43`: active adapter setting/gate. |
| `DSV41_PREFILL_SP` | `adapter/moe_b12x_next.py:110`: active adapter setting/gate. |
| `DSV41_PREFILL_SP_FP8` | `adapter/prefill_sp.py:72`: active adapter setting/gate. |
| `DSV41_L2_PREFETCH` | `adapter/l2_prefetch.py:142`: active adapter setting/gate. |
| `DSV41_L2_PREFETCH_WOA` | `adapter/ab_variant.py:57`: active adapter setting/gate. |
| `DSV41_SPEC_SYNC_FREE` | `adapter/sitecustomize.py:129`: active adapter setting/gate. |
| `DSV41_EAGER_GLUE` | `adapter/eager_glue.py:89`: active adapter setting/gate. |
| `SGLANG_ROCE_ALLREDUCE` | `runtime/roce_tp4_adapt.py`: applies this environment control to the image’s RoCEnante SGLang overlay. |
| `SGLANG_ROCE_MAX_SIZE` | `runtime/roce_tp4_adapt.py`: applies this environment control to the image’s RoCEnante SGLang overlay. |
| `DSV41_ROCE_GATHER` | `adapter/roce_gather.py:16`: active adapter setting/gate. |
| `NCCL_P2P_DISABLE` | NCCL runtime transport/buffer/protocol policy retained from the source lane. |
| `NCCL_SHM_DISABLE` | NCCL runtime transport/buffer/protocol policy retained from the source lane. |
| `NCCL_CUMEM_ENABLE` | NCCL runtime transport/buffer/protocol policy retained from the source lane. |
| `NCCL_BUFFSIZE` | NCCL runtime transport/buffer/protocol policy retained from the source lane. |
| `NCCL_PROTO` | NCCL runtime transport/buffer/protocol policy retained from the source lane. |
| `NCCL_MAX_NCHANNELS` | NCCL runtime transport/buffer/protocol policy retained from the source lane. |
| `PYTORCH_CUDA_ALLOC_CONF` | PyTorch allocator; overrides Sparkrun GB10 platform default to disable expandable segments. |
| `CHUNKED_PREFILL_SIZE` | `adapter/moe_b12x_next.py` prefill plan sizing; follows the engine flag. |
| `CUDA_GRAPH_MAX_BS_DECODE` | `adapter/moe_b12x_next.py` graph batch list; follows the engine flag even when overridden. |
| `DSPARK_BLOCK_SIZE` | `adapter/moe_b12x_next.py` and `verify_cap.py`; follows the engine flag even when overridden. |
| `EP_SIZE` | Preparation guard: this profile supports EP1. |
| `DATA_PARALLEL` | Preparation guard: this profile supports DP1. |
| `PIPELINE_PARALLEL` | Preparation guard: this profile supports PP1. |
| `TP_SIZE` | `adapter/tp3_pad.py`; TP layout also checked by preparation. |
| `NNODES` | `adapter/engram_backend.py`; divides cache budget by ranks per host. |
| `SERVER_PORT` | Image `boot.py health`; follows the actual serving port without invoking `boot.py run`. |
| `CONTEXT_LENGTH` | Preparation guard: context must stay within 4096–1048576. |
| `DSV41_SOURCE` | `adapter/sitecustomize.py` and `engram_backend.py`; actual checkpoint directory for adapters/preparation. |
| `DSV41_PACKED_DIR` | `adapter/engram_backend.py`; concrete-revision namespace for rank-local packed shards. |
| `STATE_PATH` | MoE adapter derives `state/b12x-next-compile`; preparation also keeps packer logs here. |
| `B12X_ROCE_CACHE_DIR` | b12x RoCEnante compiler cache; retain its output inside the shared runtime cache. |
| `B12X_COMPILE_CACHE_DIR` | b12x compiler and L2 prefetch compiler; retain compiled output inside the shared runtime cache. |
