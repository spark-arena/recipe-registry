# MiniMax-M3 sglang tool-call parser patch

Streams nested tool-call parameter bodies incrementally instead of buffering
the whole body until the closing tag. Without this, deeply nested params
(e.g. OpenCode's `question` tool: array→object→array→object) stall the SSE
`arguments` stream for tens of seconds while the model generates ~150-300
tokens of namespace-prefixed XML — the client sees a blank tool block and
users abort.

Measured on the 4× GB10 cluster, 3 questions × 3 options, 81-tool context:
**max gap between argument chunks 37.4s → 3.8s**.

## Contents

| | |
|---|---|
| `_llm_nom/` | full patched package; overlay onto `dist-packages/sglang/srt/function_call/_llm_nom/` |
| `m3_text.patch` | unified diff vs `scitrera/dgx-spark-sglang-mm:v0` — the actual fix |
| `generator.patch` | comment-only; documents why `StringByParts` short-buffer-as-mismatch is intentional (EOS safety) |
| `tests/` | 25 pytest cases covering streaming, correctness, malformed-input tolerance, and the StringByParts contract |
| `run_tests.sh` | runs the suite inside the container with the overlay applied |

## How the recipe applies it

The `pre_exec` hook in `minimax-m3-v0-nvfp4-4x.yaml` copies
`_llm_nom/{m3_text,generator}.py` from the HF-cache mount into
`dist-packages` after the container starts and before `sglang serve` runs.

The files must be staged on each node first:

```sh
for h in spark1 spark2 spark7 spark8; do
  ssh $h.dca.zetier.com 'mkdir -p ~/.cache/huggingface/sparkrun-patches/minimax-m3/_llm_nom'
  rsync -a _llm_nom/{m3_text,generator}.py \
    $h.dca.zetier.com:~/.cache/huggingface/sparkrun-patches/minimax-m3/_llm_nom/
done
```

## Running the tests

```sh
./run_tests.sh
```

## Behavioural notes vs the original batch parser

Model-error edge cases are handled to keep the emitted JSON parseable and the
outer parse loop in sync — bare-NS segments, missing `>`, stray close tags,
and mixed text/element content all degrade gracefully (see
`tests/test_m3_malformed_input.py`).

One accepted regression: duplicate object keys emit as repeated `"k": v`
pairs (last-wins on `json.loads`) instead of the batch parser's
`"k": [v1, v2]` collapse — streaming cannot retroactively rewrite an
already-emitted key.
