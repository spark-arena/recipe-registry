"""``_literal`` / ``StringByParts`` atomic-part contract tests.

``StringByParts`` parts are **token-aligned by contract**: each part
corresponds to one tokenizer token and therefore arrives in a single
``update()`` chunk. Under that contract, ``len(buffer) < part_len`` means the
buffered bytes are *not* (the start of) the part — they are content that
happens to share a prefix.

Treating short-buffer as mismatch is therefore intentional: it lets
``_take_any`` emit those bytes immediately instead of withholding them at
end-of-stream. The alternative (wait on any valid prefix) would correctly
handle a hypothetical split-NS — which never happens in production — at the
cost of truncating any content that ends in ``]``.

These tests pin that contract so a future "fix" doesn't reintroduce the EOS
truncation.
"""
import json

from sglang.srt.function_call._llm_nom.m3_text import M3TextParser
from sglang.srt.function_call._llm_nom.generator import (
    GeneratorParser,
    StringByParts,
)

from conftest import NS, QUESTION_FUNCTIONS, build_question_call, feed


class _ProbeParser(GeneratorParser):
    """Minimal parser: take content until a StringByParts target, then stop."""

    def __init__(self, target):
        self._target = target
        super().__init__()

    def _process(self):
        yield from self._take_any(
            until=self._target, key="content", should_consume_suffix=True
        )
        yield from self._take_any(key="tail")


# ---------------------------------------------------------------------------
# The atomic-part contract: short buffer → mismatch, no EOS withholding
# ---------------------------------------------------------------------------


def test_stringbyparts_short_buffer_is_mismatch_not_wait():
    """A buffer shorter than the part is treated as mismatch — the bytes are
    emitted as content immediately. This is the EOS-safety property."""
    target = StringByParts(parts=[NS])
    p = _ProbeParser(target)
    # Feed a lone ']' (valid prefix of NS, but NS is one token → this is content).
    p.update("]")
    final = p.get_final()
    assert final.get("content") == "]", (
        "trailing ']' must be emitted, not withheld waiting for more NS bytes; "
        f"got {final!r}"
    )


def test_m3_content_ending_in_bracket_is_not_truncated():
    """End-to-end: assistant content ending in ``]`` (common: code, JSON,
    lists) must not lose the trailing char when tools are defined."""
    p = M3TextParser(
        with_reasoning=False, reasoning_prefix="", functions=QUESTION_FUNCTIONS
    )
    for c in ["result is [1, 2, 3", "]"]:
        p.update(c)
    final = p.get_final()
    assert final.get("content") == "result is [1, 2, 3]", (
        f"trailing ']' truncated: {final!r}"
    )


def test_stringbyparts_full_part_in_one_chunk_matches():
    """When the part arrives whole (the production case), it matches."""
    target = StringByParts(parts=[NS])
    p = _ProbeParser(target)
    feed(p, ["hello", NS, "world"])
    final = p.get_final()
    assert final.get("content") == "hello"
    assert final.get("tail") == "world"


# ---------------------------------------------------------------------------
# Document the known limitation (xfail, not a bug to fix)
# ---------------------------------------------------------------------------


def test_stringbyparts_split_part_leaks_to_content_by_design():
    """If a part DOES arrive split (contract violation — would require the NS
    token to be BPE-split), it leaks to content. This is the documented cost
    of the EOS-safety property above."""
    target = StringByParts(parts=[NS])
    p = _ProbeParser(target)
    feed(p, list("hello" + NS + "world"))  # one char at a time
    final = p.get_final()
    # NS leaks because each char is shorter than the 13-char part.
    assert NS in final.get("content", ""), (
        "this test documents the atomic-part contract; if it fails, verify "
        "test_m3_content_ending_in_bracket_is_not_truncated still passes"
    )


def test_m3_ns_aligned_chunking_parses_correctly():
    """Production-realistic chunking (NS arrives whole, other text in pieces)
    parses tool calls correctly."""
    questions = [
        {
            "question": "Which?",
            "header": "H",
            "options": [{"label": "A", "description": "a"}],
        }
    ]
    raw, expected = build_question_call(questions)

    p = M3TextParser(
        with_reasoning=False, reasoning_prefix="", functions=QUESTION_FUNCTIONS
    )
    # Split on NS boundaries; NS itself stays whole.
    chunks = []
    for i, seg in enumerate(raw.split(NS)):
        if i:
            chunks.append(NS)
        for j in range(0, len(seg), 3):
            chunks.append(seg[j : j + 3])
    feed(p, [c for c in chunks if c])
    final = p.get_final()

    assert "tool_calls" in final
    assert json.loads(final["tool_calls"][0]["function"]["arguments"]) == expected
    assert NS not in (final.get("content") or "")
