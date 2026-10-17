"""Tests for incremental streaming of nested tool-call parameters in M3TextParser.

The original implementation buffers an entire nested parameter body until its
closing tag before emitting any ``arguments`` JSON. For deeply nested params
(OpenCode's ``question`` tool: array→object→array→object) this produces a
multi-second SSE stall while the model generates ~150-300 tokens of XML.

The fix streams JSON fragments as each child element closes.
"""
import json

import pytest

from sglang.srt.function_call._llm_nom.m3_text import M3TextParser

from conftest import (
    NS,
    QUESTION_FUNCTIONS,
    build_question_call,
    collect_args,
    feed,
    tag,
    tokenize_like_m3,
)


def _make_parser():
    return M3TextParser(
        with_reasoning=False, reasoning_prefix="", functions=QUESTION_FUNCTIONS
    )


# ---------------------------------------------------------------------------
# Incremental streaming behaviour
# ---------------------------------------------------------------------------


def test_nested_param_streams_first_leaf_before_close():
    """First leaf value appears in ``arguments`` before the parameter closes.

    Feed everything *up to and including* the first ``</question>`` close.
    The streamed args must already contain the question text — we must not be
    buffering until ``</questions>``.
    """
    questions = [
        {
            "question": "Which framework?",
            "header": "FW",
            "options": [
                {"label": "React", "description": "r"},
                {"label": "Vue", "description": "v"},
            ],
        }
    ]
    raw, _expected = build_question_call(questions)

    # Cut the stream right after the first leaf close: ``...</question>``
    marker = "Which framework?" + NS + "</question>"
    cut = raw.index(marker) + len(marker)
    head = raw[:cut]

    p = _make_parser()
    deltas = feed(p, tokenize_like_m3(head))
    args = collect_args(deltas)

    assert "Which framework?" in args, (
        "first leaf value should be streamed as soon as its closing tag arrives; "
        f"got args={args!r}"
    )


def test_nested_param_streams_per_array_item():
    """Each ``<item>`` in the questions array streams as it closes.

    With 3 questions, after the first ``</item>`` closes we must already have
    the first question's full JSON object in the args stream — not wait for
    all three.
    """
    questions = [
        {
            "question": f"Q{i}",
            "header": f"H{i}",
            "options": [{"label": f"L{i}", "description": f"D{i}"}],
        }
        for i in range(3)
    ]
    raw, _expected = build_question_call(questions)

    chunks = tokenize_like_m3(raw)
    p = _make_parser()

    # Feed chunk-by-chunk; record args length each time we cross a question's
    # closing ``</item>`` (depth-1 item, not the inner option item).
    args = ""
    seen_q0_at = None
    fed = ""
    q0_close = (
        tag("item", tag("label", "L0") + tag("description", "D0"))  # inner option item
    )
    # The depth-1 ``</item>`` for Q0 is the one immediately preceding Q1's ``<item>``
    q0_outer_close_marker = "D0" + NS + "</description>" + NS + "</item>" + NS + "</options>" + NS + "</item>"
    for c in chunks:
        p.update(c)
        d = p.get_delta()
        if d:
            args += "".join(
                tc.get("function", {}).get("arguments", "")
                for tc in d.get("tool_calls", [])
            )
        fed += c
        if seen_q0_at is None and fed.endswith(q0_outer_close_marker):
            seen_q0_at = args

    assert seen_q0_at is not None, "test bug: never observed Q0 outer </item>"
    assert "Q0" in seen_q0_at and "L0" in seen_q0_at, (
        "Q0's content should be in args by the time its outer </item> closes; "
        f"args at that point: {seen_q0_at!r}"
    )
    assert "Q1" not in seen_q0_at, (
        "Q1 has not been generated yet, must not appear; "
        f"args at that point: {seen_q0_at!r}"
    )


# ---------------------------------------------------------------------------
# Correctness must be preserved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "questions",
    [
        # Single question, two options
        [
            {
                "question": "Which framework?",
                "header": "FW",
                "options": [
                    {"label": "React", "description": "Use React"},
                    {"label": "Vue", "description": "Use Vue"},
                ],
            }
        ],
        # Three questions, three options each, with multiple=false
        [
            {
                "question": f"Q{i}?",
                "header": f"H{i}",
                "options": [
                    {"label": f"L{i}{j}", "description": f"D{i}{j}"} for j in range(3)
                ],
                "multiple": False,
            }
            for i in range(3)
        ],
        # Question with special chars needing JSON escaping
        [
            {
                "question": 'Use "quotes" & <tags>?',
                "header": "Esc",
                "options": [{"label": "a\nb", "description": "back\\slash"}],
            }
        ],
    ],
    ids=["single", "three_by_three", "escaping"],
)
def test_streamed_args_are_valid_json_matching_input(questions):
    """Regardless of chunking, the final concatenated args must json.loads() to
    exactly the input structure."""
    raw, expected = build_question_call(questions)

    for splitter in (lambda s: [s], tokenize_like_m3):
        p = _make_parser()
        deltas = feed(p, splitter(raw))
        args = collect_args(deltas)
        parsed = json.loads(args)
        assert parsed == expected, (
            f"chunking={splitter.__name__}: parsed args mismatch.\n"
            f"  expected: {json.dumps(expected)}\n"
            f"  got:      {json.dumps(parsed)}\n"
            f"  raw args: {args}"
        )


def test_streamed_args_chunk_invariance():
    """Final args must be identical across all chunk sizes (NS-aligned, single,
    and arbitrary char splits once the StringByParts fix lands)."""
    questions = [
        {
            "question": "Which?",
            "header": "H",
            "options": [
                {"label": "A", "description": "a"},
                {"label": "B", "description": "b"},
            ],
        }
    ]
    raw, expected = build_question_call(questions)

    results = {}
    for name, chunks in [
        ("whole", [raw]),
        ("ns_aligned", tokenize_like_m3(raw)),
    ]:
        p = _make_parser()
        deltas = feed(p, chunks)
        results[name] = collect_args(deltas)

    assert json.loads(results["whole"]) == expected
    assert results["whole"] == results["ns_aligned"], (
        f"args differ across chunkings:\n"
        f"  whole:      {results['whole']}\n"
        f"  ns_aligned: {results['ns_aligned']}"
    )


# ---------------------------------------------------------------------------
# Non-nested (primitive) parameters keep streaming char-by-char
# ---------------------------------------------------------------------------


def test_primitive_param_still_streams_char_by_char():
    """Regression guard: a flat string parameter (the ``bash``-style case) must
    continue to stream incrementally via the existing primitive path."""
    funcs = {
        "bash": {
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            }
        }
    }
    raw = (
        f'{NS}<tool_call>\n'
        f'{NS}<invoke name="bash">{tag("command", "ls -la /tmp")}{NS}</invoke>\n'
        f'{NS}</tool_call>'
    )
    p = M3TextParser(with_reasoning=False, reasoning_prefix="", functions=funcs)

    # Feed up to the middle of the command value.
    cut = raw.index("ls -la") + len("ls -la")
    deltas = feed(p, tokenize_like_m3(raw[:cut]))
    args = collect_args(deltas)
    assert "ls -la" in args, f"primitive value should stream as it arrives: {args!r}"

    # Finish; final args must be valid JSON.
    deltas += feed(p, tokenize_like_m3(raw[cut:]))
    args = collect_args(deltas)
    assert json.loads(args) == {"command": "ls -la /tmp"}


def test_empty_nested_param_emits_empty_container():
    """``<questions></questions>`` (no children) → ``[]`` per schema."""
    raw = (
        f'{NS}<tool_call>\n'
        f'{NS}<invoke name="question">{NS}<questions>{NS}</questions>{NS}</invoke>\n'
        f'{NS}</tool_call>'
    )
    p = _make_parser()
    deltas = feed(p, tokenize_like_m3(raw))
    args = collect_args(deltas)
    assert json.loads(args) == {"questions": []}
