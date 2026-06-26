"""Integration tests through ``MinimaxM3Detector`` — the layer sglang's
serving code actually drives. Verifies the streaming fix survives the
detector → ``StreamingParseResult`` plumbing."""
import json

import pytest

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.minimax_m3 import MinimaxM3Detector

from conftest import (
    NS,
    QUESTION_PARAMS,
    build_question_call,
    tag,
    tokenize_like_m3,
)


def _question_tool() -> Tool:
    return Tool(
        **{
            "type": "function",
            "function": {
                "name": "question",
                "description": "Ask the user.",
                "parameters": QUESTION_PARAMS,
            },
        }
    )


def test_detector_streaming_increments_arrive_per_item():
    """Driving ``parse_streaming_increment`` token-by-token must surface
    ``ToolCallItem`` argument fragments as each question item closes — not in
    one terminal burst."""
    questions = [
        {
            "question": f"Q{i}",
            "header": f"H{i}",
            "options": [{"label": f"L{i}", "description": f"D{i}"}],
        }
        for i in range(3)
    ]
    raw, expected = build_question_call(questions)
    tools = [_question_tool()]

    det = MinimaxM3Detector()
    args = ""
    snapshot_after_q0 = None
    fed = ""
    q0_close_marker = NS + "</options>" + NS + "</item>"
    q0_close_at = raw.index("D0") + raw[raw.index("D0") :].index(q0_close_marker) + len(
        q0_close_marker
    )

    for chunk in tokenize_like_m3(raw):
        result = det.parse_streaming_increment(chunk, tools)
        for c in result.calls:
            if c.parameters:
                args += c.parameters
        fed += chunk
        if snapshot_after_q0 is None and len(fed) >= q0_close_at:
            snapshot_after_q0 = args

    assert det._error is None, f"detector errored: {det._error!r}"
    assert snapshot_after_q0 is not None
    assert "Q0" in snapshot_after_q0 and "L0" in snapshot_after_q0, (
        f"args after Q0 close: {snapshot_after_q0!r}"
    )
    assert json.loads(args) == expected


def test_detector_non_streaming_unchanged():
    """``detect_and_parse`` (non-streaming path) still produces one complete
    ``ToolCallItem`` with valid JSON args."""
    questions = [
        {
            "question": "Which framework?",
            "header": "FW",
            "options": [
                {"label": "React", "description": "r"},
                {"label": "Vue", "description": "v"},
            ],
            "multiple": False,
        }
    ]
    raw, expected = build_question_call(questions)
    tools = [_question_tool()]

    det = MinimaxM3Detector()
    result = det.detect_and_parse(raw, tools)

    assert len(result.calls) == 1
    call = result.calls[0]
    assert call.name == "question"
    assert json.loads(call.parameters) == expected


# ---------------------------------------------------------------------------
# Model-error edge cases — emitted JSON must stay parseable
# ---------------------------------------------------------------------------


def _parse_args_via_detector(raw: str, params_schema: dict) -> dict:
    tools = [
        Tool(
            **{
                "type": "function",
                "function": {"name": "f", "description": "", "parameters": params_schema},
            }
        )
    ]
    det = MinimaxM3Detector()
    args = ""
    for chunk in tokenize_like_m3(raw):
        for c in det.parse_streaming_increment(chunk, tools).calls:
            if c.parameters:
                args += c.parameters
    assert det._error is None, f"detector errored: {det._error!r}"
    return json.loads(args)


def test_object_param_with_nested_object():
    """Non-array nested param: object→object."""
    schema = {
        "type": "object",
        "properties": {
            "config": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer"},
                },
            }
        },
    }
    raw = (
        f'{NS}<tool_call>\n{NS}<invoke name="f">'
        f'{tag("config", tag("host", "localhost") + tag("port", "8080"))}'
        f'{NS}</invoke>\n{NS}</tool_call>'
    )
    parsed = _parse_args_via_detector(raw, schema)
    assert parsed == {"config": {"host": "localhost", "port": 8080}}


def test_unclosed_inner_tag_force_closed_on_parent_close():
    """Model forgets ``</label>`` — parent close must force-close it without
    breaking the JSON stream."""
    schema = {
        "type": "object",
        "properties": {
            "opts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                },
            }
        },
    }
    # <opts><item><label>A</item></opts>   (missing </label>)
    raw = (
        f'{NS}<tool_call>\n{NS}<invoke name="f">'
        f'{NS}<opts>{NS}<item>{NS}<label>A{NS}</item>{NS}</opts>'
        f'{NS}</invoke>\n{NS}</tool_call>'
    )
    parsed = _parse_args_via_detector(raw, schema)
    assert parsed == {"opts": [{"label": "A"}]}


def test_array_of_primitives():
    """``<tags><item>a</item><item>b</item></tags>`` → ``["a", "b"]``."""
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }
    raw = (
        f'{NS}<tool_call>\n{NS}<invoke name="f">'
        f'{tag("tags", tag("item", "a") + tag("item", "b"))}'
        f'{NS}</invoke>\n{NS}</tool_call>'
    )
    parsed = _parse_args_via_detector(raw, schema)
    assert parsed == {"tags": ["a", "b"]}
