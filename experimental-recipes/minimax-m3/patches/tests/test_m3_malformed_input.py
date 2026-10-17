"""Model-error tolerance tests for ``_stream_nested_parameter``.

The replaced batch parser (``_parse_parameter`` / ``_StackItem``) was a pure
string-split that degraded gracefully on malformed M3 XML. The streaming
implementation must match that tolerance: malformed input should produce
*some* parseable JSON and keep the outer ``_process`` loop in sync — never
raise, never drain the root frame on a stray close, never swallow NS tokens
into a tag name.
"""
import json

import pytest

from sglang.srt.function_call._llm_nom.m3_text import M3TextParser

from conftest import NS, feed, tokenize_like_m3


PARAMS = {
    "type": "object",
    "properties": {
        "p": {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        },
        "arr": {"type": "array", "items": {"type": "string"}},
        "obj": {
            "type": "object",
            "properties": {
                "o": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                }
            },
        },
    },
}


def _wrap(body: str) -> str:
    return f'{NS}<tool_call>\n{NS}<invoke name="f">{body}{NS}</invoke>\n{NS}</tool_call>'


def _parse(raw: str) -> tuple[dict | None, Exception | None]:
    p = M3TextParser(
        with_reasoning=False, reasoning_prefix="", functions={"f": {"parameters": PARAMS}}
    )
    err = None
    try:
        for c in tokenize_like_m3(raw):
            p.update(c)
    except Exception as e:  # noqa: BLE001
        err = e
    final = p.get_final()
    args = None
    if final and "tool_calls" in final:
        args = json.loads(final["tool_calls"][0]["function"]["arguments"])
    return args, err


# -- #2: bare-NS segment mid-body must not raise ------------------------------


def test_bare_ns_segment_mid_body_is_tolerated():
    """``…</a>{NS}\\n{NS}<b>…`` — a non-``<`` chunk after NS inside a nested
    body is text, not a fatal error. Old parser routed it to ``append_text``."""
    raw = _wrap(f"{NS}<p>{NS}<a>v{NS}</a>{NS}\n{NS}<b>w{NS}</b>{NS}</p>")
    args, err = _parse(raw)
    assert err is None, f"must not raise; got {type(err).__name__}: {err}"
    assert args["p"]["a"] == "v"
    assert args["p"]["b"] == "w"


def test_doubled_ns_is_tolerated():
    raw = _wrap(f"{NS}<p>{NS}<a>v{NS}</a>{NS}{NS}<b>w{NS}</b>{NS}</p>")
    args, err = _parse(raw)
    assert err is None, f"must not raise; got {type(err).__name__}: {err}"
    assert args["p"] == {"a": "v", "b": "w"}


# -- #4: unmatched close tag must not drain the root --------------------------


def test_unmatched_close_does_not_drain_root():
    """``</bogus>`` matching nothing on the stack must be ignored; the root
    frame stays open so ``<b>`` lands inside ``p``, not as a new top-level
    parameter."""
    raw = _wrap(
        f"{NS}<p>{NS}<a>1{NS}</a>{NS}</bogus>{NS}<b>2{NS}</b>{NS}</p>"
    )
    args, err = _parse(raw)
    assert err is None
    assert args == {"p": {"a": "1", "b": "2"}}, f"got {args}"


# -- #8: tag read must be NS-bounded -----------------------------------------


def test_missing_gt_does_not_swallow_ns_into_tag():
    """``{NS}<bad{NS}<a>…`` — model dropped ``>``. Tag read must stop at the
    next NS, not swallow it. The malformed ``bad`` tag may be opened (and end
    up as an extra key) but the well-formed ``<a>`` that follows must still be
    parsed correctly under ``p``."""
    raw = _wrap(f"{NS}<p>{NS}<bad{NS}<a>v{NS}</a>{NS}</p>")
    args, err = _parse(raw)
    assert err is None
    # NS bytes must not appear anywhere in the emitted args.
    assert "minimax" not in json.dumps(args), f"NS leaked into args: {args}"
    # The well-formed child survives.
    assert "p" in args


# -- #6: text between/after array items becomes elements ----------------------


def test_text_between_array_items_becomes_element():
    raw = _wrap(
        f"{NS}<arr>{NS}<item>a{NS}</item>STRAY{NS}<item>b{NS}</item>{NS}</arr>"
    )
    args, err = _parse(raw)
    assert err is None
    assert args == {"arr": ["a", "STRAY", "b"]}, f"got {args}"


def test_trailing_text_in_array_becomes_element():
    raw = _wrap(f"{NS}<arr>{NS}<item>a{NS}</item>TAIL{NS}</arr>")
    args, err = _parse(raw)
    assert err is None
    assert args == {"arr": ["a", "TAIL"]}, f"got {args}"


# -- #7: object mixed-content text emits one $text ----------------------------


def test_object_mixed_content_emits_single_text_key():
    """Leading + trailing text around children of an object → one ``$text``
    key with both joined (matching the batch parser)."""
    raw = _wrap(
        f"{NS}<obj>{NS}<o>LEAD{NS}<x>1{NS}</x>TRAIL{NS}</o>{NS}</obj>"
    )
    args, err = _parse(raw)
    assert err is None
    o = args["obj"]["o"]
    assert o["x"] == "1"
    assert o.get("$text") == "LEADTRAIL", f"got {o}"
    # And only one $text key in the raw JSON.
    p = M3TextParser(
        with_reasoning=False, reasoning_prefix="", functions={"f": {"parameters": PARAMS}}
    )
    p.update(raw)
    raw_args = p.get_final()["tool_calls"][0]["function"]["arguments"]
    assert raw_args.count('"$text"') == 1, f"duplicate $text keys: {raw_args}"
