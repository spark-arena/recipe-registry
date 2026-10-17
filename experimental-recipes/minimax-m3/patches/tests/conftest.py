"""Shared fixtures and helpers for the M3 parser streaming tests."""
import json

NS = "]<]minimax[>["


def tag(name: str, body: str) -> str:
    """Wrap body in a namespace-tokened XML tag."""
    return f"{NS}<{name}>{body}{NS}</{name}>"


# OpenCode's `question` tool schema (array→object→array→object).
QUESTION_PARAMS = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "header": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                    "multiple": {"type": "boolean"},
                },
            },
        }
    },
}

QUESTION_FUNCTIONS = {"question": {"parameters": QUESTION_PARAMS}}


def build_question_call(questions: list) -> tuple[str, dict]:
    """Build a (raw_m3_text, expected_json_args) pair for a `question` tool call.

    ``questions`` is the desired ``arguments["questions"]`` value.
    """
    q_items = ""
    for q in questions:
        opt_items = "".join(
            tag("item", tag("label", o["label"]) + tag("description", o["description"]))
            for o in q["options"]
        )
        body = (
            tag("question", q["question"])
            + tag("header", q["header"])
            + tag("options", opt_items)
        )
        if "multiple" in q:
            body += tag("multiple", json.dumps(q["multiple"]))
        q_items += tag("item", body)
    raw = (
        f'{NS}<tool_call>\n'
        f'{NS}<invoke name="question">{tag("questions", q_items)}{NS}</invoke>\n'
        f'{NS}</tool_call>'
    )
    return raw, {"questions": questions}


def tokenize_like_m3(raw: str) -> list[str]:
    """Split raw text into chunks the way the M3 tokenizer would deliver them.

    The namespace token is a single special token, so it always arrives whole.
    Everything between namespace tokens is delivered in small (~4 char) pieces
    to simulate per-token streaming of regular BPE tokens.
    """
    chunks = []
    for i, seg in enumerate(raw.split(NS)):
        if i > 0:
            chunks.append(NS)
        for j in range(0, len(seg), 4):
            chunks.append(seg[j : j + 4])
    return [c for c in chunks if c]


def feed(parser, chunks: list[str]) -> list[dict]:
    """Feed chunks to a parser and collect non-empty deltas."""
    deltas = []
    for c in chunks:
        parser.update(c)
        d = parser.get_delta()
        if d:
            deltas.append(d)
    return deltas


def collect_args(deltas: list[dict], index: int = 0) -> str:
    """Reassemble the streamed ``arguments`` string for one tool call index."""
    out = []
    for d in deltas:
        for tc in d.get("tool_calls", []):
            if tc.get("index") == index:
                a = tc.get("function", {}).get("arguments")
                if a:
                    out.append(a)
    return "".join(out)
