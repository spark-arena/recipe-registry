import json
from typing import Any, Callable, Dict, Generator, List, Optional

from .data_type import AtomDataType, FunctionCallParameterDataType
from .base import FunctionCallDict
from .generator import (
    GeneratorParser,
    StringByParts,
    default_tool_call_output_key,
    json_dumps,
)


class M3TextParser(GeneratorParser):
    """
    Parser for MiniMax M3 models.

    M3 uses a namespace token `]<]minimax[>[` as delimiter before each tag.
    Parameters use actual XML tag names (not `<parameter name="...">`), and can be nested.
    Complex (nested-XML) arguments are streamed as incremental JSON — brackets
    open as containers open, leaves emit as their closing tag arrives — so
    deeply nested parameters do not stall the SSE stream until the outer close.
    Simple arguments are streamed character-by-character if possible.

    Example raw output::

        ]<]minimax[>[<tool_call>
        ]<]minimax[>[<invoke name="func1">]<]minimax[>[<p1>value1]<]minimax[>[</p1>]<]minimax[>[<p2>]<]minimax[>[<item>]<]minimax[>[<k>val]<]minimax[>[</k>]<]minimax[>[</item>]<]minimax[>[</p2>]<]minimax[>[</invoke>
        ]<]minimax[>[</tool_call>
    """

    def __init__(
        self,
        *,
        with_reasoning: bool = True,
        reasoning_prefix: str = "<mm:think>",
        reasoning_suffix: str = "</mm:think>",
        functions: Optional[Dict] = None,
        tool_call_xml_tag_name: str = "tool_call",
        tool_call_namespace_token: str = "]<]minimax[>[",
        always_nullable: bool = True,
        reasoning_field: str = "reasoning",
        content_field: str = "content",
        tool_call_output_key: Callable[
            [int, Dict], Dict
        ] = default_tool_call_output_key,
    ):
        self.with_reasoning = with_reasoning
        self._reasoning_tokens: Optional[int] = None
        self._reasoning_prefix = reasoning_prefix
        self._reasoning_suffix = reasoning_suffix
        self._reasoning_suffix_without_newline = reasoning_suffix.lstrip()
        self._functions = functions
        self._tool_call_namespace_token = tool_call_namespace_token
        self._tool_call_namespace_token_by_parts = StringByParts(
            parts=[self._tool_call_namespace_token],
        )
        self._tool_call_start_without_ns = f"<{tool_call_xml_tag_name}>"
        self._tool_call_start_with_ns = (
            self._tool_call_namespace_token + self._tool_call_start_without_ns
        )
        self._tool_call_end = (
            f"{self._tool_call_namespace_token}</{tool_call_xml_tag_name}>"
        )
        self._invoke_prefix = f'{self._tool_call_namespace_token}<invoke name="'
        self._invoke_suffix = '">'
        self._end_of_invoke = f"{self._tool_call_namespace_token}</invoke>"
        self._parameter_prefix = f"{self._tool_call_namespace_token}<"
        self._parameter_suffix = f"{self._tool_call_namespace_token}</"
        self._always_nullable = always_nullable
        self._reasoning_field = reasoning_field
        self._content_field = content_field
        self._tool_call_output_key = tool_call_output_key
        super().__init__()

    def _get_function(self, function_name: str) -> Optional[Dict]:
        if isinstance(self._functions, dict):
            return self._functions.get(function_name)

    def count_reasoning_tokens(self) -> Optional[int]:
        if self.with_reasoning:
            if self._reasoning_tokens is None:
                return self._count_consumed_tokens()
            else:
                return self._reasoning_tokens

    def _process_reasoning(self) -> Generator[dict, None, None]:
        yield from self._take_any(
            until=[
                self._reasoning_suffix,
                # NOTE: sometimes the model may start the tool calling without closing the thinking tag.
                # It's a bad case of the model and we have to handle it.
                self._tool_call_namespace_token_by_parts,
            ],
            key=self._reasoning_field,
            should_consume_suffix=False,
        )
        yield from self._literal(self._reasoning_suffix, should_raise=False)
        self._reasoning_tokens = self._count_consumed_tokens()

    def _process(self) -> Generator[dict, None, None]:
        if self.with_reasoning:
            if self._reasoning_prefix:
                with_reasoning = yield from self._literal(
                    self._reasoning_prefix, should_raise=False
                )
                if with_reasoning:
                    yield from self._process_reasoning()
                else:
                    self._reasoning_tokens = 0
                    # If reasoning is disabled, we may find the suffix at the beginning without the prefix.
                    yield from self._literal(
                        self._reasoning_suffix_without_newline,
                        should_raise=False,
                    )
            else:
                yield from self._process_reasoning()

        if not self._functions:
            yield from self._take_any(key=self._content_field)
        else:
            yield from self._take_any(
                until=self._tool_call_namespace_token_by_parts,
                key=self._content_field,
                should_consume_suffix=False,
            )

            # NOTE: Only ONE `<tool_call>` block is supported by design.
            # Multiple parallel calls must share a single wrapper and use
            # multiple `<invoke>` tags inside it. A second `<tool_call>` after
            # the first `</tool_call>` will cause `update()` to raise
            # PatternMismatched (pattern exhausted).
            tool_call_index = 0
            while True:
                # Ignore everything until `{NS}<invoke name="` or `{NS}</tool_call>`.
                yield from self._take_any(
                    until=(self._invoke_prefix, self._tool_call_end),
                    should_consume_suffix=False,
                )
                tried = yield from self._literal(
                    (self._invoke_prefix, self._tool_call_end),
                    should_raise=False,
                )
                if tried == self._tool_call_end:
                    break

                function_name = yield from self._take_any(
                    until=self._invoke_suffix, collect=True
                )
                self._append_delta(
                    self._tool_call_output_key(
                        tool_call_index, {"name": function_name, "arguments": "{"}
                    )
                )
                function = self._get_function(function_name)
                if (
                    isinstance(function, dict)
                    and "parameters" in function
                    and isinstance(function["parameters"], dict)
                    and "properties" in function["parameters"]
                    and isinstance(function["parameters"]["properties"], dict)
                ):
                    parameter_name_set = set(
                        function["parameters"]["properties"].keys()
                    )
                else:
                    parameter_name_set = None
                is_first_parameter = True
                while True:
                    # Ignore everything until `{NS}<`.
                    yield from self._take_any(
                        until=self._parameter_prefix,
                        collect=False,
                    )
                    parameter_name = yield from self._take_any(until=">", collect=True)
                    if parameter_name == "/invoke":
                        if (
                            parameter_name_set is None
                            or parameter_name not in parameter_name_set
                        ):
                            break
                        else:
                            peek_after_eof = yield from self._peek(0)
                            if peek_after_eof == "\n":
                                break
                    should_break = False
                    # Discard all `</tag>`
                    while parameter_name.startswith("/") and (
                        parameter_name_set is None
                        or parameter_name not in parameter_name_set
                    ):
                        if (
                            parameter_name == "/invoke"
                            or parameter_name == "/tool_call"
                        ):
                            should_break = True
                            break
                        # Ignore everything until the next namespace token following `<`.
                        yield from self._take_any(until=self._parameter_prefix)
                        parameter_name = yield from self._take_any(
                            until=">", collect=True
                        )
                    if should_break:
                        break
                    parameter_name_to_arguments = "{}{}: ".format(
                        "" if is_first_parameter else ", ",
                        json_dumps(parameter_name),
                    )
                    self._append_delta(
                        self._tool_call_output_key(
                            tool_call_index,
                            {"arguments": parameter_name_to_arguments},
                        )
                    )
                    parameter_data_type = (
                        FunctionCallParameterDataType.get_schema_of_parameter(
                            function, parameter_name
                        )
                    )
                    tried = yield from self._literal(
                        self._parameter_prefix,
                        should_raise=False,
                        should_consume=False,
                    )
                    if tried is not None:
                        # nested XML -> object/array
                        # NOTE: The namespace token has the highest semantic
                        # priority. Once the model emits `]<]minimax[>[<` here,
                        # we MUST treat the body as nested XML, even if the
                        # schema says this parameter should be a primitive.
                        # The model is asserting "this is a JSON level
                        # transition" — schema mismatches are reported back via
                        # the agent loop, not silently rewritten here.
                        #
                        # The body is streamed incrementally: JSON fragments are
                        # emitted as each child element closes, so deeply nested
                        # parameters (array→object→array→object) do not stall
                        # the SSE ``arguments`` stream until the outer closing
                        # tag.
                        yield from self._stream_nested_parameter(
                            tool_call_index, parameter_name, parameter_data_type
                        )
                    else:
                        # no more nested XML -> string / number / boolean
                        yield from self._take_data_type_as_json(
                            until=self._tool_call_namespace_token_by_parts,
                            key=lambda value: self._tool_call_output_key(
                                tool_call_index, {"arguments": value}
                            ),
                            data_type=parameter_data_type,
                            always_nullable=False,
                            should_consume_suffix=True,
                        )
                        # Ignore everything until the next namespace token.
                        yield from self._take_any(
                            until=self._tool_call_namespace_token_by_parts,
                            should_consume_suffix=False,
                        )
                    is_first_parameter = False
                self._append_delta(
                    self._tool_call_output_key(tool_call_index, {"arguments": "}"})
                )
                tool_call_index += 1

    def _stream_nested_parameter(
        self,
        tool_call_index: int,
        parameter_name: str,
        parameter_data_type: FunctionCallParameterDataType,
    ) -> Generator[dict, None, None]:
        """Stream a nested-XML parameter body as incremental JSON.

        Entered with the input positioned at ``{NS}<`` (the first child tag of
        ``<parameter_name>``). Consumes through the matching
        ``{NS}</parameter_name>`` and emits ``arguments`` deltas as it goes:

        - ``[`` / ``{`` as soon as the first child of a container opens
        - ``"key": `` for each object property as it opens
        - ``, `` between siblings
        - the converted leaf value when a leaf's closing tag arrives
        - ``]`` / ``}`` when a container's closing tag arrives

        This bounds the SSE stall to one leaf, instead of the whole body.

        Model-error edge cases are handled to keep the emitted JSON
        *parseable* and the outer ``_process`` loop in sync:

        - **Bare ``{NS}`` segment** (NS not followed by ``<``): treated as a
          text continuation, never raises.
        - **Missing ``>``** on a tag: tag read is bounded by the next NS so
          subsequent tags are not swallowed.
        - **Mixed content** (text alongside child tags): array parents emit it
          as an extra element in stream order; object parents accumulate it
          and emit one ``"$text"`` property at close.
        - **Unmatched close tag**: force-closes any open *non-root* frames;
          the root is only popped on its own close, so a stray ``</bogus>``
          doesn't terminate the parameter early.
        - **Unclosed children** at the parent's close: force-closed in stack
          order.
        - **Duplicate keys** in an object: emitted as repeated ``"k": v``
          pairs. Downstream ``json.loads`` is last-wins, so earlier values are
          lost — an accepted regression vs the batch parser's
          ``"k": [v1, v2]`` collapse, since streaming cannot retroactively
          rewrite an already-emitted key.
        """

        def emit(s: str) -> None:
            self._append_delta(
                self._tool_call_output_key(tool_call_index, {"arguments": s})
            )

        stack: List[_StreamFrame] = [
            _StreamFrame(tag=parameter_name, data_type=parameter_data_type)
        ]

        while stack:
            # 1. Collect any text up to the next namespace token. For a leaf
            #    frame this is (part of) its value; for a container frame this
            #    is mixed-content text.
            text = yield from self._take_any(
                until=self._tool_call_namespace_token, collect=True
            )
            if text:
                stack[-1].texts.append(text)

            # 2. Read the tag: ``<name>`` or ``</name>``. The NS token was
            #    consumed above; in well-formed output ``<`` follows. If it
            #    doesn't (doubled NS, stray newline segment), loop back and
            #    treat whatever follows as more text — never raise.
            lt = yield from self._literal("<", should_raise=False)
            if lt is None:
                continue
            # Bound the tag read by NS so a missing ``>`` doesn't swallow the
            # next NS-prefixed segment into the tag name.
            tag = yield from self._take_any(
                until=(">", self._tool_call_namespace_token),
                collect=True,
                should_consume_suffix=False,
            )
            yield from self._literal(">", should_raise=False)
            # If that returned None the tag had no ``>`` before the next NS;
            # ``tag`` is still the element name (matches the batch parser's
            # ``gt_offset == -1`` handling).

            if tag.startswith("/"):
                close_tag = tag[1:]
                if close_tag == stack[0].tag:
                    # Root close: force-close any unclosed children, then root.
                    while stack:
                        stack.pop().emit_close(emit)
                    return
                # Non-root close: pop and emit until we've closed the matching
                # frame. Never pop the root — a stray ``</bogus>`` matching
                # nothing is a no-op against the root.
                while len(stack) > 1:
                    frame = stack.pop()
                    frame.emit_close(emit)
                    if frame.tag == close_tag:
                        break
            else:
                parent = stack[-1]
                parent.emit_open_child(tag, emit)
                stack.append(parent.make_child(tag))


    def stringify_function_calls(self, function_calls: List[FunctionCallDict]) -> str:
        if not function_calls:
            return ""
        parts = [self._tool_call_start_with_ns + "\n"]
        for function_call in function_calls:
            parts.append(
                self._invoke_prefix + function_call["name"] + self._invoke_suffix
            )
            arguments = function_call["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            parts.append(self._stringify_parameter(arguments))
            parts.append(self._end_of_invoke + "\n")
        parts.append(self._tool_call_end)
        return "".join(parts)

    def _stringify_parameter(self, parameter: Any) -> str:
        # NOTE: null values are simply ignored.
        # This is limited due to the training philosophy of MiniMax M3.
        # In the training process, the model will NOT see any null values in tool calls.
        # Thus, we have to skip them here, in order to avoid OOD.
        # This cost is unwillingly accepted:
        #     `["a", null, "c"]` becomes `<items>a</items><items>c</items>`,
        #     even the size of the array is changed.
        if isinstance(parameter, dict):
            return "".join(
                f"{self._tool_call_namespace_token}<{key}>{self._stringify_parameter(value)}{self._tool_call_namespace_token}</{key}>"
                for key, value in parameter.items()
                if value is not None
            )
        elif isinstance(parameter, list):
            return "".join(
                f"{self._tool_call_namespace_token}<item>{self._stringify_parameter(value)}{self._tool_call_namespace_token}</item>"
                for value in parameter
                if value is not None
            )
        elif isinstance(parameter, str):
            return parameter
        elif parameter is None:
            # should be unreachable
            return ""
        else:
            return json.dumps(parameter, ensure_ascii=False)


class _StreamFrame:
    """One open element on the :meth:`M3TextParser._stream_nested_parameter`
    stack.

    ``kind`` is decided lazily on the *first child*: if the schema says array
    and the first child is ``<item>`` it becomes ``"array"``, otherwise
    ``"object"``. A frame whose ``kind`` is still ``None`` at close time is a
    leaf — its value is the type-converted concatenation of ``texts``.
    """

    __slots__ = ("tag", "data_type", "kind", "child_count", "texts")

    def __init__(
        self, tag: str, data_type: Optional[FunctionCallParameterDataType]
    ) -> None:
        self.tag = tag
        self.data_type = data_type
        self.kind: Optional[str] = None
        self.child_count = 0
        self.texts: List[str] = []

    def emit_open_child(self, child_tag: str, emit: Callable[[str], None]) -> None:
        """Called when a child ``<child_tag>`` opens under this frame.

        Decides this frame's container kind on the first child and emits the
        opening bracket, then the separator and (for objects) the key. For
        array frames, any mixed-content text accumulated since the last child
        is flushed as an element first so it appears in stream order.
        """
        if self.kind is None:
            if (
                self.data_type
                and AtomDataType.array in self.data_type.candidates
                and child_tag == "item"
            ):
                self.kind = "array"
                emit("[")
            else:
                self.kind = "object"
                emit("{")
        if self.kind == "array" and self.texts:
            self._emit_array_text(emit)
        if self.child_count:
            emit(", ")
        self.child_count += 1
        if self.kind == "object":
            emit(json_dumps(child_tag) + ": ")

    def make_child(self, child_tag: str) -> "_StreamFrame":
        if self.data_type is None:
            child_dt = None
        elif self.kind == "array":
            child_dt = self.data_type.get_data_type_of_item(index=self.child_count - 1)
        else:
            child_dt = self.data_type.get_data_type_of_property(child_tag)
        return _StreamFrame(tag=child_tag, data_type=child_dt)

    def emit_close(self, emit: Callable[[str], None]) -> None:
        if self.kind == "array":
            if self.texts:
                self._emit_array_text(emit)
            emit("]")
        elif self.kind == "object":
            # All mixed-content text (leading, between, trailing) accumulates
            # and is emitted once here as a single ``$text`` property.
            if self.texts:
                if self.child_count:
                    emit(", ")
                emit(json_dumps("$text") + ": " + json_dumps("".join(self.texts)))
            emit("}")
        else:
            # Leaf: convert accumulated text via the schema-derived data type.
            text = "".join(self.texts)
            value = self.data_type.convert(text) if self.data_type else text
            emit(json_dumps(value))

    def _emit_array_text(self, emit: Callable[[str], None]) -> None:
        """Flush accumulated text as one array element (in stream order)."""
        text = "".join(self.texts)
        self.texts = []
        dt = (
            self.data_type.get_data_type_of_item(index=self.child_count)
            if self.data_type
            else None
        )
        value = dt.convert(text) if dt else text
        if self.child_count:
            emit(", ")
        emit(json_dumps(value))
        self.child_count += 1
        self.child_count += 1


