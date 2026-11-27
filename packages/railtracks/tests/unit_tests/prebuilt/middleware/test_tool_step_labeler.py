import json
from unittest.mock import AsyncMock, Mock

import pytest
from railtracks.llm import (
    AssistantMessage,
    MessageHistory,
    Response,
    ToolCall,
    ToolMessage,
    ToolResponse,
    UserMessage,
)
from railtracks.prebuilt.middleware import ToolStepLabeler
from railtracks.prebuilt.middleware import tool_step_labeler as module


def _history(*results: str) -> MessageHistory:
    return MessageHistory(
        [
            UserMessage("Find the note."),
            AssistantMessage(
                [
                    ToolCall(
                        identifier="call-1",
                        name="search_notes",
                        arguments={"query": "rate limits"},
                    )
                ]
            ),
            *[
                ToolMessage(
                    ToolResponse(
                        identifier="call-1", name="search_notes", result=result
                    )
                )
                for result in results
            ],
        ]
    )


@pytest.fixture
def emitted_labels(monkeypatch):
    events = []

    async def emit(event):
        events.append(event)

    monkeypatch.setattr(module, "emit", emit)
    return events


@pytest.mark.asyncio
async def test_template_label_preserves_model_history_and_response(emitted_labels):
    middleware = ToolStepLabeler(
        templates={
            "search_notes": "Searched for {query}: {_tool_result} ({_tool_call_id})"
        }
    )
    history = _history("Found eight results.")
    original = list(history)
    response = Response(AssistantMessage("done"))
    call = AsyncMock(return_value=response)

    result = await middleware._middleware_fn(call, history, None, None)

    assert result is response
    assert history == original
    assert all(actual is expected for actual, expected in zip(history, original))
    call.assert_awaited_once_with(history, None, None)
    assert len(emitted_labels) == 1
    event = emitted_labels[0]
    assert event.tool_call_id == "call-1"
    assert event.tool_name == "search_notes"
    assert event.label == "Searched for rate limits: Found eight results. (call-1)"


@pytest.mark.asyncio
async def test_unconfigured_tool_uses_its_name_as_the_default_label(emitted_labels):
    middleware = ToolStepLabeler()
    history = _history("Found eight results.")
    call = AsyncMock(return_value=Response(AssistantMessage("done")))

    await middleware._middleware_fn(call, history, None, None)

    assert emitted_labels[0].label == "search_notes"
    call.assert_awaited_once_with(history, None, None)


@pytest.mark.asyncio
async def test_duplicate_tool_call_identifiers_are_labelled_once(emitted_labels):
    middleware = ToolStepLabeler(
        templates={"search_notes": lambda step: step.arguments["query"]}
    )
    history = _history("first result", "second result")
    call = AsyncMock(return_value=Response(AssistantMessage("done")))

    await middleware._middleware_fn(call, history, None, None)

    assert len(emitted_labels) == 1
    assert emitted_labels[0].label == "rate limits"
    call.assert_awaited_once()


@pytest.mark.asyncio
async def test_model_fallback_bounds_and_marks_untrusted_input(
    mock_llm, monkeypatch, emitted_labels
):
    model = mock_llm()
    middleware = ToolStepLabeler(model, input_max_chars=20)
    label_call = AsyncMock(return_value=Mock(content="  Searched docs\n"))
    monkeypatch.setattr(module, "call_node", label_call)
    history = _history("0123456789abcdefghijklmnopqrstuvwxyz")
    call = AsyncMock(return_value=Response(AssistantMessage("done")))

    await middleware._middleware_fn(call, history, None, None)

    label_call.assert_awaited_once()
    prompt = label_call.await_args.args[1]
    assert "untrusted tool-call data" in prompt.casefold()
    payload = json.loads(prompt.split("\n", 1)[1])
    assert len(payload["arguments_excerpt"]) + len(payload["result_excerpt"]) <= 20
    assert "search_notes" == payload["tool_name"]
    assert emitted_labels[0].label == "Searched docs"
    call.assert_awaited_once_with(history, None, None)


@pytest.mark.asyncio
async def test_model_failure_uses_plain_label_and_does_not_fail_agent(
    mock_llm, monkeypatch, emitted_labels
):
    middleware = ToolStepLabeler(mock_llm())
    label_call = AsyncMock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr(module, "call_node", label_call)
    history = _history("result")
    response = Response(AssistantMessage("done"))
    call = AsyncMock(return_value=response)

    result = await middleware._middleware_fn(call, history, None, None)

    assert result is response
    assert emitted_labels[0].label == "search_notes"
    call.assert_awaited_once_with(history, None, None)


@pytest.mark.asyncio
async def test_event_delivery_failure_does_not_fail_agent(monkeypatch):
    middleware = ToolStepLabeler(templates={"search_notes": "Searched for {query}"})
    monkeypatch.setattr(module, "emit", AsyncMock(side_effect=RuntimeError("offline")))
    history = _history("Found eight results.")
    response = Response(AssistantMessage("done"))
    call = AsyncMock(return_value=response)

    result = await middleware._middleware_fn(call, history, None, None)

    assert result is response
    call.assert_awaited_once_with(history, None, None)


@pytest.mark.asyncio
async def test_template_failure_falls_back_without_calling_model(
    mock_llm, monkeypatch, emitted_labels
):
    label_call = AsyncMock()
    monkeypatch.setattr(module, "call_node", label_call)
    middleware = ToolStepLabeler(
        mock_llm(),
        templates={"search_notes": lambda _step: (_ for _ in ()).throw(ValueError())},
    )
    history = _history("result")
    call = AsyncMock(return_value=Response(AssistantMessage("done")))

    await middleware._middleware_fn(call, history, None, None)

    assert emitted_labels[0].label == "search_notes"
    label_call.assert_not_awaited()
    call.assert_awaited_once_with(history, None, None)


@pytest.mark.parametrize("input_max_chars", [0, -1])
def test_rejects_non_positive_input_limit(input_max_chars):
    with pytest.raises(ValueError, match="input_max_chars"):
        ToolStepLabeler(input_max_chars=input_max_chars)


@pytest.mark.parametrize("label_max_chars", [0, -1])
def test_rejects_non_positive_label_limit(label_max_chars):
    with pytest.raises(ValueError, match="label_max_chars"):
        ToolStepLabeler(label_max_chars=label_max_chars)


def test_requires_template_to_be_a_string_or_callable():
    with pytest.raises(TypeError, match="string or callable"):
        ToolStepLabeler(templates={"search_notes": 3})
