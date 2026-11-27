import json

import railtracks as rt
from railtracks.llm import AssistantMessage, Response, ToolCall, ToolMessage
from railtracks.observability import JsonlWriter, configure_writers
from railtracks.observability import configure as observability_configure
from railtracks.prebuilt.middleware import ToolStep, ToolStepLabeler


async def test_labels_are_live_recorded_once_and_leave_agent_output_unchanged(
    mock_llm, tmp_path
):
    @rt.function_node
    def search_notes(query: str) -> str:
        """Search notes for a phrase.

        Args:
            query: Phrase to search for.

        Returns:
            A short search result.
        """
        return f"Found eight results for {query}."

    agent_model = mock_llm()
    tool_results_seen = []

    def agent_chat(messages, tools, **kwargs):
        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        tool_results_seen.append([message.text for message in tool_messages])
        if len(tool_messages) < 2:
            return Response(
                AssistantMessage(
                    [
                        ToolCall(
                            identifier="same-call-id",
                            name="search_notes",
                            arguments={"query": "rate limits"},
                        )
                    ]
                )
            )
        return Response(AssistantMessage("done"))

    async def agent_achat(messages, tools, **kwargs):
        return agent_chat(messages, tools, **kwargs)

    agent_model._chat_with_tools = agent_chat
    agent_model._achat_with_tools = agent_achat
    callback_steps: list[ToolStep] = []

    def label_search(step: ToolStep) -> str:
        callback_steps.append(step)
        return f"Searched notes for {step.arguments['query']}"

    label_events = []
    observability_configure.add_inline_listener(
        lambda event: label_events.append(event)
        if event.event_type == "tool_label.created"
        else None
    )
    events_dir = tmp_path / "events"
    configure_writers([JsonlWriter(events_dir)])

    agent = rt.agent_node(
        "labelled-search",
        llm=agent_model,
        tool_nodes=[search_notes],
        model_middleware=[ToolStepLabeler(templates={"search_notes": label_search})],
    )
    result = await rt.Flow("labelled-search", agent).ainvoke("Search for rate limits")

    assert result.text == "done"
    assert tool_results_seen == [
        [],
        ["Found eight results for rate limits."],
        [
            "Found eight results for rate limits.",
            "Found eight results for rate limits.",
        ],
    ]
    returned_tool_results = [
        message.text
        for message in result.message_history
        if isinstance(message, ToolMessage)
    ]
    assert returned_tool_results == [
        "Found eight results for rate limits.",
        "Found eight results for rate limits.",
    ]
    assert len(callback_steps) == 1
    assert len(label_events) == 1
    assert label_events[0].payload["tool_call_id"] == "same-call-id"
    assert label_events[0].payload["tool_name"] == "search_notes"
    assert label_events[0].payload["label"] == "Searched notes for rate limits"

    saved_events = [
        json.loads(line)
        for path in events_dir.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    recorded_labels = [
        event for event in saved_events if event["event_type"] == "tool_label.created"
    ]
    assert len(recorded_labels) == 1
    assert recorded_labels[0]["payload"]["tool_call_id"] == "same-call-id"
    assert recorded_labels[0]["payload"]["label"] == "Searched notes for rate limits"


async def test_model_fallback_cost_is_recorded_with_the_run(mock_llm, tmp_path):
    @rt.function_node
    def lookup_note(query: str) -> str:
        """Look up one note.

        Args:
            query: Phrase to look up.

        Returns:
            The matching note.
        """
        return f"Found a note about {query}."

    agent_model = mock_llm()

    def agent_chat(messages, tools, **kwargs):
        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        if not tool_messages:
            return Response(
                AssistantMessage(
                    [
                        ToolCall(
                            identifier="lookup-call",
                            name="lookup_note",
                            arguments={"query": "scope"},
                        )
                    ]
                )
            )
        return Response(AssistantMessage("done"))

    async def agent_achat(messages, tools, **kwargs):
        return agent_chat(messages, tools, **kwargs)

    agent_model._chat_with_tools = agent_chat
    agent_model._achat_with_tools = agent_achat
    label_model = mock_llm()
    label_model_inputs = []

    def label_chat(messages, **kwargs):
        label_model_inputs.append(messages)
        return Response(
            AssistantMessage("Looked up the note"),
            message_info=label_model.mocked_message_info,
        )

    async def label_achat(messages, **kwargs):
        return label_chat(messages, **kwargs)

    label_model._chat = label_chat
    label_model._achat = label_achat
    live_labels = []
    observability_configure.add_inline_listener(
        lambda event: live_labels.append(event)
        if event.event_type == "tool_label.created"
        else None
    )
    events_dir = tmp_path / "events"
    configure_writers([JsonlWriter(events_dir)])

    agent = rt.agent_node(
        "model-labelled-search",
        llm=agent_model,
        tool_nodes=[lookup_note],
        model_middleware=[ToolStepLabeler(label_model)],
    )
    result = await rt.Flow("model-labelled-search", agent).ainvoke("Look up scope")

    assert result.text == "done"
    assert [event.payload["label"] for event in live_labels] == ["Looked up the note"]
    assert len(label_model_inputs) == 1
    assert "untrusted data" in label_model_inputs[0][0].text
    assert '"tool_name": "lookup_note"' in label_model_inputs[0][1].text
    assert "Found a note about scope." in label_model_inputs[0][1].text

    saved_events = [
        json.loads(line)
        for path in events_dir.glob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    labeler_nodes = [
        event
        for event in saved_events
        if event["event_type"] == "node.creation"
        and event["payload"].get("name") == "ToolStepLabeler"
    ]
    assert len(labeler_nodes) == 1
    labeler_node_id = labeler_nodes[0]["payload"]["node_id"]
    labeler_responses = [
        event
        for event in saved_events
        if event["event_type"] == "llm.response"
        and event["payload"].get("spatial_parent_node_id") == labeler_node_id
    ]
    assert len(labeler_responses) == 1
    assert labeler_responses[0]["payload"]["total_cost"] == 0.00042
