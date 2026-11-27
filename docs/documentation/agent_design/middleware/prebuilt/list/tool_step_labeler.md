# ToolStepLabeler

`ToolStepLabeler` records a short, human-readable label for every completed tool call. Attach it as model middleware with `model_middleware=`. It does not change the messages sent to the agent model or the agent's returned message history.

Use a per-tool template or callback first. A template receives the tool arguments as format fields, plus `_tool_name`, `_tool_call_id`, and `_tool_result`; a callback receives a `ToolStep` with the tool-call identifier, name, copied arguments, and a bounded result excerpt. Supply an optional model for tools without a configured template or callback. The label model runs as a child agent through `rt.call`, so its usage is recorded with the run.

```python
--8<-- "docs/scripts/prebuilt_middleware.py:tool_step_labeler"
```

Each label is emitted as a `tool_label.created` session event with `tool_call_id`, `tool_name`, and `label`. Live observers can display it while the run is active; saved event records expose it through the `tool_label` query namespace. Only one label is generated and recorded for each tool-call identifier within an agent invocation.

Tool names, arguments, and result excerpts are treated as untrusted data in the model prompt. The excerpt passed to the model is bounded by `input_max_chars`. If a template, callback, or model fails, the tool name is emitted as the label. If extracting a tool step or delivering its event fails, the error is logged and the agent continues.
