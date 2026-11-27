from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from railtracks.context.scope_link import ScopeLink
from railtracks.context.session_context import ScopeEntry
from railtracks.events._base import (
    LLMAndMiddlewareSpatialParent,
    MiddlewareParent,
    ParentEventBase,
)
from railtracks.events._resolve import (
    middleware_parent,
    model_middleware_spatial_parent,
)


@dataclass(kw_only=True)
class ToolStepLabelEvent(
    ParentEventBase[LLMAndMiddlewareSpatialParent, MiddlewareParent]
):
    """A human-readable label attached to one completed tool call."""

    tool_call_id: str
    tool_name: str
    label: str

    def _get_spatial_parent(
        self, scope: ScopeLink[ScopeEntry] | None
    ) -> LLMAndMiddlewareSpatialParent:
        return cast(
            LLMAndMiddlewareSpatialParent, model_middleware_spatial_parent(scope)
        )

    def _get_parent(self, scope: ScopeLink[ScopeEntry] | None) -> MiddlewareParent:
        return cast(MiddlewareParent, middleware_parent(scope))

    def event_type(self) -> str:
        return "tool_label.created"
