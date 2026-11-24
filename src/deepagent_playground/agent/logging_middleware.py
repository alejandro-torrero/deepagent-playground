from __future__ import annotations

import logging
import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware

from deepagent_playground.utils.string_utils import truncate

logger = logging.getLogger(__name__)

# Deep agents delegate to sub agents through this tool.
SUBAGENT_TOOL_NAME = "task"


class SubAgentLoggingMiddleware(AgentMiddleware):
    """Logs every time the orchestrator delegates work to a sub agent."""

    def __init__(self, agent_name: str):
        super().__init__()
        self.agent_name = agent_name

    def _log_start(self, request: Any) -> float | None:
        call = request.tool_call
        if call.get("name") != SUBAGENT_TOOL_NAME:
            return None
        args = call.get("args", {})
        logger.info(
            "[%s] sub agent requested: %s | task: %s",
            self.agent_name,
            args.get("subagent_type"),
            truncate(args.get("description", "")),
        )
        return time.perf_counter()

    def _log_end(self, request: Any, started: float) -> None:
        logger.info(
            "[%s] sub agent finished: %s (%.2fs)",
            self.agent_name,
            request.tool_call.get("args", {}).get("subagent_type"),
            time.perf_counter() - started,
        )

    def wrap_tool_call(self, request, handler):
        started = self._log_start(request)
        result = handler(request)
        if started is not None:
            self._log_end(request, started)
        return result

    async def awrap_tool_call(self, request, handler):
        started = self._log_start(request)
        result = await handler(request)
        if started is not None:
            self._log_end(request, started)
        return result
