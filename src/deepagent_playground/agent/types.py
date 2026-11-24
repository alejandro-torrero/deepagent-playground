from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NotRequired, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

Tool = BaseTool | Callable | dict[str, Any]
Model = str | BaseChatModel


class SubAgentConfig(TypedDict):
    """Configuration for a single sub agent."""

    name: str
    description: str
    system_prompt: NotRequired[str]
    tools: NotRequired[Sequence[Tool]]    
    model: NotRequired[Model]    
    extra: NotRequired[dict[str, Any]]


class OrchestratorConfig(TypedDict):
    """Configuration for the orchestrator agent."""

    name: str
    model: NotRequired[Model]
    system_prompt: NotRequired[str]
    tools: NotRequired[Sequence[Tool]]
    deep_agent_kwargs: NotRequired[dict[str, Any]]


class AgentConfig(TypedDict):
    """Top-level config: the orchestrator plus its sub agents."""

    orchestrator: OrchestratorConfig
    sub_agents: NotRequired[Sequence[SubAgentConfig]]
