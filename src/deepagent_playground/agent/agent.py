from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from deepagent_playground.agent.constants import DEFAULT_SUB_AGENT_REASONING_EFFORT
from deepagent_playground.agent.logging_middleware import \
    SubAgentLoggingMiddleware
from deepagent_playground.agent.types import Tool


def _resolve_model(model: str | BaseChatModel | None) -> str | BaseChatModel | None:
    """Turn 'openai:<name>' strings into a ChatOpenAI client with low reasoning.

    Other strings and BaseChatModel instances are returned unchanged.
    """
    if isinstance(model, str) and model.startswith("openai:"):
        return ChatOpenAI(
            model=model.removeprefix("openai:"),
            reasoning_effort=DEFAULT_SUB_AGENT_REASONING_EFFORT,
        )
    return model


class Agent:
    """Deep agent orchestrator.

    Usage:
        agent = Agent("assistant", model="openai:gpt-5", system_prompt="...")
        agent.add_sub_agent(
            name="researcher",
            description="Researches topics in depth",
            system_prompt="You are a researcher...",
            tools=[search],
        )
        agent.build()
        agent.invoke("Research X")
    """

    def __init__(
        self,
        name: str,
        model: str | BaseChatModel | None = None,
        system_prompt: str | None = None,
        tools: Sequence[Tool] | None = None,
        **deep_agent_kwargs: Any,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools: list[Tool] = list(tools or [])
        self.deep_agent_kwargs = dict(deep_agent_kwargs)
        self.sub_agents: list[SubAgent] = []
        self._agent: Any | None = None

    @property
    def is_built(self) -> bool:
        return self._agent is not None

    def add_sub_agent(
        self,
        name: str,
        description: str,
        system_prompt: str | None = None,
        tools: Sequence[Tool] | None = None,
        model: str | BaseChatModel | None = None,
        **extra: Any,
    ) -> Agent:
        """Register a subagent configuration. Must be called before `build()`.

        `extra` accepts any other `SubAgent` field (middleware, skills, ...).
        Returns self so calls can be chained.
        """
        if self.is_built:
            raise RuntimeError(
                "Agent is already built; add sub agents before calling build()."
            )
        if any(sa["name"] == name for sa in self.sub_agents):
            raise ValueError(f"Sub agent '{name}' is already registered.")

        config: dict[str, Any] = {"name": name, "description": description, **extra}
        if system_prompt is not None:
            config["system_prompt"] = system_prompt
        if tools is not None:
            config["tools"] = list(tools)
        sub_model = model if model is not None else self.model
        if sub_model is not None:
            config["model"] = _resolve_model(sub_model)

        self.sub_agents.append(config)  # type: ignore[arg-type]
        return self

    def build(self) -> Agent:
        """Create the underlying deep agent with the registered sub agents."""
        kwargs = dict(self.deep_agent_kwargs)
        kwargs["middleware"] = [
            SubAgentLoggingMiddleware(self.name),
            *kwargs.get("middleware", ()),
        ]
        self._agent = create_deep_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            subagents=self.sub_agents or None,
            name=self.name,
            **kwargs,
        )
        return self

    def invoke(
        self,
        message: str | list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a request to the agent and return the full result state.

        `message` can be a plain string or a list of chat messages. Builds the
        agent first if `build()` has not been called yet.
        """
        if not self.is_built:
            self.build()

        messages = (
            [{"role": "user", "content": message}]
            if isinstance(message, str)
            else message
        )
        return self._agent.invoke({"messages": messages}, config=config)

    def ask(self, message: str, config: dict[str, Any] | None = None) -> str:
        """Convenience wrapper around `invoke` returning only the final text."""
        result = self.invoke(message, config=config)
        return result["messages"][-1].text
