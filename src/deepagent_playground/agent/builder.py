from __future__ import annotations

from deepagent_playground.agent.agent import Agent
from deepagent_playground.agent.types import AgentConfig


def build_agent(agent_config: AgentConfig) -> Agent:
    """Create a built `Agent` from a config with an orchestrator and sub agents.

    Example:
        agent = build_agent({
            "orchestrator": {
                "name": "assistant",
                "model": "openai:gpt-5",
                "system_prompt": "You coordinate the team.",
            },
            "sub_agents": [
                {
                    "name": "researcher",
                    "description": "Researches topics in depth",
                    "system_prompt": "You are a researcher...",
                },
            ],
        })
        agent.ask("Research X")
    """
    orchestrator = agent_config["orchestrator"]

    agent = Agent(
        name=orchestrator["name"],
        model=orchestrator.get("model"),
        system_prompt=orchestrator.get("system_prompt"),
        tools=orchestrator.get("tools"),
        **orchestrator.get("deep_agent_kwargs", {}),
    )

    for sub_agent in agent_config.get("sub_agents", ()):
        agent.add_sub_agent(
            name=sub_agent["name"],
            description=sub_agent["description"],
            system_prompt=sub_agent.get("system_prompt"),
            tools=sub_agent.get("tools"),
            model=sub_agent.get("model"),
            **sub_agent.get("extra", {}),
        )

    return agent.build()
