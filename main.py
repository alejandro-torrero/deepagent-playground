import logging

from dotenv import load_dotenv

from deepagent_playground.agent.builder import build_agent
from deepagent_playground.agent.types import AgentConfig

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ORCHESTRATOR_MODEL = "openai:gpt-5-nano"

SUB_AGENT_MODEL = "openai:gpt-5-nano"


# Dummy tools
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# Dummy tool
def get_population(city: str) -> str:
    """Get the population of a given city."""
    return f"{city} has about 9 million people."


AGENT_CONFIG: AgentConfig = {
    "orchestrator": {
        "name": "orchestrator",
        "model": ORCHESTRATOR_MODEL,
        "system_prompt": (
            "You are a coordinator. Delegate weather questions to the "
            "weather_expert and demographic questions to the demographics_expert, "
            "then combine their answers into one short reply."
        ),
    },
    "sub_agents": [
        {
            "name": "weather_expert",
            "description": "Answers questions about the weather in a city.",
            "system_prompt": "You are a weather expert. Use your tool to answer.",
            "tools": [get_weather],
        },
        {
            "name": "demographics_expert",
            "description": "Answers questions about the population of a city.",
            "system_prompt": "You are a demographics expert. Use your tool to answer.",
            "tools": [get_population],
        },
    ],
}


def main() -> None:
    agent = build_agent(AGENT_CONFIG)

    answer = agent.ask("How is the weather in Mexico City?")
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
