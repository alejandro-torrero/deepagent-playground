import logging

from dotenv import load_dotenv

from deepagent_playground.agent.agent import Agent

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ORCHESTRATOR_MODEL = "openai:gpt-5-nano"

SUB_AGENT_MODEL = "openai:gpt-5-nano"


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_population(city: str) -> str:
    """Get the population of a given city."""
    return f"{city} has about 9 million people."


def main() -> None:
    agent = Agent(
        "orchestrator",
        model=ORCHESTRATOR_MODEL,
        system_prompt=(
            "You are a coordinator. Delegate weather questions to the "
            "weather_expert and demographic questions to the demographics_expert, "
            "then combine their answers into one short reply."
        ),
    )

    # Configure sub agents before building the agent.
    agent.add_sub_agent(
        name="weather_expert",
        description="Answers questions about the weather in a city.",
        system_prompt="You are a weather expert. Use your tool to answer.",
        tools=[get_weather],
    ).add_sub_agent(
        name="demographics_expert",
        description="Answers questions about the population of a city.",
        system_prompt="You are a demographics expert. Use your tool to answer.",
        tools=[get_population],
    )

    agent.build()

    answer = agent.ask("How is the weather in Mexico City?")
    print(f"\n{answer}")


if __name__ == "__main__":
    main()
