import logging
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from deepagent_playground.agent.builder import build_agent
from deepagent_playground.agent.types import AgentConfig

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

MODEL = "openai:gpt-5-nano"

EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit"}
NEW_COMMAND = "/new"

AGENT_CONFIG: AgentConfig = {
    "orchestrator": {
        "name": "assistant",
        "model": MODEL,
        "system_prompt": "You are a helpful assistant. Keep answers concise.",        
        "deep_agent_kwargs": {"checkpointer": InMemorySaver()},
    },
}


def _new_thread_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def chat() -> None:
    agent = build_agent(AGENT_CONFIG)
    config = _new_thread_config()

    print("Chat started. Type 'exit' to quit, '/new' to start a new conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            print("Bye!")
            break
        if user_input.lower() == NEW_COMMAND:
            config = _new_thread_config()
            print("-- New conversation started --\n")
            continue

        try:
            answer = agent.ask(user_input, config=config)
        except KeyboardInterrupt:
            print("\n-- Interrupted --\n")
            continue
        except Exception as exc:  # keep the chat alive on API/tool errors
            print(f"Error: {exc}\n")
            continue

        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    chat()
