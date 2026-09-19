import os
from dotenv import load_dotenv

print("1. START", flush=True)

from smolagents import ToolCallingAgent, OpenAIServerModel
print("2. smolagents imported", flush=True)

from tools import ALL_TOOLS
print(f"3. Tools loaded: {len(ALL_TOOLS)}", flush=True)

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found")

print("4. API key found", flush=True)

print("5. Creating OpenAIServerModel...", flush=True)

model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=api_key,
    tool_choice="auto",
)

print("6. Model created", flush=True)

print("7. Creating ToolCallingAgent...", flush=True)

agent = ToolCallingAgent(
    tools=ALL_TOOLS,
    model=model,
    verbosity_level=1,
)
print("AVAILABLE AGENT TOOLS:")
print(list(agent.tools.keys()))
print()
print("8. Agent created", flush=True)

print("9. Sending request to agent...", flush=True)

response = agent.run(
    """
    Use the available web search tool to search for:
    "AI regulation arXiv June 2022"

    Return the search results briefly.
    """
)

print("10. Agent finished", flush=True)

print("==============================")
print("ANSWER")
print("==============================")
print(response)