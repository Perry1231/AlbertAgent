import os

from dotenv import load_dotenv
from smolagents import ToolCallingAgent, OpenAIServerModel

from tools import ALL_TOOLS


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found")


model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=api_key,
    tool_choice="auto",
)


agent = ToolCallingAgent(
    tools=ALL_TOOLS,
    model=model,
    verbosity_level=0,
)


response = agent.run(
    """
    Search the web for information about an AI regulation
    paper submitted to arXiv in June 2022.

    Use the available web search tool.
    """
)


print("\n==============================")
print("ANSWER")
print("==============================")
print(response)