import os
from dotenv import load_dotenv

print("1. START", flush=True)

from smolagents import ToolCallingAgent, OpenAIServerModel, DuckDuckGoSearchTool
print("2. smolagents imported", flush=True)

from tools import ALL_TOOLS
print(f"3. ALL_TOOLS loaded: {len(ALL_TOOLS)}", flush=True)

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
    temperature=0.2,
    max_tokens=512,
)

print("6. Model created", flush=True)

print("7. Creating tools...", flush=True)

search_tool = DuckDuckGoSearchTool()

tools = [search_tool] + ALL_TOOLS

print("NUMBER OF TOOLS:", len(tools), flush=True)
print("TOOLS:", [tool.name for tool in tools], flush=True)

print("8. Creating ToolCallingAgent...", flush=True)

agent = ToolCallingAgent(
    model=model,
    tools=tools,
    max_steps=3,
    verbosity_level=2,
)

print("9. Agent created", flush=True)

print("MAX STEPS:", agent.max_steps, flush=True)

print("AVAILABLE AGENT TOOLS:")
print(list(agent.tools.keys()))
print()

print("10. Sending request to agent...", flush=True)

response = agent.run(
    'Search the web for "AI regulation arXiv June 2022" and give me 3 short results.'
)

print("11. Agent finished", flush=True)

print("==============================")
print("ANSWER")
print("==============================")
print(response)