from smolagents import OpenAIServerModel
from tools import ALL_TOOLS

model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key="test",
    tool_choice="auto",
    max_tokens=500,
)

kwargs = model._prepare_completion_kwargs(
    messages=[
        {
            "role": "user",
            "content": "Say hello"
        }
    ],
    tools_to_call_from=ALL_TOOLS,
    model=model.model_id,
)

print("\n========== PREPARED GROQ REQUEST ==========")
print("tool_choice:", kwargs.get("tool_choice"))
print("model:", kwargs.get("model"))
print("tools count:", len(kwargs.get("tools", [])))
print("max_tokens:", kwargs.get("max_tokens"))
print("============================================")

print("\nTool names:")
for tool in kwargs.get("tools", []):
    print(" -", tool["function"]["name"])