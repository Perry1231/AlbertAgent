import os
from smolagents import ToolCallingAgent, OpenAIServerModel
from tools import ALL_TOOLS

model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    tool_choice="auto",
    max_tokens=500,
)

agent = ToolCallingAgent(
    tools=ALL_TOOLS,
    model=model,
    max_steps=3,
    verbosity_level=2,
    return_full_result=True,
)

result = agent.run(
    "Open https://ar5iv.labs.arxiv.org/html/2207.01510 using visit_webpage and tell me the paper title.",
    return_full_result=True,
)

print("\n========== RESULT ==========")
print(result)
