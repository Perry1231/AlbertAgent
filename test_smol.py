import os
from dotenv import load_dotenv
from smolagents import OpenAIServerModel

load_dotenv()

model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    tool_choice="none",
)

messages = [
    {
        "role": "user",
        "content": """Return ONLY this Python code and nothing else:

print("HELLO FROM CODE")
"""
    }
]

result = model.generate(messages)

print("CONTENT:")
print(result.content)

print("\nTOOL CALLS:")
print(result.tool_calls)