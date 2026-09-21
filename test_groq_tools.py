import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate a simple mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression, for example 25 * 4"
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Calculate 25 * 4 using the calculate tool."
        }
    ],
    tools=tools,
    tool_choice="auto",
    max_tokens=256,
)

print(response)