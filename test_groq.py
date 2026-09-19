import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",

    messages=[
        {
            "role": "user",
            "content": (
                "Search the web for the paper about AI regulation "
                "originally submitted to arXiv in June 2022."
            ),
        }
    ],

    tools=tools,

    tool_choice="auto",
)

print(response)