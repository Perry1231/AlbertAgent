import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
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
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


def calculate(expression):
    return eval(expression, {"__builtins__": {}}, {})


messages = [
    {
        "role": "user",
        "content": "Calculate 25 * 4 using the available tool.",
    }
]

print("Sending request...", flush=True)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    tools=tools,
    tool_choice="auto",
    max_tokens=256,
)

message = response.choices[0].message

print("MODEL RESPONSE:")
print(message)

if message.tool_calls:

    for tool_call in message.tool_calls:

        print("\nTOOL CALL:")
        print(tool_call.function.name)
        print(tool_call.function.arguments)

        arguments = json.loads(tool_call.function.arguments)

        if tool_call.function.name == "calculate":
            result = calculate(arguments["expression"])

            print("TOOL RESULT:", result)

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    final_response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        max_tokens=256,
    )

    print("\nFINAL ANSWER:")
    print(final_response.choices[0].message.content)

else:
    print("\nMODEL DID NOT CALL A TOOL")
    print(message.content)