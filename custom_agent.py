import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools import ALL_TOOLS


# ==========================================
# CONFIG
# ==========================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found")


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
)

MODEL = "openai/gpt-oss-120b"

MAX_STEPS = 6


# ==========================================
# CONVERT smolagents TOOLS -> OPENAI TOOLS
# ==========================================

def build_tool_schemas(tools):

    schemas = []

    for tool in tools:

        schemas.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.inputs,
                    "required": list(tool.inputs.keys()),
                },
            },
        })

    return schemas


# ==========================================
# TOOL DISPATCHER
# ==========================================

TOOL_MAP = {
    tool.name: tool
    for tool in ALL_TOOLS
}


def execute_tool(name, arguments):

    if name not in TOOL_MAP:
        raise RuntimeError(f"Unknown tool: {name}")

    tool = TOOL_MAP[name]

    print()
    print("=" * 60)
    print(f"EXECUTING TOOL: {name}")
    print(f"ARGUMENTS: {arguments}")
    print("=" * 60)

    try:

        result = tool(**arguments)

        print(f"RESULT: {result}")

        return result

    except Exception as e:

        error = f"Tool execution error: {type(e).__name__}: {e}"

        print(error)

        return error


# ==========================================
# AGENT
# ==========================================

def run_agent(user_message):

    messages = [
        {
            "role": "system",
            "content": (
                "You are Albert, a helpful AI agent. "
                "Use the available tools when they are useful. "
                "Do not call tools unnecessarily. "
                "After receiving tool results, continue reasoning "
                "and provide a concise final answer."
            ),
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    tool_schemas = build_tool_schemas(ALL_TOOLS)

    for step in range(1, MAX_STEPS + 1):

        print()
        print("#" * 60)
        print(f"AGENT STEP {step}/{MAX_STEPS}")
        print("#" * 60)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",
            max_tokens=1024,
        )

        message = response.choices[0].message

        # ------------------------------------------
        # NO TOOL CALL -> FINAL ANSWER
        # ------------------------------------------

        if not message.tool_calls:

            print()
            print("=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)

            print(message.content)

            return message.content

        # ------------------------------------------
        # ADD ASSISTANT TOOL CALL MESSAGE
        # ------------------------------------------

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ],
        })

        # ------------------------------------------
        # EXECUTE ALL TOOL CALLS
        # ------------------------------------------

        for call in message.tool_calls:

            try:
                arguments = json.loads(call.function.arguments)

            except json.JSONDecodeError as e:

                result = f"Invalid tool arguments JSON: {e}"

            else:

                result = execute_tool(
                    call.function.name,
                    arguments,
                )

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return "Agent stopped because MAX_STEPS was reached."


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("ALBERT AGENT")
    print("=" * 60)

    print(f"Model: {MODEL}")
    print(f"Tools: {len(ALL_TOOLS)}")
    print(f"Max steps: {MAX_STEPS}")

    print()
    print("Available tools:")

    for tool in ALL_TOOLS:
        print(f" - {tool.name}")

    print()
    print("=" * 60)

    user_input = input("You: ")

    run_agent(user_input)