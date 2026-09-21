import os
import json
import time

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

MAX_STEPS = 8
MAX_TOOL_RESULT_CHARS = 8000
MAX_RETRIES = 3


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
        return f"Unknown tool: {name}"

    tool = TOOL_MAP[name]

    print()
    print("=" * 60)
    print(f"EXECUTING TOOL: {name}")
    print(f"ARGUMENTS: {arguments}")
    print("=" * 60)

    try:

        result = tool(**arguments)

        result = str(result)

        # Prevent huge tool outputs from destroying context
        if len(result) > MAX_TOOL_RESULT_CHARS:
            result = (
                result[:MAX_TOOL_RESULT_CHARS]
                + "\n\n[TOOL OUTPUT TRUNCATED]"
            )

        print(f"RESULT: {result}")

        return result

    except Exception as e:

        error = (
            f"Tool execution error: "
            f"{type(e).__name__}: {e}"
        )

        print(error)

        return error


# ==========================================
# GROQ REQUEST WITH RETRY
# ==========================================

def create_completion(messages, tool_schemas):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                max_tokens=800,
            )

        except Exception as e:

            error_text = str(e)

            # Retry rate limits
            if "429" in error_text or "rate_limit" in error_text.lower():

                if attempt == MAX_RETRIES:
                    raise

                wait_time = attempt * 10

                print()
                print(
                    f"Rate limit detected. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

            else:
                raise


# ==========================================
# AGENT
# ==========================================

def run_agent(user_message):

    messages = [
        {
            "role": "system",
            "content": (
                "You are Albert, a helpful AI agent solving GAIA "
                "benchmark tasks.\n\n"

                "Use tools when necessary.\n"
                "Do not call tools unnecessarily.\n\n"

                "For calculations, prefer execute_python_code.\n"
                "For web information, use search and visit_webpage.\n\n"

                "After you have enough information to answer the "
                "question, STOP using tools and provide the final answer.\n\n"

                "Do not continue searching if the answer can already "
                "be calculated.\n\n"

                "Always provide a final answer."
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

        try:

            response = create_completion(
                messages,
                tool_schemas,
            )

        except Exception as e:

            error = (
                f"Agent API error: "
                f"{type(e).__name__}: {e}"
            )

            print()
            print(error)

            return error

        message = response.choices[0].message

        # ------------------------------------------
        # NO TOOL CALL -> FINAL ANSWER
        # ------------------------------------------

        if not message.tool_calls:

            final_answer = message.content or ""

            print()
            print("=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)

            print(final_answer)

            return final_answer

        # ------------------------------------------
        # ADD ASSISTANT TOOL CALL MESSAGE
        # ------------------------------------------

        messages.append({
            "role": "assistant",
            "content": message.content or "",
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
        # EXECUTE TOOL CALLS
        # ------------------------------------------

        for call in message.tool_calls:

            raw_arguments = call.function.arguments

            print()
            print(f"RAW TOOL ARGUMENTS: {raw_arguments}")

            try:

                arguments = json.loads(raw_arguments)

            except json.JSONDecodeError as e:

                result = (
                    "The tool arguments were invalid JSON. "
                    "Do not repeat the same tool call. "
                    "Return valid JSON arguments only. "
                    f"JSON error: {e}"
                )

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

    return (
        "Agent stopped because MAX_STEPS was reached. "
        "No final answer was produced."
    )


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

