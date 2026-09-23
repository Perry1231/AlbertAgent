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
MAX_TOOL_RESULT_CHARS = 2500
MAX_RETRIES = 2


# ==========================================
# TOOL SCHEMAS
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
# TOOL MAP
# ==========================================

TOOL_MAP = {
    tool.name: tool
    for tool in ALL_TOOLS
}

SEARCH_MAX_CALLS = 2

# ==========================================
# TOOL EXECUTION
# ==========================================

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

        if len(result) > MAX_TOOL_RESULT_CHARS:

            result = (
                result[:MAX_TOOL_RESULT_CHARS]
                + "\n\n[TOOL OUTPUT TRUNCATED]"
            )

        print("RESULT:")
        print(result)

        return result

    except Exception as e:

        error = (
            f"Tool execution error: "
            f"{type(e).__name__}: {e}"
        )

        print(error)

        return error


# ==========================================
# GROQ REQUEST
# ==========================================

def create_completion(messages, tool_schemas):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                max_tokens=400,
            )

        except Exception as e:

            error_text = str(e).lower()

            # ==================================
            # GROQ DAILY TOKEN LIMIT
            # ==================================

            if (
                "tokens per day" in error_text
                or "tpd" in error_text
            ):

                print()
                print("=" * 60)
                print("GROQ DAILY TOKEN LIMIT REACHED")
                print("=" * 60)
                print(e)

                raise

            # ==================================
            # TEMPORARY RATE LIMIT
            # ==================================

            if (
                "429" in error_text
                or "rate_limit" in error_text
            ):

                if attempt == MAX_RETRIES:
                    raise

                wait_time = attempt * 5

                print()
                print(
                    f"Temporary rate limit. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

                continue

            # ==================================
            # OUTPUT PARSE ERROR
            # ==================================

            if "output_parse_failed" in error_text:

                print()
                print("=" * 60)
                print("GROQ OUTPUT PARSING ERROR")
                print("=" * 60)
                print(e)

                if attempt == MAX_RETRIES:
                    raise

                # Add an instruction for the next attempt.
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed. "
                        "Do not write reasoning or planning text. "
                        "If a tool returns enough information to "
                        "solve the task, calculate the answer and "
                        "finish immediately.\n\n"

                        "If a search result contains the required "
                        "numerical value, do not search again. Use "
                        "execute_python_code for the calculation if needed.\n\n"
                    ),
                })

                continue

            # ==================================
            # OTHER API ERROR
            # ==================================

            raise


# ==========================================
# AGENT
# ==========================================

def run_agent(user_message):

    messages = [

        {
            "role": "system",
            "content": (
                "You are Albert, an efficient AI agent solving "
                "GAIA benchmark tasks.\n\n"

                "Rules:\n\n"

"1. Use tools only when necessary.\n\n"

"2. You may ONLY call tools that are present in the "
"provided tool list. Never invent a tool name.\n\n"

"3. Available tools include search, visit_webpage, "
"execute_python_code, and other tools provided by the API. "
"Do not call find, find_in_page, browser, calculator, "
"or any other tool unless it is explicitly provided.\n\n"

"4. For arithmetic and calculations, use "
"execute_python_code.\n\n"

"5. For current or factual web information, use search "
"or visit_webpage.\n\n"

"6. If search results already contain the required "
"information, do not search again and do not try to "
"find text inside the result. Use the information directly.\n\n"

"7. Do not repeat the same search unless the previous "
"result was insufficient.\n\n"

"8. Once a tool result contains enough information to "
"answer the question, STOP calling tools and immediately "
"provide the final answer.\n\n"

"9. If a search result contains the answer explicitly, "
"return that answer directly. Do not search for the same "
"information again.\n\n"

"10. For numerical calculations, prefer "
"execute_python_code rather than mental arithmetic.\n\n"

"11. Do not write reasoning or planning text before "
"a tool call.\n\n"

"12. If you need a tool, call it directly.\n\n"

"13. After receiving a tool result, either call another "
"available tool or provide the final answer.\n\n"

"14. Never output internal reasoning as plain text."

"15. If a search returns empty results, try a different "
"search query once. Do not repeatedly call the same empty search.\n\n"

"16. Never finish with an empty response. If you have "
"enough information, always provide a concise textual answer.\n\n"

"17. Use search at most twice for a task. After "
"two searches, use the information already obtained "
"and provide the final answer.\n\n"

"18. If fetch_json_api returns a 403, Forbidden, or access "
"error, do not retry the same API request. Use search or "
"visit_webpage instead.\n\n"

"19. Never repeat a tool call that failed with 403 Forbidden. "
"Use another available source or method.\n\n"

"20. When a search result provides a useful webpage URL, "
"prefer visit_webpage to retrieve the page content. "
"Do not use a JSON API unless it is clearly necessary.\n\n"
            ),
        },

        {
            "role": "user",
            "content": user_message,
        },
    ]

    tool_schemas = build_tool_schemas(ALL_TOOLS)

    # ======================================
    # AGENT LOOP
    # ======================================

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

        # ==================================
        # FINAL ANSWER
        # ==================================

        if not message.tool_calls:

            final_answer = (message.content or "").strip()

            if not final_answer:

                print()
                print("EMPTY MODEL RESPONSE")
                print("Requesting final answer...")

                messages.append({
                "role": "user",
                "content": (
                    "Provide the final answer now. "
                    "Do not call any tools. "
                    "Return only the concise answer."
            ),
        })

                continue

            print()
            print("=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)

            print(final_answer)

            return final_answer

        # ==================================
        # ASSISTANT TOOL CALL
        # ==================================

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

        # ==================================
        # EXECUTE TOOLS
        # ==================================

        for call in message.tool_calls:

            raw_arguments = call.function.arguments

            print()
            print("RAW TOOL ARGUMENTS:")
            print(raw_arguments)

            try:

                arguments = json.loads(raw_arguments)

            except json.JSONDecodeError as e:

                result = (
                    "INVALID TOOL ARGUMENTS.\n"
                    "The arguments must be valid JSON.\n"
                    "Do not call the tool again with "
                    "the same invalid arguments.\n"
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

    # ======================================
    # MAX STEPS
    # ======================================

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