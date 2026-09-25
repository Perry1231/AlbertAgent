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

MAX_STEPS = 10
MAX_TOOL_RESULT_CHARS = 2500
MAX_RETRIES = 2
SEARCH_MAX_CALLS = 2


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
                max_tokens=700,
            )

        except Exception as e:

            error_text = str(e).lower()

            # ==================================
            # DAILY TOKEN LIMIT
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
            # RATE LIMIT
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

                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed.\n"
                        "Do not write reasoning or planning text.\n"
                        "Use the available tools correctly.\n"
                        "If enough information is already available, "
                        "provide the final answer immediately."
                    ),
                })

                continue

            raise


# ==========================================
# FORCE FINAL ANSWER
# ==========================================

def force_final_answer(messages):

    print()
    print("=" * 60)
    print("FORCING FINAL ANSWER")
    print("=" * 60)

    final_messages = messages.copy()

    final_messages.append({
        "role": "user",
        "content": (
            "You must provide the final answer now.\n\n"
            "Do NOT call any tools.\n"
            "Do NOT search.\n"
            "Do NOT perform additional actions.\n"
            "Use only the information already present "
            "in the conversation.\n\n"
            "Return only the final concise answer."
        ),
    })

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=final_messages,
            tool_choice="none",
            max_tokens=700,
        )

        answer = (
            response.choices[0].message.content or ""
        ).strip()

        if answer:
            print()
            print("FINAL ANSWER:")
            print(answer)

            return answer

    except Exception as e:

        print()
        print("FINALIZATION ERROR:")
        print(e)

    return (
        "Unable to produce a final answer "
        "within the available execution limits."
    )


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

                "2. You may ONLY call tools that are present "
                "in the provided tool list. Never invent a tool name.\n\n"

                "3. Available tools include search, "
                "visit_webpage, execute_python_code, and other "
                "tools provided by the API.\n\n"

                "4. For arithmetic and calculations, use "
                "execute_python_code.\n\n"

                "5. For factual web information, use search first.\n\n"

                "6. If search results already contain the required "
                "information, do not search again.\n\n"

                "7. Never repeat the exact same tool call.\n\n"

                "8. If a tool returns 403 Forbidden or an access "
                "error, never repeat the same request.\n\n"

                "9. Once a tool result contains enough information "
                "to answer the question, stop using tools and "
                "provide the final answer.\n\n"

                "10. Do not write reasoning or planning text.\n\n"

                "11. If you need a tool, call it directly.\n\n"

                "12. After receiving a tool result, either call "
                "another necessary tool or provide the final answer.\n\n"

                "13. Never output internal reasoning.\n\n"

                "14. If search returns empty results, you may try "
                "one different search query.\n\n"

                "15. Search may be used at most twice per task.\n\n"

                "16. Never finish with an empty response.\n\n"

                "17. If enough information is available, "
                "always provide a concise final answer.\n\n"

                "18. If fetch_json_api returns 403, Forbidden, "
                "or an access error, do not retry it. "
                "Use search or another available tool instead.\n\n"

                "19. Do not call find, find_in_page, browser, "
                "calculator, or any tool that is not explicitly "
                "provided.\n\n"

                "20. When a search result contains a useful URL, "
                "you may use visit_webpage to retrieve it.\n\n"

                "21. Do not perform unnecessary searches."
            ),
        },

        {
            "role": "user",
            "content": user_message,
        },
    ]

    tool_schemas = build_tool_schemas(ALL_TOOLS)

    # ======================================
    # TRACKING
    # ======================================

    tool_history = set()
    search_calls = 0

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

            final_answer = (
                message.content or ""
            ).strip()

            if final_answer:

                print()
                print("=" * 60)
                print("FINAL ANSWER")
                print("=" * 60)

                print(final_answer)

                return final_answer

            # Empty response

            print()
            print("EMPTY MODEL RESPONSE")
            print("Requesting final answer...")

            messages.append({
                "role": "user",
                "content": (
                    "Provide the final answer now.\n"
                    "Do not call any tools.\n"
                    "Return only the concise answer."
                ),
            })

            continue

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

            tool_name = call.function.name
            raw_arguments = call.function.arguments

            print()
            print("RAW TOOL ARGUMENTS:")
            print(raw_arguments)

            # ==================================
            # SEARCH LIMIT
            # ==================================

            if tool_name == "search":

                search_calls += 1

                if search_calls > SEARCH_MAX_CALLS:

                    result = (
                        "SEARCH LIMIT REACHED.\n"
                        "Do not perform another search.\n"
                        "Use the information already obtained "
                        "and provide the final answer."
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    })

                    continue

            # ==================================
            # PARSE JSON
            # ==================================

            try:

                arguments = json.loads(
                    raw_arguments
                )

            except json.JSONDecodeError as e:

                result = (
                    "INVALID TOOL ARGUMENTS.\n"
                    "The arguments must be valid JSON.\n"
                    "Do not repeat the same invalid call.\n"
                    f"JSON error: {e}"
                )

            else:

                # ==================================
                # DUPLICATE TOOL CALL PROTECTION
                # ==================================

                tool_key = (
                    tool_name,
                    json.dumps(
                        arguments,
                        sort_keys=True,
                    ),
                )

                if tool_key in tool_history:

                    result = (
                        "DUPLICATE TOOL CALL.\n"
                        "This exact tool call was already executed.\n"
                        "Do not call it again.\n"
                        "Use the previous result and provide "
                        "the final answer."
                    )

                else:

                    tool_history.add(tool_key)

                    result = execute_tool(
                        tool_name,
                        arguments,
                    )

            # ==================================
            # TOOL RESULT
            # ==================================

            messages.append({

                "role": "tool",

                "tool_call_id": call.id,

                "content": str(result),
            })

    # ==========================================
    # MAX STEPS
    # ==========================================

    print()
    print("=" * 60)
    print("MAX STEPS REACHED")
    print("=" * 60)

    return force_final_answer(messages)


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

    result = run_agent(user_input)

    print()
    print("=" * 60)
    print("AGENT RESULT")
    print("=" * 60)
    print(result)