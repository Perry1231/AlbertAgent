import os
import json
import time

from dotenv import load_dotenv
from openai import OpenAI

from tools import ALL_TOOLS


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found")


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
)


MODEL = "openai/gpt-oss-120b"

# Groq limit from your error:
# TPM = 8000
#
# We intentionally stay below it.
SAFE_CONTEXT_CHARS = 18000

MAX_STEPS = 10

MAX_TOOL_RESULT_CHARS = 1200

MAX_RETRIES = 2

SEARCH_MAX_CALLS = 2

# Keep only this many completed tool exchanges
MAX_HISTORY_BLOCKS = 5

# Maximum generated tokens per normal request
MAX_OUTPUT_TOKENS = 400

# Maximum generated tokens for forced final answer
MAX_FINAL_TOKENS = 500


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Albert, an efficient AI agent solving GAIA benchmark tasks.

Rules:

1. Use tools only when necessary.
2. Only use tools from the provided tool list.
3. Never invent tool names.
4. Use search for factual web information.
5. Use execute_python_code for calculations.
6. Do not repeat the same tool call.
7. Never repeat a failed 403/Forbidden request.
8. Search at most twice per task.
9. If a tool result contains enough information, answer immediately.
10. Do not write reasoning or planning text.
11. Always provide a final answer.
12. If information is already available, do not search again.
13. Do not call unavailable tools such as find, browser, calculator,
    or find_in_page unless they are explicitly present in the tool list.
14. If a search result contains the required answer, use it directly.
15. Prefer concise answers.
""".strip()


# ============================================================
# TOOL SCHEMAS
# ============================================================

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


# ============================================================
# TOOL MAP
# ============================================================

TOOL_MAP = {
    tool.name: tool
    for tool in ALL_TOOLS
}


# ============================================================
# TOOL EXECUTION
# ============================================================

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


# ============================================================
# MESSAGE SIZE
# ============================================================

def message_size(message):

    try:
        return len(
            json.dumps(
                message,
                ensure_ascii=False,
            )
        )
    except Exception:
        return len(str(message))


def messages_size(messages):

    return sum(
        message_size(message)
        for message in messages
    )


# ============================================================
# HISTORY BLOCKS
#
# A tool exchange is:
#
# assistant
#   └── tool_calls
#
# tool
#   └── tool_call_id
#
# We NEVER keep the tool message without its assistant
# tool-call message.
# ============================================================

def split_history_into_blocks(messages):

    if not messages:
        return []

    blocks = []

    current = []

    for message in messages:

        role = message.get("role")

        # System and initial user message are handled separately.
        if role in ("system", "user"):

            if current:
                blocks.append(current)
                current = []

            blocks.append([message])

            continue

        # Assistant with tool calls starts a new exchange.
        if (
            role == "assistant"
            and message.get("tool_calls")
        ):

            if current:
                blocks.append(current)

            current = [message]

            continue

        # Tool belongs to current assistant tool call.
        if role == "tool":

            current.append(message)
            continue

        # Normal assistant final answer.
        if role == "assistant":

            if current:
                blocks.append(current)

            current = [message]

            continue

        current.append(message)

    if current:
        blocks.append(current)

    return blocks


# ============================================================
# COMPACT HISTORY
# ============================================================

def compact_messages(messages):

    if not messages:
        return messages

    # --------------------------------------------
    # Keep system message
    # --------------------------------------------

    system_messages = [
        m
        for m in messages
        if m.get("role") == "system"
    ]

    # --------------------------------------------
    # Keep original user task
    # --------------------------------------------

    user_messages = [
        m
        for m in messages
        if m.get("role") == "user"
    ]

    original_user = (
        user_messages[0]
        if user_messages
        else None
    )

    # --------------------------------------------
    # Find actual conversation blocks
    # --------------------------------------------

    blocks = split_history_into_blocks(messages)

    # Remove system/user blocks.
    conversation_blocks = []

    for block in blocks:

        if all(
            m.get("role") not in ("system", "user")
            for m in block
        ):
            conversation_blocks.append(block)

    # --------------------------------------------
    # Keep latest blocks
    # --------------------------------------------

    conversation_blocks = conversation_blocks[
        -MAX_HISTORY_BLOCKS:
    ]

    compacted = []

    compacted.extend(system_messages)

    if original_user:
        compacted.append(original_user)

    # --------------------------------------------
    # Add recent conversation
    # --------------------------------------------

    for block in conversation_blocks:

        compacted.extend(block)

    # --------------------------------------------
    # Hard character safety
    # --------------------------------------------

    while (
        messages_size(compacted)
        > SAFE_CONTEXT_CHARS
        and len(conversation_blocks) > 1
    ):

        conversation_blocks.pop(0)

        compacted = []

        compacted.extend(system_messages)

        if original_user:
            compacted.append(original_user)

        for block in conversation_blocks:
            compacted.extend(block)

    print()
    print(
        f"CONTEXT SIZE: "
        f"{messages_size(compacted)} chars"
    )

    return compacted


# ============================================================
# BUILD REQUEST MESSAGES
# ============================================================

def prepare_messages(messages):

    compacted = compact_messages(messages)

    # Additional emergency protection.
    #
    # If still too large, keep:
    # system
    # original user
    # latest complete tool blocks
    #

    if messages_size(compacted) <= SAFE_CONTEXT_CHARS:
        return compacted

    system = next(
        (
            m
            for m in compacted
            if m.get("role") == "system"
        ),
        None,
    )

    users = [
        m
        for m in compacted
        if m.get("role") == "user"
    ]

    original_user = users[0] if users else None

    blocks = split_history_into_blocks(compacted)

    conversation = [
        block
        for block in blocks
        if all(
            m.get("role") not in ("system", "user")
            for m in block
        )
    ]

    result = []

    if system:
        result.append(system)

    if original_user:
        result.append(original_user)

    # Add only newest complete block.
    if conversation:
        result.extend(conversation[-1])

    return result


# ============================================================
# GROQ COMPLETION
# ============================================================

def create_completion(messages, tool_schemas):

    request_messages = prepare_messages(messages)

    request_size = messages_size(request_messages)

    print()
    print(
        f"REQUEST SIZE: "
        f"{request_size} chars"
    )

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            return client.chat.completions.create(
                model=MODEL,
                messages=request_messages,
                tools=tool_schemas,
                tool_choice="auto",
                max_tokens=MAX_OUTPUT_TOKENS,
            )

        except Exception as e:

            error_text = str(e).lower()

            # ==========================================
            # TPM / REQUEST TOO LARGE
            # ==========================================

            if (
                "tokens per minute" in error_text
                or "tpm" in error_text
                or "request too large" in error_text
                or "413" in error_text
            ):

                print()
                print("=" * 60)
                print("GROQ REQUEST TOO LARGE")
                print("=" * 60)
                print(e)

                # Emergency retry with only essential context.
                if attempt < MAX_RETRIES:

                    emergency = []

                    system = next(
                        (
                            m
                            for m in messages
                            if m.get("role") == "system"
                        ),
                        None,
                    )

                    user = next(
                        (
                            m
                            for m in messages
                            if m.get("role") == "user"
                        ),
                        None,
                    )

                    if system:
                        emergency.append(system)

                    if user:
                        emergency.append(user)

                    # Add newest COMPLETE tool exchange.
                    blocks = split_history_into_blocks(
                        messages
                    )

                    valid_blocks = [
                        block
                        for block in blocks
                        if all(
                            m.get("role")
                            not in ("system", "user")
                            for m in block
                        )
                    ]

                    if valid_blocks:
                        emergency.extend(
                            valid_blocks[-1]
                        )

                    request_messages = emergency

                    print(
                        "Retrying with emergency compact context..."
                    )

                    continue

                raise

            # ==========================================
            # DAILY TOKEN LIMIT
            # ==========================================

            if (
                "tokens per day" in error_text
                or "tpd" in error_text
            ):

                print()
                print(
                    "GROQ DAILY TOKEN LIMIT REACHED"
                )

                raise

            # ==========================================
            # RATE LIMIT
            # ==========================================

            if (
                "429" in error_text
                or "rate_limit" in error_text
            ):

                if attempt == MAX_RETRIES:
                    raise

                wait_time = attempt * 5

                print(
                    f"Rate limit. "
                    f"Retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

                continue

            # ==========================================
            # OUTPUT PARSE ERROR
            # ==========================================

            if "output_parse_failed" in error_text:

                print()
                print(
                    "GROQ OUTPUT PARSING ERROR"
                )
                print(e)

                if attempt == MAX_RETRIES:
                    raise

                messages.append({
                    "role": "user",
                    "content": (
                        "Give a valid tool call or final answer. "
                        "Do not output reasoning."
                    ),
                })

                continue

            raise


# ============================================================
# FORCE FINAL ANSWER
# ============================================================

def force_final_answer(messages):

    print()
    print("=" * 60)
    print("FORCING FINAL ANSWER")
    print("=" * 60)

    # --------------------------------------------
    # Build compact final context.
    # --------------------------------------------

    compacted = prepare_messages(messages)

    compacted.append({
        "role": "user",
        "content": (
            "Provide the final answer now.\n"
            "Do NOT call tools.\n"
            "Do NOT search.\n"
            "Use only information already obtained.\n"
            "Return only the concise final answer."
        ),
    })

    # --------------------------------------------
    # Emergency context protection.
    # --------------------------------------------

    while (
        messages_size(compacted)
        > SAFE_CONTEXT_CHARS
    ):

        # Keep system + original user + newest block.
        system = next(
            (
                m
                for m in compacted
                if m.get("role") == "system"
            ),
            None,
        )

        users = [
            m
            for m in compacted
            if m.get("role") == "user"
        ]

        original_user = (
            users[0]
            if users
            else None
        )

        final_instruction = compacted[-1]

        blocks = split_history_into_blocks(
            compacted[:-1]
        )

        conversation = [
            block
            for block in blocks
            if all(
                m.get("role")
                not in ("system", "user")
                for m in block
            )
        ]

        compacted = []

        if system:
            compacted.append(system)

        if original_user:
            compacted.append(original_user)

        if conversation:
            compacted.extend(
                conversation[-1]
            )

        compacted.append(
            final_instruction
        )

        break

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=compacted,
            tool_choice="none",
            max_tokens=MAX_FINAL_TOKENS,
        )

        answer = (
            response.choices[0].message.content
            or ""
        ).strip()

        if answer:

            print()
            print("=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)

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


# ============================================================
# AGENT
# ============================================================

def run_agent(user_message):

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },

        {
            "role": "user",
            "content": user_message,
        },
    ]

    tool_schemas = build_tool_schemas(
        ALL_TOOLS
    )

    # ==========================================
    # TRACKING
    # ==========================================

    tool_history = set()

    search_calls = 0

    # ==========================================
    # AGENT LOOP
    # ==========================================

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        print()
        print("#" * 60)
        print(
            f"AGENT STEP "
            f"{step}/{MAX_STEPS}"
        )
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

        # ==========================================
        # FINAL ANSWER
        # ==========================================

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

            # Empty model response

            print()
            print("EMPTY MODEL RESPONSE")

            messages.append({
                "role": "user",
                "content": (
                    "Provide the final answer now. "
                    "Do not call tools. "
                    "Return only the concise answer."
                ),
            })

            continue

        # ==========================================
        # ASSISTANT TOOL CALL
        # ==========================================

        assistant_tool_message = {

            "role": "assistant",

            "content": (
                message.content or ""
            ),

            "tool_calls": [

                {
                    "id": call.id,

                    "type": "function",

                    "function": {

                        "name": call.function.name,

                        "arguments": (
                            call.function.arguments
                        ),
                    },
                }

                for call in message.tool_calls
            ],
        }

        messages.append(
            assistant_tool_message
        )

        # ==========================================
        # EXECUTE TOOLS
        # ==========================================

        for call in message.tool_calls:

            tool_name = (
                call.function.name
            )

            raw_arguments = (
                call.function.arguments
            )

            print()
            print("RAW TOOL ARGUMENTS:")
            print(raw_arguments)

            # ======================================
            # SEARCH LIMIT
            # ======================================

            if tool_name == "search":

                search_calls += 1

                if (
                    search_calls
                    > SEARCH_MAX_CALLS
                ):

                    result = (
                        "SEARCH LIMIT REACHED. "
                        "Do not search again. "
                        "Use information already obtained "
                        "and provide the final answer."
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    })

                    continue

            # ======================================
            # JSON PARSE
            # ======================================

            try:

                arguments = json.loads(
                    raw_arguments
                )

            except json.JSONDecodeError as e:

                result = (
                    "INVALID TOOL ARGUMENTS.\n"
                    "Arguments must be valid JSON.\n"
                    "Do not repeat the same invalid call.\n"
                    f"JSON error: {e}"
                )

            else:

                # ==================================
                # DUPLICATE CALL PROTECTION
                # ==================================

                tool_key = (

                    tool_name,

                    json.dumps(
                        arguments,
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                )

                if tool_key in tool_history:

                    result = (
                        "DUPLICATE TOOL CALL.\n"
                        "This exact tool call was already "
                        "executed.\n"
                        "Do not repeat it.\n"
                        "Use the previous result and answer."
                    )

                else:

                    tool_history.add(
                        tool_key
                    )

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

    return force_final_answer(
        messages
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ALBERT AGENT")
    print("=" * 60)

    print(f"Model: {MODEL}")
    print(f"Tools: {len(ALL_TOOLS)}")
    print(f"Max steps: {MAX_STEPS}")
    print(
        f"Max tool result: "
        f"{MAX_TOOL_RESULT_CHARS} chars"
    )
    print(
        f"Safe context: "
        f"{SAFE_CONTEXT_CHARS} chars"
    )

    print()
    print("Available tools:")

    for tool in ALL_TOOLS:

        print(
            f" - {tool.name}"
        )

    print()
    print("=" * 60)

    user_input = input("You: ")

    result = run_agent(
        user_input
    )

    print()
    print("=" * 60)
    print("AGENT RESULT")
    print("=" * 60)

    print(result)