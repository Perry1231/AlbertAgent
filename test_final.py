
import os
import sys
import io
import traceback

# ============================================================
# FORCE UTF-8
# ============================================================

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

    sys.stderr.reconfigure(
        encoding="utf-8",
        errors="replace"
    )

except AttributeError:

    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace"
    )

    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer,
        encoding="utf-8",
        errors="replace"
    )


# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
from openai import OpenAI

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
)

from smolagents.models import get_tool_json_schema

from tools import ALL_TOOLS


# ============================================================
# ENVIRONMENT
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:

    raise RuntimeError(
        "GROQ_API_KEY was not found in .env"
    )


# ============================================================
# SAFE PRINT
# ============================================================

def safe_print(text=""):

    try:

        print(
            text,
            flush=True
        )

    except UnicodeEncodeError:

        print(
            str(text).encode(
                "utf-8",
                errors="replace"
            ).decode(
                "utf-8",
                errors="replace"
            ),
            flush=True
        )


# ============================================================
# HEADER
# ============================================================

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "GPT-OSS FINAL ANSWER TEST"
)
safe_print(
    "Groq + GPT-OSS-120B + smolagents"
)
safe_print(
    "============================================================"
)
safe_print("")


# ============================================================
# CREATE SMOLAGENTS MODEL
# ============================================================

safe_print(
    "Creating smolagents model..."
)

model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    tool_choice="auto",
    max_tokens=500,
)


# ============================================================
# CREATE AGENT
# ============================================================

safe_print(
    "Creating ToolCallingAgent..."
)

agent = ToolCallingAgent(
    tools=ALL_TOOLS,
    model=model,
    max_steps=3,
)


# ============================================================
# SHOW AGENT TOOLS
# ============================================================

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "AGENT TOOLS"
)
safe_print(
    "============================================================"
)

safe_print(
    f"Custom tools: {len(ALL_TOOLS)}"
)

safe_print(
    f"Agent tools: {len(agent.tools)}"
)

safe_print("")

for name in agent.tools.keys():

    safe_print(
        f"- {name}"
    )


# ============================================================
# BUILD REAL REQUEST TOOL SCHEMAS
# ============================================================

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "TOOLS SENT TO GROQ"
)
safe_print(
    "============================================================"
)

tools = [
    get_tool_json_schema(tool)
    for tool in agent.tools_and_managed_agents
]

for tool in tools:

    function = tool.get(
        "function",
        {}
    )

    name = function.get(
        "name",
        "UNKNOWN"
    )

    safe_print(
        f"- {name}"
    )


# ============================================================
# CHECK FINAL ANSWER TOOL
# ============================================================

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "FINAL ANSWER SCHEMA"
)
safe_print(
    "============================================================"
)

final_answer_schema = None

for tool in tools:

    function = tool.get(
        "function",
        {}
    )

    if function.get("name") == "final_answer":

        final_answer_schema = tool

        break


if final_answer_schema is None:

    safe_print(
        "ERROR: final_answer is NOT present!"
    )

    sys.exit(1)


safe_print(
    final_answer_schema
)


# ============================================================
# DIRECT OPENAI CLIENT
# ============================================================

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "CREATING DIRECT GROQ CLIENT"
)
safe_print(
    "============================================================"
)

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are solving a research task.

The research has already been completed.

You MUST submit the final answer using the tool named exactly:

final_answer

The tool name is exactly "final_answer".

Never call a tool named "answer".

Never call a tool named "json".

The final_answer tool accepts exactly one argument:

answer

The argument "answer" must contain the final answer to the task.

Do not put the final answer only in reasoning.

Do not finish with an empty response.
"""


# ============================================================
# USER PROMPT
# ============================================================

USER_PROMPT = """
The research established that the paper title is:

Fairness in Agreement With European Values:
An Interdisciplinary Perspective on AI Regulation

Submit this as the final answer using the final_answer tool.
"""


# ============================================================
# DIRECT GROQ TEST
# ============================================================

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "CALLING GROQ DIRECTLY"
)
safe_print(
    "============================================================"
)

try:

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT,
            },
        ],

        tools=tools,

        tool_choice="auto",

        max_tokens=500,
    )


except Exception as e:

    safe_print("")
    safe_print(
        "============================================================"
    )
    safe_print(
        "GROQ REQUEST ERROR"
    )
    safe_print(
        "============================================================"
    )

    safe_print(
        f"TYPE: {type(e).__name__}"
    )

    safe_print(
        f"ERROR: {repr(e)}"
    )

    traceback.print_exc()

    sys.exit(1)


# ============================================================
# RESPONSE
# ============================================================

message = response.choices[0].message

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "GROQ RESPONSE"
)
safe_print(
    "============================================================"
)

safe_print(
    f"FINISH REASON: "
    f"{response.choices[0].finish_reason}"
)

safe_print("")

safe_print(
    f"CONTENT:"
)
safe_print(
    repr(message.content)
)

safe_print("")

safe_print(
    f"TOOL CALLS:"
)
safe_print(
    repr(message.tool_calls)
)

safe_print("")

safe_print(
    f"REASONING:"
)
safe_print(
    repr(
        getattr(
            message,
            "reasoning",
            None
        )
    )
)


# ============================================================
# USAGE
# ============================================================

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "TOKEN USAGE"
)
safe_print(
    "============================================================"
)

if response.usage:

    safe_print(
        f"Prompt tokens: "
        f"{response.usage.prompt_tokens}"
    )

    safe_print(
        f"Completion tokens: "
        f"{response.usage.completion_tokens}"
    )

    safe_print(
        f"Total tokens: "
        f"{response.usage.total_tokens}"
    )


# ============================================================
# ANALYZE TOOL CALL
# ============================================================

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "ANALYSIS"
)
safe_print(
    "============================================================"
)

tool_calls = message.tool_calls

if not tool_calls:

    safe_print(
        "NO TOOL CALL WAS GENERATED."
    )

    safe_print("")

    if message.content:

        safe_print(
            "The model returned normal content."
        )

    elif getattr(
        message,
        "reasoning",
        None
    ):

        safe_print(
            "The model returned reasoning but no "
            "content and no tool call."
        )

    else:

        safe_print(
            "The model returned neither content "
            "nor a tool call."
        )

else:

    safe_print(
        f"Tool calls generated: "
        f"{len(tool_calls)}"
    )

    for index, call in enumerate(
        tool_calls,
        start=1
    ):

        safe_print("")

        safe_print(
            f"TOOL CALL #{index}"
        )

        safe_print(
            f"ID: {call.id}"
        )

        safe_print(
            f"TYPE: {call.type}"
        )

        safe_print(
            f"NAME: {call.function.name}"
        )

        safe_print(
            f"ARGUMENTS: "
            f"{call.function.arguments}"
        )

        if call.function.name == "final_answer":

            safe_print("")
            safe_print(
                "SUCCESS: Model correctly called "
                "'final_answer'."
            )

        elif call.function.name == "answer":

            safe_print("")
            safe_print(
                "FAILURE: Model called 'answer' "
                "instead of 'final_answer'."
            )

        elif call.function.name == "json":

            safe_print("")
            safe_print(
                "FAILURE: Model called 'json'."
            )

        else:

            safe_print("")
            safe_print(
                f"Model called another tool: "
                f"{call.function.name}"
            )


# ============================================================
# FINAL STATUS
# ============================================================

safe_print("")
safe_print(
    "============================================================"
)
safe_print(
    "FINAL STATUS"
)
safe_print(
    "============================================================"
)

if tool_calls:

    names = [
        call.function.name
        for call in tool_calls
    ]

    if "final_answer" in names:

        safe_print(
            "PASS"
        )

        safe_print(
            "GPT-OSS correctly generated "
            "the final_answer tool call."
        )

    elif "answer" in names:

        safe_print(
            "FAIL"
        )

        safe_print(
            "GPT-OSS generated 'answer' instead "
            "of 'final_answer'."
        )

    elif "json" in names:

        safe_print(
            "FAIL"
        )

        safe_print(
            "GPT-OSS generated 'json' instead "
            "of 'final_answer'."
        )

    else:

        safe_print(
            "UNKNOWN"
        )

        safe_print(
            f"Generated tools: {names}"
        )

else:

    safe_print(
        "FAIL"
    )

    safe_print(
        "No tool call was generated."
    )

safe_print("")
safe_print(
    "============================================================"
)
