
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# ENV
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY not found")


# ============================================================
# CLIENT
# ============================================================

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)


# ============================================================
# ONLY FINAL ANSWER TOOL
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Provides the final answer to the given problem."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "The final answer to the problem"
                        ),
                    }
                },
                "required": ["answer"],
            },
        },
    }
]


# ============================================================
# PROMPT
# ============================================================

system_prompt = """
You are a research assistant.

The research is already complete.

You MUST submit the final answer by calling the tool:

final_answer

The tool has exactly one argument:

answer

The argument must be a valid JSON object.

Correct example:

{
  "answer": "some answer"
}

DO NOT call any tool named "answer".

DO NOT invent another tool name.

DO NOT put the answer directly into the arguments field.

Use exactly:

final_answer({
  "answer": "your final answer"
})
"""


user_prompt = """
Return the following as the final answer:

Fairness in Agreement With European Values:
An Interdisciplinary Perspective on AI Regulation
"""


# ============================================================
# REQUEST
# ============================================================

print()
print("=" * 60)
print("MINIMAL FINAL ANSWER TEST")
print("=" * 60)
print()

try:

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],

        tools=tools,

        tool_choice="required",

        max_tokens=300,
    )

except Exception as e:

    print()
    print("=" * 60)
    print("REQUEST FAILED")
    print("=" * 60)
    print()

    print(type(e).__name__)
    print(str(e))

    sys.exit(1)


# ============================================================
# RESPONSE
# ============================================================

message = response.choices[0].message

print()
print("=" * 60)
print("RESPONSE")
print("=" * 60)
print()

print("FINISH:", response.choices[0].finish_reason)

print()
print("CONTENT:")
print(repr(message.content))

print()
print("TOOL CALLS:")
print(repr(message.tool_calls))

print()
print("REASONING:")
print(
    repr(
        getattr(
            message,
            "reasoning",
            None
        )
    )
)


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("RESULT")
print("=" * 60)
print()

if message.tool_calls:

    for call in message.tool_calls:

        print("TOOL NAME:")
        print(call.function.name)

        print()
        print("ARGUMENTS:")
        print(call.function.arguments)

        print()

        if call.function.name == "final_answer":

            print("SUCCESS")
            print(
                "GPT-OSS correctly used final_answer."
            )

        else:

            print("FAIL")
            print(
                "GPT-OSS generated:",
                call.function.name
            )

else:

    print("FAIL")
    print("No tool call was generated.")

