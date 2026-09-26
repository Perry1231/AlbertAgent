import os
import traceback

from dotenv import load_dotenv

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
)


# ============================================================
# ENV
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found in .env"
    )


# ============================================================
# MODEL
# ============================================================

model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    tool_choice="auto",
    max_tokens=500,
)


# ============================================================
# AGENT
# ============================================================

# IMPORTANT:
# We intentionally provide NO custom tools.
#
# ToolCallingAgent automatically adds:
#     final_answer
#
# This lets us test whether GPT-OSS can correctly use
# smolagents' built-in final_answer tool.

agent = ToolCallingAgent(
    tools=[],
    model=model,
    max_steps=2,
    verbosity_level=2,
    return_full_result=True,
)


# ============================================================
# SHOW TOOLS
# ============================================================

print()
print("=" * 60)
print("AGENT TOOLS")
print("=" * 60)
print()

for name in agent.tools:
    print("-", name)


print()
print("=" * 60)
print("TOOLS SENT TO MODEL")
print("=" * 60)
print()

for tool in agent.tools_and_managed_agents:
    print("-", tool.name)


# ============================================================
# CHECK FINAL ANSWER
# ============================================================

if "final_answer" not in agent.tools:

    print()
    print("=" * 60)
    print("ERROR")
    print("=" * 60)
    print()
    print("final_answer was not added automatically.")
    print()

    raise RuntimeError(
        "ToolCallingAgent did not create final_answer"
    )


print()
print("=" * 60)
print("FINAL ANSWER TOOL")
print("=" * 60)
print()

print(
    agent.tools["final_answer"]
)


# ============================================================
# TASK
# ============================================================

task = """
The research is already complete.

The final answer is:

Fairness in Agreement With European Values:
An Interdisciplinary Perspective on AI Regulation

Submit this as the final answer.

You MUST use the tool named exactly:

final_answer

The tool argument must be:

answer

Do NOT call a tool named "answer".
"""


# ============================================================
# RUN
# ============================================================

print()
print("=" * 60)
print("RUNNING AGENT")
print("=" * 60)
print()

try:

    result = agent.run(
        task,
        return_full_result=True,
    )

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    print()

    print(result)

except Exception as e:

    print()
    print("=" * 60)
    print("ERROR")
    print("=" * 60)
    print()

    print(
        "TYPE:",
        type(e).__name__
    )

    print(
        "ERROR:",
        str(e)
    )

    print()
    traceback.print_exc()
