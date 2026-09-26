import os

from dotenv import load_dotenv

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
)

from tools import ALL_TOOLS


# ============================================================
# ENV
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found"
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

agent = ToolCallingAgent(
    tools=ALL_TOOLS,
    model=model,
    max_steps=3,
    verbosity_level=2,
    return_full_result=True,
)


# ============================================================
# REMOVE AUTO FINAL ANSWER
# ============================================================

print()
print("=" * 60)
print("TOOLS BEFORE")
print("=" * 60)

print(
    list(agent.tools.keys())
)


if "final_answer" in agent.tools:

    del agent.tools["final_answer"]


print()
print("=" * 60)
print("TOOLS AFTER")
print("=" * 60)

print(
    list(agent.tools.keys())
)


# ============================================================
# TEST
# ============================================================

task = """
Open this webpage using the visit_webpage tool:

https://ar5iv.labs.arxiv.org/html/2207.01510

Tell me the title of the paper.

Do not use a final answer tool.
"""


print()
print("=" * 60)
print("RUN")
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
        type(e).__name__,
        str(e)
    )

    import traceback

    traceback.print_exc()

