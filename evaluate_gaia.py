# -*- coding: utf-8 -*-

import os
import sys
import io
import json
import logging
import traceback
from pathlib import Path


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
from datasets import load_dataset
from smolagents import ToolCallingAgent, OpenAIServerModel
from tqdm import tqdm

from tools import ALL_TOOLS


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s: %(message)s"
)

logging.getLogger("smolagents").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = BASE_DIR / "submission.jsonl"

GAIA_FILES_DIR = BASE_DIR / "gaia_files"


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not GROQ_API_KEY:

    raise RuntimeError(
        "\n"
        "============================================================\n"
        "ERROR: GROQ_API_KEY not found!\n"
        "============================================================\n"
        "Create a .env file next to this script:\n\n"
        "GROQ_API_KEY=your_groq_api_key\n"
        "HF_TOKEN=your_huggingface_token\n"
        "============================================================\n"
    )


# ============================================================
# MODEL
# ============================================================

print(
    "Initializing Groq model...",
    flush=True
)

model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    tool_choice="auto",
    max_tokens=500,
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an autonomous research and problem-solving agent.

You are solving tasks from the GAIA benchmark.

General rules:

1. Carefully understand the user's question before acting.

2. Use available tools whenever external information,
   calculations, files, web pages, documents, or data
   are required.

3. Do not invent facts.

4. If a tool returns information, use that information
   rather than guessing.

5. For web research, verify important information
   from reliable sources.

6. When using smart_web_scraper:
   ALWAYS provide BOTH:
   - url
   - prompt

7. If a file is provided, inspect the file when necessary.

8. Perform calculations carefully.

9. Think through multi-step problems before giving
   the final answer.

10. The final answer should be concise and directly
    answer the original question.

11. Do not include unnecessary explanations in the
    final answer unless they are required.

12. Never output a fabricated answer merely because
    a tool failed.

13. If you have enough information to answer the task,
    stop researching and provide the final answer.

14. Do not repeatedly search for the same information
    unless the previous result was insufficient.

15. For multi-hop research tasks, keep track of the
    exact evidence needed for the final answer.
"""


# ============================================================
# AGENT
# ============================================================

print(
    "Initializing agent...",
    flush=True
)

agent = ToolCallingAgent(
    tools=ALL_TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
    max_steps=6,
    verbosity_level=1,
    return_full_result=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_print(text=""):
    """
    UTF-8-safe console output.
    """

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
# FIND ATTACHMENT
# ============================================================

def find_attachment(file_name):
    """
    Try to find a GAIA attachment locally.

    Search locations:
        ./gaia_files/
        ./files/
        ./attachments/
        current directory
    """

    if not file_name:
        return None

    file_name = str(file_name).strip()

    if not file_name:
        return None

    candidates = [

        GAIA_FILES_DIR / file_name,

        BASE_DIR / "files" / file_name,

        BASE_DIR / "attachments" / file_name,

        BASE_DIR / file_name,

    ]

    for candidate in candidates:

        if candidate.exists() and candidate.is_file():

            return candidate

    return None


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(question, file_name=""):
    """
    Build the final prompt sent to the agent.
    """

    prompt_parts = []

    prompt_parts.append(
        "Solve the following GAIA task carefully."
    )

    prompt_parts.append("")

    prompt_parts.append(
        "QUESTION:"
    )

    prompt_parts.append(
        str(question)
    )

    if file_name:

        prompt_parts.append("")

        prompt_parts.append(
            f"ATTACHED FILE NAME: {file_name}"
        )

        attachment = find_attachment(file_name)

        if attachment:

            prompt_parts.append(
                f"LOCAL FILE PATH: {attachment}"
            )

            prompt_parts.append(
                "The attached file is available locally. "
                "Inspect it if the task requires information "
                "from the file."
            )

        else:

            prompt_parts.append(
                "The attachment is referenced by name, but "
                "its local file was not found automatically."
            )

    prompt_parts.append("")

    prompt_parts.append(
        "Return the final answer to the question."
    )

    return "\n".join(prompt_parts)


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(output_file, task_id, prediction):
    """
    Append one result to submission.jsonl.
    """

    result = {
        "task_id": task_id,
        "model_answer": prediction
    }

    with open(
        output_file,
        "a",
        encoding="utf-8",
        errors="replace"
    ) as f:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# GET PREDICTION
# ============================================================

def get_prediction(response):
    """
    Convert agent response into a clean string.
    """

    if response is None:

        return "Error: Empty response"

    # RunResult
    if hasattr(response, "output"):

        response = response.output

    if response is None:

        return "Error: Empty response"

    try:

        prediction = str(
            response
        ).strip()

    except Exception:

        prediction = repr(
            response
        )

    if not prediction:

        return "Error: Empty response"

    return prediction


# ============================================================
# DEBUG RUN RESULT
# ============================================================

def print_run_result(result):
    """
    Print detailed RunResult diagnostics.
    """

    safe_print("")
    safe_print(
        "============================================================"
    )
    safe_print(
        "AGENT RESULT"
    )
    safe_print(
        "============================================================"
    )

    safe_print(
        f"RESULT TYPE: {type(result).__name__}"
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    if hasattr(result, "state"):

        safe_print(
            f"STATE: {result.state}"
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    if hasattr(result, "output"):

        safe_print(
            f"OUTPUT: {repr(result.output)}"
        )

    # --------------------------------------------------------
    # TOKEN USAGE
    # --------------------------------------------------------

    if hasattr(result, "token_usage"):

        safe_print(
            f"TOKEN USAGE: {result.token_usage}"
        )

        token_usage = result.token_usage

        if token_usage is not None:

            if hasattr(
                token_usage,
                "input_tokens"
            ):

                safe_print(
                    f"INPUT TOKENS: "
                    f"{token_usage.input_tokens}"
                )

            if hasattr(
                token_usage,
                "output_tokens"
            ):

                safe_print(
                    f"OUTPUT TOKENS: "
                    f"{token_usage.output_tokens}"
                )

    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    if hasattr(result, "timing"):

        safe_print(
            f"TIMING: {result.timing}"
        )

    # --------------------------------------------------------
    # STEPS
    # --------------------------------------------------------

    if hasattr(result, "steps"):

        safe_print(
            f"TOTAL STEPS: {len(result.steps)}"
        )

        for step_index, step in enumerate(
            result.steps
        ):

            safe_print("")
            safe_print(
                f"--- STEP {step_index} ---"
            )

            if isinstance(
                step,
                dict
            ):

                safe_print(
                    f"TYPE: "
                    f"{step.get('type', 'unknown')}"
                )

                if step.get("error") is not None:

                    safe_print(
                        f"ERROR: "
                        f"{step.get('error')}"
                    )

                if step.get("token_usage") is not None:

                    safe_print(
                        f"TOKEN USAGE: "
                        f"{step.get('token_usage')}"
                    )

            else:

                safe_print(
                    f"TYPE: "
                    f"{type(step).__name__}"
                )

    safe_print(
        "============================================================"
    )


# ============================================================
# DATASET
# ============================================================

def load_gaia_dataset():

    safe_print(
        "STEP 1: Loading GAIA dataset..."
    )

    try:

        dataset = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_all",
            split="validation",
            token=HF_TOKEN
        )

    except Exception as e:

        safe_print("")
        safe_print(
            "============================================================"
        )

        safe_print(
            "ERROR LOADING GAIA DATASET"
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

        raise

    safe_print(
        f"STEP 2: Dataset loaded. Tasks: {len(dataset)}"
    )

    return dataset


# ============================================================
# MAIN
# ============================================================

def main():

    safe_print("")
    safe_print(
        "============================================================"
    )
    safe_print(
        "GAIA BENCHMARK"
    )
    safe_print(
        "Groq + GPT-OSS-120B + smolagents"
    )
    safe_print(
        "============================================================"
    )
    safe_print("")

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    dataset = load_gaia_dataset()

    # --------------------------------------------------------
    # LIMIT
    # --------------------------------------------------------

    limit_env = os.getenv(
        "GAIA_LIMIT",
        "1"
    ).strip().lower()

    if limit_env == "all":

        test_dataset = dataset

    else:

        try:

            limit = int(
                limit_env
            )

        except ValueError:

            safe_print(
                f"Invalid GAIA_LIMIT={limit_env}. "
                "Using 1."
            )

            limit = 1

        if limit <= 0:

            limit = 1

        limit = min(
            limit,
            len(dataset)
        )

        test_dataset = dataset.select(
            range(limit)
        )

    safe_print(
        f"STEP 3: Tasks selected: "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    safe_print(
        f"STEP 4: Output file: "
        f"{OUTPUT_FILE}"
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        errors="replace"
    ):
        pass

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    safe_print("")
    safe_print(
        "STEP 5: Starting evaluation..."
    )
    safe_print("")

    successful = 0
    failed = 0

    for index, item in enumerate(
        tqdm(
            test_dataset,
            ascii=True,
            desc="Evaluating"
        )
    ):

        task_number = index + 1

        task_id = item.get(
            "task_id",
            f"unknown-{task_number}"
        )

        question = item.get(
            "Question",
            ""
        )

        file_name = item.get(
            "file_name",
            ""
        )

        # ----------------------------------------------------
        # TASK HEADER
        # ----------------------------------------------------

        safe_print("")
        safe_print(
            "============================================================"
        )

        safe_print(
            f"TASK {task_number}/"
            f"{len(test_dataset)}"
        )

        safe_print(
            f"TASK ID: {task_id}"
        )

        safe_print(
            "============================================================"
        )

        # ----------------------------------------------------
        # BUILD PROMPT
        # ----------------------------------------------------

        prompt = build_prompt(
            question=question,
            file_name=file_name
        )

        safe_print("")
        safe_print(
            "PROMPT:"
        )

        safe_print(
            prompt
        )

        safe_print("")
        safe_print(
            "Running agent..."
        )

        # ----------------------------------------------------
        # RUN AGENT
        # ----------------------------------------------------

        prediction = None

        try:

            result = agent.run(
                prompt,
                return_full_result=True
            )

            # ------------------------------------------------
            # PRINT COMPLETE RESULT
            # ------------------------------------------------

            print_run_result(
                result
            )

            # ------------------------------------------------
            # EXTRACT ANSWER
            # ------------------------------------------------

            prediction = get_prediction(
                result
            )

            # ------------------------------------------------
            # SUCCESS / FAILURE
            # ------------------------------------------------

            if prediction.startswith(
                "Error:"
            ):

                failed += 1

            else:

                successful += 1

        except KeyboardInterrupt:

            safe_print("")
            safe_print(
                "Evaluation interrupted by user."
            )

            raise

        except Exception as e:

            failed += 1

            safe_print("")
            safe_print(
                "============================================================"
            )

            safe_print(
                "AGENT ERROR"
            )

            safe_print(
                "============================================================"
            )

            safe_print(
                f"TASK ID: {task_id}"
            )

            safe_print(
                f"ERROR TYPE: "
                f"{type(e).__name__}"
            )

            safe_print(
                f"ERROR: {repr(e)}"
            )

            # ------------------------------------------------
            # CAUSE
            # ------------------------------------------------

            if e.__cause__ is not None:

                safe_print(
                    f"CAUSE TYPE: "
                    f"{type(e.__cause__).__name__}"
                )

                safe_print(
                    f"CAUSE: "
                    f"{repr(e.__cause__)}"
                )

            # ------------------------------------------------
            # CONTEXT
            # ------------------------------------------------

            if e.__context__ is not None:

                safe_print(
                    f"CONTEXT TYPE: "
                    f"{type(e.__context__).__name__}"
                )

                safe_print(
                    f"CONTEXT: "
                    f"{repr(e.__context__)}"
                )

            # ------------------------------------------------
            # TRACEBACK
            # ------------------------------------------------

            traceback.print_exc()

            prediction = (
                f"Error: "
                f"{type(e).__name__}: "
                f"{e}"
            )

        # ----------------------------------------------------
        # PRINT ANSWER
        # ----------------------------------------------------

        safe_print("")
        safe_print(
            "================ MODEL ANSWER ================="
        )

        safe_print(
            prediction
        )

        safe_print(
            "================================================="
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        try:

            save_result(
                output_file=OUTPUT_FILE,
                task_id=task_id,
                prediction=prediction
            )

        except Exception as e:

            safe_print("")
            safe_print(
                "ERROR SAVING RESULT:"
            )

            safe_print(
                repr(e)
            )

            traceback.print_exc()

            failed += 1

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    safe_print("")
    safe_print(
        "============================================================"
    )

    safe_print(
        "EVALUATION FINISHED"
    )

    safe_print(
        "============================================================"
    )

    safe_print(
        f"Total tasks: "
        f"{len(test_dataset)}"
    )

    safe_print(
        f"Successful: "
        f"{successful}"
    )

    safe_print(
        f"Failed: "
        f"{failed}"
    )

    safe_print(
        f"Results: "
        f"{OUTPUT_FILE}"
    )

    safe_print(
        "============================================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        safe_print("")
        safe_print(
            "Program stopped by user."
        )

        sys.exit(130)

    except Exception as e:

        safe_print("")
        safe_print(
            "============================================================"
        )

        safe_print(
            "FATAL ERROR"
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