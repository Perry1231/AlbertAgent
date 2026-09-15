# -*- coding: utf-8 -*-

import os
import sys
import io
import json
import logging
import traceback

# ============================================================
# FORCE UTF-8
# ============================================================

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
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
from tqdm import tqdm

from smolagents import OpenAIServerModel, ToolCallingAgent

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
# ENVIRONMENT
# ============================================================

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not groq_key:
    raise ValueError(
        "GROQ_API_KEY not found in .env file!"
    )


# ============================================================
# MODEL
# ============================================================

model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=groq_key,

    # ВАЖЛИВО:
    # Для ToolCallingAgent дозволяємо native tool calling
    tool_choice="auto",

    flatten_messages_as_text=True,
)


# ============================================================
# AGENT
# ============================================================

agent = ToolCallingAgent(
    tools=ALL_TOOLS,
    model=model,
    verbosity_level=0,
)


# ============================================================
# EVALUATION
# ============================================================

def main():

    print(
        "STEP 1: Starting program...",
        flush=True
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    try:

        print(
            "STEP 2: Loading GAIA dataset...",
            flush=True
        )

        dataset = load_dataset(
            "gaia-benchmark/GAIA",
            "2023_all",
            split="validation",
            token=hf_token
        )

        print(
            f"STEP 3: Dataset loaded. Tasks: {len(dataset)}",
            flush=True
        )

    except Exception as e:

        print(
            "\n!!! ERROR LOADING DATASET !!!",
            flush=True
        )

        print(
            "TYPE:",
            type(e).__name__,
            flush=True
        )

        print(
            "ERROR:",
            repr(e),
            flush=True
        )

        traceback.print_exc()

        raise

    # --------------------------------------------------------
    # OUTPUT FILE
    # --------------------------------------------------------

    output_file = "submission.jsonl"

    print(
        f"STEP 4: Starting evaluation on {len(dataset)} tasks...",
        flush=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
        errors="replace"
    ):
        pass

    # --------------------------------------------------------
    # TEMPORARY TEST
    # ONLY FIRST TASK
    # --------------------------------------------------------

    test_dataset = dataset.select(range(1))

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    for index, item in enumerate(
        tqdm(
            test_dataset,
            ascii=True,
            desc="Evaluating"
        )
    ):

        print(
            f"\nProcessing task {index + 1}/{len(dataset)}",
            flush=True
        )

        task_id = item["task_id"]
        question = item["Question"]
        file_name = item.get("file_name", "")

        prompt = question

        if file_name:
            prompt += (
                f"\n\nAttached file: {file_name}"
            )

        # ----------------------------------------------------
        # RUN AGENT
        # ----------------------------------------------------

        try:

            print(
                "\n========== PROMPT SENT TO AGENT ==========",
                flush=True
            )

            print(
                prompt,
                flush=True
            )

            print(
                "==========================================\n",
                flush=True
            )

            response = agent.run(prompt)

            if response is None:

                prediction = "Error: Empty response"

            else:

                prediction = str(response).strip()

            print(
                "\n========== MODEL ANSWER ==========",
                flush=True
            )

            print(
                prediction,
                flush=True
            )

            print(
                "==================================\n",
                flush=True
            )

        except Exception as e:

            print(
                "\n========== AGENT ERROR ==========",
                flush=True
            )

            print(
                "TASK ID:",
                repr(task_id),
                flush=True
            )

            print(
                "ERROR TYPE:",
                type(e).__name__,
                flush=True
            )

            print(
                "ERROR:",
                repr(e),
                flush=True
            )

            if e.__cause__ is not None:

                print(
                    "CAUSE TYPE:",
                    type(e.__cause__).__name__,
                    flush=True
                )

                print(
                    "CAUSE:",
                    repr(e.__cause__),
                    flush=True
                )

            if e.__context__ is not None:

                print(
                    "CONTEXT TYPE:",
                    type(e.__context__).__name__,
                    flush=True
                )

                print(
                    "CONTEXT:",
                    repr(e.__context__),
                    flush=True
                )

            traceback.print_exc()

            prediction = f"Error: {repr(e)}"

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        result_entry = {
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
                    result_entry,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print(
        "\nEvaluation finished!",
        flush=True
    )

    print(
        f"Results saved to {output_file}",
        flush=True
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()