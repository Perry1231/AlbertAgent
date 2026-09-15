import os
import sys
import io

# ============================================================
# FORCE UTF-8
# ============================================================

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

# Безпечне переналаштування stdout/stderr
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

import json
import logging

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from smolagents import OpenAIServerModel, CodeAgent
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
    raise ValueError("GROQ_API_KEY not found in .env file!")


# ============================================================
# MODEL
# ============================================================

model = OpenAIServerModel(
    model_id="llama-3.3-70b-versatile",
    api_base="https://api.groq.com/openai/v1",
    api_key=groq_key,
)


# ============================================================
# AGENT
# ============================================================

agent = CodeAgent(
    tools=ALL_TOOLS,
    model=model,
    verbosity_level=0,
    additional_authorized_imports=[
        "requests",
        "bs4",
        "pandas",
        "numpy",
        "math"
    ],
)


# ============================================================
# EVALUATION
# ============================================================

def main():

    print("STEP 1: Starting program...", flush=True)

    try:
        print("STEP 2: Loading GAIA dataset...", flush=True)

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
        print("\n!!! ERROR WHILE LOADING DATASET !!!", flush=True)
        print("Error type:", type(e).__name__, flush=True)
        print("Error:", repr(e), flush=True)

        import traceback
        traceback.print_exc()

        raise

    output_file = "submission.jsonl"

    print(
        f"STEP 4: Starting evaluation on {len(dataset)} tasks...",
        flush=True
    )

    # Очистити старий submission
    with open(
        output_file,
        "w",
        encoding="utf-8",
        errors="replace"
    ):
        pass

    for index, item in enumerate(
        tqdm(dataset, ascii=True, desc="Evaluating")
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
            prompt += f"\n\nAttached file: {file_name}"

        try:

            response = agent.run(prompt)

            if response is None:
                prediction = "Error: Empty response"
            else:
                prediction = str(response).strip()

        except Exception as e:

            print(
                "\n!!! ERROR IN AGENT !!!",
                flush=True
            )

            print(
                "Task ID:",
                task_id,
                flush=True
            )

            print(
                "Error type:",
                type(e).__name__,
                flush=True
            )

            print(
                "Error:",
                repr(e),
                flush=True
            )

            import traceback
            traceback.print_exc()

            prediction = f"Error: {repr(e)}"

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

    print(
        "\nEvaluation finished!",
        flush=True
    )

    print(
        f"Results saved to {output_file}",
        flush=True
    )