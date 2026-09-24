import os
import json

from dotenv import load_dotenv
from datasets import load_dataset

from custom_agent import run_agent


# ==========================================
# CONFIG
# ==========================================

load_dotenv()

GAIA_LEVEL = "2023_level1"

OUTPUT_FILE = "gaia_results.jsonl"


# ==========================================
# LOAD GAIA
# ==========================================

print("=" * 60)
print("GAIA EVALUATION")
print("=" * 60)

print(f"Loading: {GAIA_LEVEL}")

dataset = load_dataset(
    "gaia-benchmark/GAIA",
    GAIA_LEVEL,
    split="validation",
)

print(f"Tasks loaded: {len(dataset)}")
print()
print("DATASET COLUMNS:")
print(dataset.column_names)

print()
print("FIRST TASK:")
print(dataset[0])

# ==========================================
# RUN
# ==========================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as output:

    for index, task in enumerate(dataset):
        task_id = task["task_id"]
        question = task["Question"]

        print()
        print("=" * 80)
        print(f"TASK {index + 1}/{len(dataset)}")
        print(f"ID: {task_id}")
        print("=" * 80)

        print()
        print("QUESTION:")
        print(question)

        try:

            answer = run_agent(question)

        except Exception as e:

            answer = f"ERROR: {type(e).__name__}: {e}"

            print()
            print("AGENT ERROR:")
            print(answer)

        result = {
            "task_id": task_id,
            "model_answer": answer,
        }

        output.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )

        output.flush()

        print()
        print("SAVED RESULT:")
        print(result)


print()
print("=" * 60)
print("GAIA RUN FINISHED")
print("=" * 60)

print(f"Results saved to: {OUTPUT_FILE}")