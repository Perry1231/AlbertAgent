# -*- coding: utf-8 -*-

import os
import sys
import io
import json
import logging
import traceback
from pathlib import Path
from huggingface_hub import hf_hub_download

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
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from custom_agent import run_agent


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
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
        "\n============================================================\n"
        "ERROR: GROQ_API_KEY not found!\n"
        "============================================================\n"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_print(text=""):
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(
            str(text).encode("utf-8", errors="replace").decode("utf-8", errors="replace"),
            flush=True,
        )


def find_attachment(file_name):
    if not file_name:
        return None

    file_name = str(file_name).strip()
    if not file_name:
        return None

    GAIA_FILES_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [
        GAIA_FILES_DIR / file_name,
        BASE_DIR / "files" / file_name,
        BASE_DIR / "attachments" / file_name,
        BASE_DIR / file_name,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    # Якщо файлу немає локально — докачуємо з Hugging Face (validation split)
    try:
        downloaded_path = hf_hub_download(
            repo_id="gaia-benchmark/GAIA",
            filename=f"2023/validation/{file_name}",
            repo_type="dataset",
            token=HF_TOKEN,
            local_dir=GAIA_FILES_DIR,
        )
        return Path(downloaded_path)
    except Exception as e:
        safe_print(f"Warning: Could not download attachment '{file_name}': {e}")
        return None


def build_prompt(question, file_name=""):
    prompt_parts = [
        "Solve the following GAIA task carefully.",
        "",
        "QUESTION:",
        str(question),
    ]

    if file_name:
        prompt_parts.append("")
        prompt_parts.append(f"ATTACHED FILE NAME: {file_name}")
        attachment = find_attachment(file_name)

        if attachment:
            prompt_parts.append(f"LOCAL FILE PATH: {attachment}")
            prompt_parts.append(
                "The attached file is available locally. "
                "Inspect it if the task requires information from the file."
            )
        else:
            prompt_parts.append(
                "The attachment is referenced by name, but "
                "its local file was not found automatically."
            )

    prompt_parts.append("")
    prompt_parts.append("Return only the concise final answer to the question.")

    return "\n".join(prompt_parts)


def load_completed_tasks(output_file):
    """
    Зчитує вже пройдені task_id із submission.jsonl, щоб не перезаписувати їх
    і не витрачати токени повторно.
    """
    completed = set()
    if not output_file.exists():
        return completed

    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                tid = data.get("task_id")
                ans = str(data.get("model_answer", "")).strip()
                # Пропускаємо тільки якщо є реальна відповідь, а не помилка API/ліміту
                if tid and ans and not ans.startswith(("Error:", "ERROR:", "Agent API error:")):
                    completed.add(tid)
            except json.JSONDecodeError:
                continue

    return completed


def save_result(output_file, task_id, prediction):
    result = {
        "task_id": task_id,
        "model_answer": prediction,
    }

    with open(output_file, "a", encoding="utf-8", errors="replace") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def is_daily_limit_error(text):
    lower = str(text).lower()
    return (
        "tokens per day" in lower
        or "tpd" in lower
        or "daily token limit" in lower
    )


# ============================================================
# DATASET
# ============================================================

def load_gaia_dataset():
    safe_print("STEP 1: Loading GAIA dataset...")
    dataset = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        split="validation",
        token=HF_TOKEN,
    )
    safe_print(f"STEP 2: Dataset loaded. Tasks: {len(dataset)}")
    return dataset


# ============================================================
# MAIN
# ============================================================

def main():
    safe_print("")
    safe_print("============================================================")
    safe_print("GAIA BENCHMARK (custom_agent + checkpointing)")
    safe_print("============================================================")
    safe_print("")

    dataset = load_gaia_dataset()

    limit_env = os.getenv("GAIA_LIMIT", "5").strip().lower()

    if limit_env == "all":
        test_dataset = dataset
    else:
        try:
            limit = max(1, int(limit_env))
        except ValueError:
            safe_print(f"Invalid GAIA_LIMIT={limit_env}. Using 5.")
            limit = 5

        limit = min(limit, len(dataset))
        test_dataset = dataset.select(range(limit))

    safe_print(f"STEP 3: Tasks selected: {len(test_dataset)}")
    safe_print(f"STEP 4: Output file: {OUTPUT_FILE}")

    # НЕ очищаємо файл ("w"), а зчитуємо вже виконані завдання
    completed_tasks = load_completed_tasks(OUTPUT_FILE)
    safe_print(f"STEP 5: Already completed tasks (will be skipped): {len(completed_tasks)}")
    safe_print("")

    successful = 0
    skipped = 0
    failed = 0

    for index, item in enumerate(tqdm(test_dataset, ascii=True, desc="Evaluating")):
        task_number = index + 1
        task_id = item.get("task_id", f"unknown-{task_number}")

        # Пропускаємо вже пройдені питання
        if task_id in completed_tasks:
            skipped += 1
            continue

        question = item.get("Question", "")
        file_name = item.get("file_name", "")

        safe_print("")
        safe_print("============================================================")
        safe_print(f"TASK {task_number}/{len(test_dataset)} | ID: {task_id}")
        safe_print("============================================================")

        prompt = build_prompt(question=question, file_name=file_name)

        try:
            prediction = str(run_agent(prompt) or "").strip()
            if not prediction:
                prediction = "Error: Empty response"

        except KeyboardInterrupt:
            safe_print("\nEvaluation interrupted by user.")
            raise

        except Exception as e:
            traceback.print_exc()
            prediction = f"Error: {type(e).__name__}: {e}"

        # Миттєва зупинка при вичерпанні денного ліміту Groq (TPD)
        if is_daily_limit_error(prediction):
            safe_print("")
            safe_print("============================================================")
            safe_print("GROQ DAILY TOKEN LIMIT (TPD) REACHED!")
            safe_print("Stopping immediately so progress is saved and no time is wasted.")
            safe_print("============================================================")
            break

        safe_print("")
        safe_print("================ MODEL ANSWER =================")
        safe_print(prediction)
        safe_print("=================================================")

        if prediction.startswith(("Error:", "ERROR:", "Agent API error:")):
            failed += 1
        else:
            successful += 1
            completed_tasks.add(task_id)

        save_result(
            output_file=OUTPUT_FILE,
            task_id=task_id,
            prediction=prediction,
        )

    safe_print("")
    safe_print("============================================================")
    safe_print("EVALUATION FINISHED")
    safe_print("============================================================")
    safe_print(f"Total selected: {len(test_dataset)}")
    safe_print(f"Skipped (already done): {skipped}")
    safe_print(f"Newly successful: {successful}")
    safe_print(f"Failed: {failed}")
    safe_print(f"Results saved in: {OUTPUT_FILE}")
    safe_print("============================================================")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        safe_print("\nProgram stopped by user.")
        sys.exit(130)
    except Exception as e:
        safe_print(f"\nFATAL ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)