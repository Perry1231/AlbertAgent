import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


import re

def normalize(text: str) -> str:
    s = str(text).strip().lower()
    s = s.strip('"\'')
    s = re.sub(r"\s*([,;])\s*", r"\1 ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def main():
    sub_path = BASE_DIR / "submission.jsonl"
    if not sub_path.exists():
        print("Файл submission.jsonl не знайдено.")
        return

    predictions = {}
    with open(sub_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("task_id") or row.get("id")
            ans = row.get("model_answer") or row.get("answer") or row.get("prediction") or ""
            if tid:
                predictions[tid] = str(ans).strip()

    if not predictions:
        print("submission.jsonl порожній.")
        return

    print("🔄 Завантаження еталонних відповідей GAIA (validation split)...")
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=os.getenv("HF_TOKEN"))

    correct = 0
    total = 0

    print("\n--- ПОРІВНЯННЯ ВІДПОВІДЕЙ ---")
    for item in ds:
        tid = item.get("task_id")
        if tid not in predictions:
            continue

        total += 1
        pred = predictions[tid]
        gold = str(item.get("Final answer") or "").strip()

        is_ok = normalize(pred) == normalize(gold)
        if is_ok:
            correct += 1

        icon = "✅" if is_ok else "❌"
        print(f"{icon} ID: {tid[:8]}... | Ваша відповідь: '{pred}' | Еталон: '{gold}'")

    acc = (correct / total * 100) if total else 0.0
    print(f"\n📊 ПІДСУМОК: {correct}/{total} правильних відповідей ({acc:.2f}% Accuracy)")


if __name__ == "__main__":
    main()