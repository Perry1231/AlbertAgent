import os
import json
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

print("🔄 Завантаження еталонних відповідей GAIA (validation split)...")
dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=hf_token)

# Словник правильних відповідей: task_id -> Final answer
ground_truth = {item["task_id"]: str(item["Final answer"]).strip().lower() for item in dataset}

# Зчитуємо відповіді з вашого submission.jsonl
predictions = {}
submission_file = "submission.jsonl"

if not os.path.exists(submission_file):
    print(f"❌ Файл {submission_file} не знайдено! Спочатку запустіть evaluate_gaia.py")
    exit()

with open(submission_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            predictions[data["task_id"]] = str(data["model_answer"]).strip().lower()

correct = 0
total = len(predictions)

print("\n--- ПОРІВНЯННЯ ВІДПОВІДЕЙ ---")
for task_id, pred in predictions.items():
    gt = ground_truth.get(task_id, "N/A")
    is_correct = (pred == gt)
    if is_correct:
        correct += 1
    status = "✅" if is_correct else "❌"
    print(f"{status} ID: {task_id[:8]}... | Ваша відповідь: '{pred}' | Еталон: '{gt}'")

if total > 0:
    accuracy = (correct / total) * 100
    print(f"\n📊 ПІДСУМОК: {correct}/{total} правильних відповідей ({accuracy:.2f}% Accuracy)")
else:
    print("⚠️ Файл submission.jsonl порожній.")