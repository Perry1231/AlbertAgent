import os
import json
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Перевірка зчитування токена
if not hf_token:
    raise ValueError("❌ HF_TOKEN не знайдено у файлі .env! Перевірте вміст .env")

print(f"🔑 Зчитано HF_TOKEN: {hf_token[:5]}...{hf_token[-4:]}")
login(token=hf_token)

# Імпортуємо агента
from app import agent

def main():
    print("🚀 Завантаження датасету GAIA...")
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="test", token=hf_token)
    output_file = "submission.jsonl"
    print(f"📊 Початок прогону {len(dataset)} завдань GAIA...")

    for item in tqdm(dataset):
        task_id = item["task_id"]
        question = item["Question"]
        file_name = item.get("file_name", "")

        prompt = question
        if file_name:
            prompt += f"\n\nAttached file: {file_name}"

        try:
            response = agent.run(prompt)
            prediction = str(response).strip()
        except Exception as e:
            prediction = f"Error: {str(e)}"

        result_entry = {
            "task_id": task_id,
            "model_answer": prediction
        }

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry) + "\n")

    print(f"\n✅ Тестування завершено! Результати збережено у {output_file}")

if __name__ == "__main__":
    main()