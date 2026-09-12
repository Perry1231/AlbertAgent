import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from smolagents import OpenAIServerModel, CodeAgent
from tools import ALL_TOOLS

load_dotenv()

# Перевірка ключів доступу
openrouter_key = os.getenv("OPENROUTER_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not openrouter_key:
    raise ValueError("❌ OPENROUTER_API_KEY не знайдено у файлі .env!")

# Ініціалізація безкоштовної моделі через OpenRouter
model = OpenAIServerModel(
    model_id="qwen/qwen-2.5-coder-32b-instruct:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
)

# Створення CodeAgent з підключеними інструментами
agent = CodeAgent(
    tools=ALL_TOOLS,
    model=model,
    additional_authorized_imports=["requests", "bs4", "pandas", "numpy", "math"],
)

def main():
    print("🚀 Завантаження датасету GAIA...")
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=hf_token)
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
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    print(f"\n✅ Тестування завершено! Результати збережено у {output_file}")

if __name__ == "__main__":
    main()