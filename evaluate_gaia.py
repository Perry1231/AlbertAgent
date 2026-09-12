import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

from smolagents import OpenAIServerModel, CodeAgent
from tools import ALL_TOOLS

load_dotenv()

# Отримуємо ключ Groq та HF токен
groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not groq_key:
    raise ValueError("❌ GROQ_API_KEY не знайдено у файлі .env!")

# Підключаємо Llama 3.3 70B через безкоштовний та швидкий Groq API
groq_key = os.getenv("GROQ_API_KEY")

model = OpenAIServerModel(
    model_id="llama-3.3-70b-versatile",
    api_base="https://api.groq.com/openai/v1",  # Переконуємось, що endpoint Groq, а не OpenRouter
    api_key=groq_key,
)

# Створення CodeAgent з інструментами
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
            if response is None:
                prediction = "Error: Model returned empty response (None)"
            else:
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