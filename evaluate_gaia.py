import json
import os
from datasets import load_dataset
from tqdm import tqdm

# 1. Import the agent from app.py
from app import agent

def main():
    print("Loading GAIA dataset...")
    # 'validation' — для перевірки та тестування локально
    # 'test' — для фінальної подачі на Leaderboard
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

    output_file = "submission.jsonl"
    
    # Очищаємо файл перед новим запуском
    with open(output_file, "w", encoding="utf-8") as f:
        pass

    print(f"📊 Starting evaluation of {len(dataset)} GAIA tasks...")

    for item in tqdm(dataset):
        task_id = item["task_id"]
        question = item["Question"]
        file_name = item.get("file_name", "")

        # Формуємо промпт для агента
        prompt = question
        if file_name:
            prompt += f"\n\nAttached file: {file_name}"

        try:
            # Запускаємо вашого агента
            response = agent.run(prompt)
            prediction = str(response).strip()
        except Exception as e:
            prediction = f"Error: {str(e)}"

        # Формат, який вимагає GAIA Leaderboard
        result_entry = {
            "task_id": task_id,
            "model_answer": prediction
        }

        # Одразу дописуємо в submission.jsonl (щоб зберегти прогрес, якщо щось впаде)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_entry) + "\n")

    print(f"\n✅ Evaluation completed! Results saved to: {output_file}")

if __name__ == "__main__":
    main()