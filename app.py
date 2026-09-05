import os
import datetime
import pytz
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool, GradioUI

# 1. Завантажуємо токен з .env файлу
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 2. Створюємо кастомний інструмент (Tool)
@tool
def get_current_time(timezone: str) -> str:
    """
    Отримує поточний час у вказаному часовому поясі.

    Args:
        timezone: Часовий пояс (наприклад, 'Europe/Kyiv', 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"Поточний час у {timezone}: {local_time}"
    except Exception as e:
        return f"Помилка часового поясу '{timezone}': {str(e)}"

# 3. Ініціалізуємо модель з Hugging Face
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token
)

# 4. Підключаємо інструменти (пошуковик + наш кастомний)
search_tool = DuckDuckGoSearchTool()
tools = [search_tool, get_current_time]

# 5. Створюємо агента
agent = CodeAgent(
    model=model,
    tools=tools,
    max_steps=6,
    verbosity_level=1
)

# 6. Запускаємо вебінтерфейс Gradio
if __name__ == "__main__":
    GradioUI(agent).launch()