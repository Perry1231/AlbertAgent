import os
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, GradioUI
from tools import ALL_TOOLS

# 1. Завантажуємо токен з .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 2. Ініціалізуємо модель
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token
)

# 3. Збираємо інструменти
search_tool = DuckDuckGoSearchTool()
tools = [search_tool] + ALL_TOOLS

# 4. Створюємо агента
agent = CodeAgent(
    model=model,
    tools=tools,
    max_steps=6,
    verbosity_level=1
)

# 5. Запуск GradioUI без некоректних аргументів
if __name__ == "__main__":
    ui = GradioUI(agent)
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )