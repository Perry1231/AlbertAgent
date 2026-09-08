import os
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, E2BExecutor, InferenceClientModel, GradioUI
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
    verbosity_level=1,
    add_base_tools=True,
    executor=E2BExecutor(), # Виконує код у віддаленій MicroVM
    authorized_imports=[     # Для локального інтерпретатора (якщо приберете E2BExecutor)
        "pandas", "numpy", "PIL", "fitz", "requests", 
        "bs4", "json", "csv", "zipfile", "os", "re", "math"
    ]
)

# 5. Запуск GradioUI
if __name__ == "__main__":
    # Запуск тестового запиту (якщо потрібно перевірити в консолі)
    # response = agent.run("take a JSON file with a list of numbers, calculate the sum and average, and save the results to a new text file.")
    
    # Запуск Gradio UI
    ui = GradioUI(agent)
    ui.launch(share=False)