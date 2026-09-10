import os
from dotenv import load_dotenv
from smolagents import (
    AgentLogger,
    CodeAgent,
    DuckDuckGoSearchTool,
    E2BExecutor,
    InferenceClientModel,
    GradioUI
)
from tools import ALL_TOOLS

# 1. Завантажуємо змінні оточення з .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 2. Ініціалізуємо модель
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token
)

# 3. Збираємо інструменти та логер
search_tool = DuckDuckGoSearchTool()
tools = [search_tool] + ALL_TOOLS
logger = AgentLogger()

# 4. Список додаткових бібліотек для середовища E2B
additional_imports = [
    "pandas", "numpy", "pillow", "pymupdf", "requests", "bs4"
]

# 5. Створюємо агента (збільшено max_steps для складних завдань GAIA)
agent = CodeAgent(
    model=model,
    tools=tools,
    max_steps=12,
    verbosity_level=1,
    add_base_tools=True,
    executor=E2BExecutor(additional_imports=additional_imports, logger=logger)
)

# 6. Запуск GradioUI
if __name__ == "__main__":
    ui = GradioUI(agent)
    ui.launch(share=False)