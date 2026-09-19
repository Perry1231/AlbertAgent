import os
from dotenv import load_dotenv

from smolagents import (
    AgentLogger,
    CodeAgent,
    DuckDuckGoSearchTool,
    E2BExecutor,
    GradioUI,
    OpenAIServerModel,
)

from tools import ALL_TOOLS


# ============================================================
# 1. Завантаження .env
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError(
        "GROQ_API_KEY не знайдено у файлі .env"
    )


# ============================================================
# 2. Модель Groq
# ============================================================

model = OpenAIServerModel(
    model_id="openai/gpt-oss-120b",
    api_base="https://api.groq.com/openai/v1",
    api_key=groq_api_key,

    # Низька температура для стабільнішого виконання коду
    temperature=0.2,

    # Не дозволяємо моделі генерувати величезні відповіді
    max_tokens=1024,
)


# ============================================================
# 3. Інструменти
# ============================================================

search_tool = DuckDuckGoSearchTool()

tools = [search_tool] + ALL_TOOLS


# ============================================================
# 4. Logger
# ============================================================

logger = AgentLogger()


# ============================================================
# 5. Бібліотеки для E2B
# ============================================================

additional_imports = [
    "pandas",
    "numpy",
    "pillow",
    "pymupdf",
    "requests",
    "bs4",
]


# ============================================================
# 6. E2B Executor
# ============================================================

executor = E2BExecutor(
    additional_imports=additional_imports,
    logger=logger,
)


# ============================================================
# 7. Agent
# ============================================================

agent = CodeAgent(
    model=model,

    tools=tools,

    # Максимум 5 кроків
    max_steps=5,

    # Показуємо мінімум зайвої інформації
    verbosity_level=1,

    # Додаємо стандартні tools smolagents
    add_base_tools=True,

    executor=executor,
)


# ============================================================
# 8. Запуск Gradio
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AlbertAgent starting...")
    print("=" * 60)

    print("Model: openai/gpt-oss-120b")
    print("Provider: Groq")
    print("Max tokens: 1024")
    print("Max steps: 5")
    print(f"Tools loaded: {len(tools)}")

    print("=" * 60)

    ui = GradioUI(agent)

    ui.launch(
        share=False
    )