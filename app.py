import os
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, E2BExecutor, InferenceClientModel, GradioUI
from tools import ALL_TOOLS

# 1. load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 2. Initialize the model
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token
)

# 3. Initialize the search tool and all other tools
search_tool = DuckDuckGoSearchTool()
tools = [search_tool] + ALL_TOOLS

# 4. Create the agent
agent = CodeAgent(
    model=model,
    tools=tools,
    max_steps=6,
    verbosity_level=1,
    add_base_tools=True,
    authorized_imports=[     # Для локального інтерпретатора (якщо приберете E2BExecutor)
        "pandas", "numpy", "PIL", "fitz", "requests", 
        "bs4", "json", "csv", "zipfile", "os", "re", "math"
    ]
)

# 5. Launch GradioUI
if __name__ == "__main__":
    # Launch test query (if needed to verify in console)
    # response = agent.run("take a JSON file with a list of numbers, calculate the sum and average, and save the results to a new text file.")
    
    # Launch Gradio UI
    ui = GradioUI(agent)
    ui.launch(share=False)