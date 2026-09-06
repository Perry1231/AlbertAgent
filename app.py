import os
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, GradioUI
from tools import ALL_TOOLS

# 1. Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 2. Initialize the model through the Inference API (computations are performed on the HF side)
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token
)

# 3. Gather all tools together
search_tool = DuckDuckGoSearchTool()
tools = [search_tool] + ALL_TOOLS

# 4. Create the agent
agent = CodeAgent(
    model=model,
    tools=tools,
    max_steps=6,
    verbosity_level=1
)

# 5. Launch the single Gradio UI
if __name__ == "__main__":
    GradioUI(agent).launch()