import os
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, GradioUI
from tools import ALL_TOOLS

# 1. Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 2. Initialize model
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token
)

# 3. Assemble tools
search_tool = DuckDuckGoSearchTool()
tools = [search_tool] + ALL_TOOLS

# 4. Initialize CodeAgent
agent = CodeAgent(
    model=model,
    tools=tools,
    max_steps=6,
    verbosity_level=1
)

# 5. Build Gradio UI and extract underlying Blocks app
if __name__ == "__main__":
    ui = GradioUI(agent)
    
    # Access the raw Gradio interface instance directly
    demo = ui.app if hasattr(ui, "app") else ui
    
    # Launch with explicit container parameters
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        ssr=False
    )