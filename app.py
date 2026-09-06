import os
import datetime
import pytz
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool, GradioUI
from tools import ALL_TOOLS
import gradio as gr
import spaces
import torch
import datetime
import pytz

zero = torch.Tensor([0]).cuda()
print(zero.device) # <-- 'cpu' 🤔

@spaces.GPU
def greet(n):
    print(zero.device) # <-- 'cuda:0' 🤗
    return f"Hello {zero + n} Tensor"

demo = gr.Interface(fn=greet, inputs=gr.Number(), outputs=gr.Text())
demo.launch()

# 1. Launch tocken from .env file
load_dotenv()
hf_token = os.getenv("HF_TOKEN")


# 3. Initialize the model
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token
)

# 4. Connect the tools to the agent
search_tool = DuckDuckGoSearchTool()
tools = [search_tool] + ALL_TOOLS

# 5. Creating agent
agent = CodeAgent(model=model, tools=tools)

# 6. Launch the Gradio UI
if __name__ == "__main__":
    GradioUI(agent).launch()