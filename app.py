import os
import datetime
import pytz
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, tool, GradioUI
import gradio as gr
import spaces
import torch

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

# 2. Створюємо кастомний інструмент (Tool)
@tool
def get_current_time(timezone: str) -> str:
    """
    Getts the current time in the specified timezone.
    Args:
        timezone: The timezone for which to get the current time (e.g., 'Europe/Kyiv', 'America/New_York').
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"Current time in {timezone}: {local_time}"
    except Exception as e:
        return f"Error with timezone '{timezone}': {str(e)}"

# 3. Initialize the model
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token
)

# 4. Connect the tools to the agent
search_tool = DuckDuckGoSearchTool()
tools = [search_tool, get_current_time]

# 5. Creating agent
agent = CodeAgent(
    model=model,
    tools=tools,
    max_steps=12,
    verbosity_level=1
)

# 6. Launch the Gradio UI
if __name__ == "__main__":
    GradioUI(agent).launch()