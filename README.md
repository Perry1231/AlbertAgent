# AlbertAgent

AI agent for solving tasks from the [GAIA Benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA).

The project uses an LLM through the Groq API together with `smolagents` and a set of external tools to solve multi-step research and reasoning tasks.

## Features

* AI agent based on `smolagents`
* `openai/gpt-oss-120b` model via Groq
* Tool calling for external operations
* GAIA Benchmark evaluation
* Automatic generation of `submission.jsonl`
* API keys loaded from `.env`
* Ability to test a single task before running the full benchmark

## Project Structure

```text
AlbertAgent/
│
├── evaluate_gaia.py      # Main GAIA evaluation script
├── tools.py              # Agent tools
├── .env                  # API keys (not committed)
├── .gitignore
├── submission.jsonl      # Generated evaluation results
└── README.md
```

## Requirements

* Python 3.10+
* Groq API key
* Hugging Face token
* Internet connection

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AlbertAgent.git
cd AlbertAgent
```

Create a virtual environment.

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist yet:

```bash
pip install smolagents openai datasets python-dotenv tqdm
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

Never commit `.env` to GitHub.

Add it to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
submission.jsonl
```

## Model

The project currently uses:

```text
openai/gpt-oss-120b
```

through the Groq OpenAI-compatible API:

```text
https://api.groq.com/openai/v1
```

The agent uses `ToolCallingAgent` from `smolagents` to allow the model to call available tools while solving tasks.

## Running the Evaluator

Run:

```powershell
python evaluate_gaia.py
```

The program will:

1. Load the GAIA validation dataset.
2. Select the configured tasks for evaluation.
3. Send each task to the AI agent.
4. Allow the model to use available tools.
5. Receive the final answer.
6. Save the result to `submission.jsonl`.

Example output:

```text
STEP 1: Starting program...
STEP 2: Loading GAIA dataset...
STEP 3: Dataset loaded. Tasks: 165
STEP 4: Starting evaluation...

Processing task 1/165

========== PROMPT SENT TO AGENT ==========

...

========== MODEL ANSWER ==========

...

Evaluation finished!
Results saved to submission.jsonl
```

## Testing With One Task

During development, it is recommended to test only one task first.

In `evaluate_gaia.py`:

```python
test_dataset = dataset.select(range(1))
```

This runs only the first task.

After the agent works correctly, the project can be configured to evaluate the full dataset:

```python
test_dataset = dataset
```

## Output

Results are saved to:

```text
submission.jsonl
```

Each line contains a task ID and the model's answer:

```json
{
  "task_id": "example-task-id",
  "model_answer": "Example answer"
}
```

## Agent Architecture

The basic workflow is:

```text
GAIA Dataset
     |
     v
   Task
     |
     v
ToolCallingAgent
     |
     v
GPT-OSS-120B
     |
     +---------> Tool
     |             |
     |             v
     |        Tool Result
     |             |
     <-------------+
     |
     v
Final Answer
     |
     v
submission.jsonl
```

## Tools

Available tools are defined in:

```text
tools.py
```

and collected through:

```python
ALL_TOOLS
```

The agent can use these tools to perform operations required by individual GAIA tasks.

## Troubleshooting

### GROQ_API_KEY not found

Make sure `.env` exists in the project root:

```env
GROQ_API_KEY=your_key
```

### Model not found / 404

Make sure the configured model is available through Groq.

Current model:

```text
openai/gpt-oss-120b
```

### Unicode / encoding errors on Windows

The evaluator configures UTF-8 output:

```python
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
```

and configures `stdout` and `stderr` to use UTF-8.

### Agent does not call tools correctly

Make sure the project uses:

```python
from smolagents import ToolCallingAgent
```

and:

```python
agent = ToolCallingAgent(
    tools=ALL_TOOLS,
    model=model,
    verbosity_level=0,
)
```

## Security

Do not publish API keys or tokens.

Never commit:

```text
.env
```

or any file containing:

```text
GROQ_API_KEY
HF_TOKEN
```

If a key is accidentally pushed to GitHub, revoke it immediately and generate a new one.

## Development

Recommended development workflow:

```text
1. Modify the agent or tools
        |
        v
2. Run one GAIA task
        |
        v
3. Check the model answer
        |
        v
4. Fix errors
        |
        v
5. Run several tasks
        |
        v
6. Run the full benchmark
```

This makes debugging easier than immediately running all GAIA tasks.

## License

this project is under MIT License
