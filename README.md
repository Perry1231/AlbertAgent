#  AlbertAgent

**AlbertAgent** is an autonomous AI agent built on top of the [`smolagents`](https://github.com/huggingface/smolagents) framework and powered by the **Qwen2.5-Coder-32B-Instruct** LLM backbone. It is designed to tackle complex, multimodal reasoning tasks and evaluate performance on the **GAIA (General AI Assistants)** benchmark.

The agent features secure code execution within an **E2B** sandbox environment, system-level Bash integration, web scraping, data analysis, and multimodal processing capabilities (audio, image, and video).

---

##  Key Features

- **LLM Backbone:** `Qwen/Qwen2.5-Coder-32B-Instruct` served via Hugging Face Inference API.
- **Code Execution:** Secure Python execution and data analysis using the **E2BExecutor** sandbox.
- **Extensive Toolset (Custom Tools):**
  - **Web & Search:** Web scraping (`BeautifulSoup4`) and webpage fetching.
  - **Bash & System:** System command execution, `pip` package installation, and file management.
  - **Data Analysis:** JSON/CSV parsing and analytics via `pandas` and `numpy`.
  - **Media Processing:** Audio transcription (`openai-whisper`), image processing (`Pillow`), and video frame extraction (`OpenCV`).
- **Benchmarking:** Built-in evaluation pipeline for the **GAIA** dataset (`gaia-benchmark/GAIA`).

---

##  Project Structure

```text
AlbertAgent/
├── tools/                  # Modular custom tools directory
│   ├── __init__.py         # Exports ALL_TOOLS list
│   ├── api_tools.py        # External API tools (Crypto, Weather)
│   ├── audio_tools.py      # Audio processing & transcription (Whisper)
│   ├── bash_tools.py       # System shell and bash execution
│   ├── code_execution.py   # Dynamic Python code execution
│   ├── file_tools.py       # File reading/writing utilities
│   ├── image_tools.py      # Image processing tools
│   ├── math_tools.py       # Basic arithmetic operations
│   ├── video_tools.py      # Video frame extraction
│   ├── web_tool.py         # Web scraping and navigation
│   └── tools.py            # Helper tools (time, discount calculation)
├── app.py                  # CodeAgent setup and Gradio UI
├── evaluate_gaia.py        # GAIA evaluation runner script
├── check_score.py          # Benchmark accuracy calculator
├── .env                    # Environment variables (HF_TOKEN, E2B_API_KEY)
└── requirements.txt        # Project dependencies

```

---

##  Quick Start

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/AlbertAgent.git](https://github.com/your-username/AlbertAgent.git)
cd AlbertAgent

```

### 2. Set Up a Virtual Environment

**Windows (PowerShell):**

```powershell
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process
python -m venv venv
.\venv\Scripts\Activate.ps1

```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install smolagents pandas numpy pillow pymupdf requests beautifulsoup4 pytz openai-whisper opencv-python python-dotenv datasets e2b-code-interpreter

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_token_here
E2B_API_KEY=your_e2b_api_key_here  # Optional: for enhanced code sandbox execution

```

---

## 💻 Usage

### Interactive Web UI (Gradio)

To launch the interactive web interface in your browser:

```bash
python app.py

```

### Running GAIA Benchmark Evaluation

To execute the evaluation suite against the GAIA benchmark:

```bash
python evaluate_gaia.py

```

Once the run completes, evaluate your score:

```bash
python check_score.py

```

---

##  Development Guidelines for Custom Tools

When adding new tools to the `tools/` directory, ensure compliance with `smolagents` AST validation requirements:

* **Local Imports:** All external package imports (`requests`, `json`, `os`, `subprocess`, `whisper`, etc.) **must be placed inside the function body** decorated with `@tool`.
* **Type Annotations & Docstrings:** Every function requires clear type hints and Google-style docstrings describing the arguments (`Args:`) and return values (`Returns:`).

---

##  License

This project is licensed under the MIT License.
