# -*- coding: utf-8 -*-

import os
import re
import json
import time
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

import tools


# ============================================================
# ENV & CLIENT SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY not found")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=60.0,
    max_retries=1,
)

MODEL = "nvidia/nemotron-3.5-super-vl:free"


# ============================================================
# CONFIG
# ============================================================

MAX_STEPS = 5
MAX_TOOL_RESULT_CHARS = 2500
MAX_HISTORY_BLOCKS = 3

MAX_OUTPUT_TOKENS = 1000
MAX_FINAL_TOKENS = 1500

MAX_RETRIES = 3
SEARCH_MAX_CALLS = 3
SAFE_CONTEXT_CHARS = 14000


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are AlbertAgent, an accurate, concise, tool-using AI assistant designed for GAIA benchmark tasks.

Follow these rules strictly:
1. Read the user question carefully and identify the exact entity, number, word, or fact requested.
2. Use tools only when external information, web pages, files, or calculations are needed.
3. Never repeat the exact same tool call with the same arguments.
4. Keep search queries short, specific, and keyword-focused.
5. Search at most 3 times per task. After searching, if a relevant URL is found, open and read it with smart_web_scraper instead of searching again.
6. NEVER invent or guess URLs (never use placeholders like XXXX or IDs you did not see in search results). Only scrape exact URLs returned by search.
7. If a local file path is provided in the prompt, inspect or read that local file using the available file/code tools before answering.
8. When performing math, counting, or data parsing, use a code/calculator tool if available rather than guessing.
9. Once you have enough evidence to answer accurately, stop calling tools immediately and return the final answer.
10. Output ONLY the exact final answer without explanations, filler, markdown formatting, or full sentences unless explicitly asked.
11. If the answer is a symbol or character name (like backtick, comma, asterisk), write its plain English word name without the symbol in brackets unless asked for the symbol itself.
"""


# ============================================================
# DYNAMIC TOOL REGISTRY (FROM tools.py)
# ============================================================

TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(annotation: Any) -> Dict[str, Any]:
    if annotation in TYPE_MAP:
        return {"type": TYPE_MAP[annotation]}

    origin = getattr(annotation, "__origin__", None)
    if origin in (list, List):
        args = getattr(annotation, "__args__", ())
        item_type = _python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": item_type}
    if origin in (dict, Dict):
        return {"type": "object"}

    return {"type": "string"}


def discover_tools() -> tuple[Dict[str, Callable[..., Any]], List[Dict[str, Any]]]:
    registry: Dict[str, Callable[..., Any]] = {}
    schemas: List[Dict[str, Any]] = []

    for name, obj in inspect.getmembers(tools):
        if name.startswith("_"):
            continue

        # Підтримка smolagents @tool об'єктів
        if hasattr(obj, "forward") and hasattr(obj, "name") and hasattr(obj, "inputs"):
            tool_name = getattr(obj, "name", name)
            description = getattr(obj, "description", f"Tool {tool_name}")
            inputs = getattr(obj, "inputs", {}) or {}

            properties: Dict[str, Any] = {}
            required: List[str] = []

            for param_name, meta in inputs.items():
                if param_name == "self":
                    continue
                param_type = meta.get("type", "string") if isinstance(meta, dict) else "string"
                if param_type not in ("string", "integer", "number", "boolean", "array", "object"):
                    param_type = "string"
                prop: Dict[str, Any] = {
                    "type": param_type,
                    "description": meta.get("description", "") if isinstance(meta, dict) else "",
                }
                if param_type == "array":
                    prop["items"] = {"type": "string"}
                properties[param_name] = prop
                if isinstance(meta, dict) and not meta.get("nullable", False):
                    required.append(param_name)

            registry[tool_name] = obj.forward
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": str(description)[:500],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )
            continue

        # Підтримка звичайних Python-функцій із tools.py
        if inspect.isfunction(obj) and obj.__module__ == tools.__name__:
            sig = inspect.signature(obj)
            doc = inspect.getdoc(obj) or f"Execute {name}"

            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                schema = _python_type_to_json_schema(param.annotation)
                schema["description"] = f"Parameter {param_name}"
                properties[param_name] = schema
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            registry[name] = obj
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": doc[:500],
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
            )

    return registry, schemas


TOOL_REGISTRY, TOOL_SCHEMAS = discover_tools()


def execute_tool(tool_name: str, args: Dict[str, Any]) -> str:
    func = TOOL_REGISTRY.get(tool_name)
    if not func:
        return f"Tool error: Unknown tool '{tool_name}'. Available tools: {list(TOOL_REGISTRY.keys())}"

    try:
        output = func(**args)
        return str(output)
    except TypeError as e:
        return f"Tool argument error in '{tool_name}': {e}"
    except Exception as e:
        return f"Tool execution error in '{tool_name}': {type(e).__name__}: {e}"


# ============================================================
# CONTEXT & TEXT COMPACTION
# ============================================================

def compact_text(text: Any, max_chars: int) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n\n[TOOL OUTPUT TRUNCATED]"


def clean_final_answer(text: Any) -> str:
    s = str(text or "").strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"^(final\s*answer|answer)\s*:\s*", "", s, flags=re.IGNORECASE).strip()

    m = re.match(r"^`?\s*[^\w\s]\s*`?\s*\(([a-zA-Z0-9 _-]+)\)$", s)
    if m:
        s = m.group(1).strip()

    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    return s



def estimate_chars(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += len(str(msg.get("content") or ""))
        if msg.get("tool_calls"):
            total += len(json.dumps(msg["tool_calls"], ensure_ascii=False))
    return total


def compact_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Зберігає system + user prompt та останні MAX_HISTORY_BLOCKS блоків (assistant + tool).
    """
    if len(messages) <= 2:
        return list(messages)

    head = messages[:2]
    tail = messages[2:]

    # Групуємо хвіст по блоках (assistant + його tool відповіді)
    blocks: List[List[Dict[str, Any]]] = []
    current_block: List[Dict[str, Any]] = []

    for msg in tail:
        if msg.get("role") == "assistant" and current_block:
            blocks.append(current_block)
            current_block = [msg]
        else:
            current_block.append(msg)

    if current_block:
        blocks.append(current_block)

    kept_blocks = blocks[-MAX_HISTORY_BLOCKS:]
    compacted = list(head)
    for blk in kept_blocks:
        compacted.extend(blk)

    # Додатково стискаємо, якщо все ще перевищує SAFE_CONTEXT_CHARS
    while len(kept_blocks) > 1 and estimate_chars(compacted) > SAFE_CONTEXT_CHARS:
        kept_blocks.pop(0)
        compacted = list(head)
        for blk in kept_blocks:
            compacted.extend(blk)

    return compacted


# ============================================================
# GROQ COMPLETION WITH RATE LIMIT & PARSE HANDLING
# ============================================================

def create_completion(
    messages: List[Dict[str, Any]],
    tool_schemas: List[Dict[str, Any]] | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
):
    working_messages = compact_messages(messages)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": MODEL,
                "messages": working_messages,
                "temperature": 0.1,
                "max_tokens": max_tokens,
            }
            if tool_schemas:
                kwargs["tools"] = tool_schemas
                kwargs["tool_choice"] = "auto"

            return client.chat.completions.create(**kwargs)

        except Exception as e:
            error_text = str(e).lower()

            # 1. Денний ліміт TPD — не чекаємо, одразу прокидаємо в evaluate_gaia.py для зупинки
            if "tokens per day" in error_text or "tpd" in error_text or "daily token limit" in error_text:
                raise

            # 2. Хвилинний ліміт TPM або завеликий запит
            if (
                "tokens per minute" in error_text
                or "tpm" in error_text
                or "request too large" in error_text
                or "413" in error_text
            ):
                print()
                print("=" * 60)
                print(f"GROQ TPM / SIZE LIMIT (attempt {attempt}/{MAX_RETRIES})")
                print("=" * 60)
                print(e)
                if attempt < MAX_RETRIES:
                    print("Waiting 12s for Groq TPM limit reset...")
                    time.sleep(12)
                    # Стискаємо історію ще сильніше перед повтором
                    if len(working_messages) > 2:
                        working_messages = working_messages[:2] + working_messages[-2:]
                    continue
                raise

            # 3. Загальний Rate Limit 429
            if "429" in error_text or "rate limit" in error_text:
                wait_sec = 8 * attempt
                print(f"Rate limit 429 (attempt {attempt}/{MAX_RETRIES}). Sleeping {wait_sec}s...")
                if attempt < MAX_RETRIES:
                    time.sleep(wait_sec)
                    continue
                raise

            # 4. Помилка парсингу tool call від моделі (400 parsing failed)
            if (
                "parsing failed" in error_text
                or "output_parse_failed" in error_text
                or "tool_use_failed" in error_text
                or "tool call validation failed" in error_text
            ):
                print(f"Tool call validation/parsing failed (attempt {attempt}/{MAX_RETRIES}). Retrying...")
                if attempt < MAX_RETRIES:
                    working_messages = list(working_messages) + [
                        {
                            "role": "user",
                            "content": "Invalid tool name or format. Call ONLY an exact tool name from the provided tools list, or provide the final answer directly.",
                        }
                    ]
                    continue
                raise

            raise


def force_final_answer(messages: List[Dict[str, Any]]) -> str:
    print()
    print("=" * 60)
    print("FORCING FINAL ANSWER")
    print("=" * 60)

    final_messages = compact_messages(messages) + [
        {
            "role": "user",
            "content": (
                "Stop using tools now. Based ONLY on the information already gathered above, "
                "provide the concise, exact final answer to the original question. "
                "Output only the answer without explanation."
            ),
        }
    ]

    print(f"CONTEXT SIZE: {estimate_chars(final_messages)} chars")
    time.sleep(2)

    try:
        response = create_completion(
            messages=final_messages,
            tool_schemas=None,
            max_tokens=MAX_FINAL_TOKENS,
        )
        content = response.choices[0].message.content or ""
        answer = clean_final_answer(content)

        print()
        print("=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(answer)

        return answer if answer else "Unable to produce a final answer within the available execution limits."

    except Exception as e:
        error_text = str(e).lower()
        if "tokens per day" in error_text or "tpd" in error_text:
            return f"Agent API error: {type(e).__name__}: {e}"
        print(f"Force final answer error: {type(e).__name__}: {e}")
        return "Unable to produce a final answer within the available execution limits."


# ============================================================
# MAIN AGENT LOOP
# ============================================================

def run_agent(prompt: str) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(prompt)},
    ]

    seen_calls = set()
    search_calls = 0

    for step in range(1, MAX_STEPS + 1):
        print()
        print("#" * 60)
        print(f"AGENT STEP {step}/{MAX_STEPS}")
        print("#" * 60)

        compacted = compact_messages(messages)
        ctx_size = estimate_chars(compacted)
        print()
        print(f"CONTEXT SIZE: {ctx_size} chars")
        print()
        print(f"REQUEST SIZE: {ctx_size} chars")

        try:
            response = create_completion(
                messages=compacted,
                tool_schemas=TOOL_SCHEMAS,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as e:
            return f"Agent API error: {type(e).__name__}: {e}"

        message = response.choices[0].message

        # Якщо модель дала фінальну відповідь без виклику інструментів
        if not message.tool_calls:
            answer = clean_final_answer(message.content or "")
            if answer:
                print()
                print("=" * 60)
                print("FINAL ANSWER")
                print("=" * 60)
                print(answer)
                return answer
            return force_final_answer(messages)

        # Записуємо assistant message із tool_calls в історію
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in message.tool_calls
            ],
        }
        messages.append(assistant_msg)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"

            print()
            print("RAW TOOL ARGUMENTS:")
            print(raw_args)

            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    args = {}
            except Exception as e:
                result = f"Invalid JSON tool arguments: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result,
                    }
                )
                continue

            # Перевірка на дублікат виклику з однаковими аргументами
            call_signature = (
                tool_name,
                json.dumps(args, sort_keys=True, ensure_ascii=False),
            )

            if call_signature in seen_calls:
                print()
                print("=" * 60)
                print(f"DUPLICATE TOOL CALL BLOCKED: {tool_name}")
                print("=" * 60)
                if messages and messages[-1].get("role") == "assistant":
                    messages.pop()
                return force_final_answer(messages)

            seen_calls.add(call_signature)

            # Ліміт пошуку
            if tool_name == "search":
                search_calls += 1
                if search_calls > SEARCH_MAX_CALLS:
                    print()
                    print("=" * 60)
                    print(f"SEARCH LIMIT REACHED ({SEARCH_MAX_CALLS} calls)")
                    print("=" * 60)
                    if messages and messages[-1].get("role") == "assistant":
                        messages.pop()
                    return force_final_answer(messages)

            print()
            print("=" * 60)
            print(f"EXECUTING TOOL: {tool_name}")
            print(f"ARGUMENTS: {args}")
            print("=" * 60)

            result = execute_tool(tool_name, args)

            # Повертаємо спробу пошуку, якщо DuckDuckGo повернув порожньо або помилку
            if tool_name == "search" and (
                "No results found" in str(result)
                or "Search error:" in str(result)
            ):
                search_calls = max(0, search_calls - 1)
                print(f"Search returned empty/error — refunded search attempt (used: {search_calls}/{SEARCH_MAX_CALLS})")

            result = compact_text(result, MAX_TOOL_RESULT_CHARS)

            print("RESULT:")
            print(result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result,
                }
            )

    print()
    print("=" * 60)
    print("MAX STEPS REACHED")
    print("=" * 60)
    return force_final_answer(messages)