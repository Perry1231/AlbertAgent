from smolagents import tool

@tool
def read_json_file(file_path: str) -> str:
    """Reads and parses a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        str: String representation of the JSON content or error message.
    """
    import json
    import os

    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return f"Error reading JSON file: {str(e)}"

@tool
def save_text_to_file(file_path: str, content: str) -> str:
    """Saves text content to a file.

    Args:
        file_path: Target file path.
        content: Text content to save.

    Returns:
        str: Status message.
    """
    import os

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully saved to '{file_path}'."
    except Exception as e:
        return f"Error saving file: {str(e)}"