from smolagents import tool
import json

@tool
def read_json_file(file_path: str) -> dict:
    """Reads the content of a JSON file and returns it as a dictionary.

    Args:
        file_path: Path to the JSON file on the disk.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

@tool
def save_text_to_file(content: str, filename: str) -> str:
    """Saves text content to a file.

    Args:
        content: Text or code to save.
        filename: Name of the file (e.g., 'result.txt' or 'script.py').
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"File {filename} saved successfully."