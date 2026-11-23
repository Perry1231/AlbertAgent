from smolagents import tool

@tool
def process_image_tool(image_path: str) -> str:
    """Processes an image file and returns its size and format details.

    Args:
        image_path: Path to the image file.

    Returns:
        str: Image metadata or error message.
    """
    import os
    from PIL import Image

    if not os.path.exists(image_path):
        return f"Error: Image file '{image_path}' not found."

    try:
        with Image.open(image_path) as img:
            return f"Image format: {img.format}, size: {img.size}, mode: {img.mode}"
    except Exception as e:
        return f"Error processing image: {str(e)}"