import os
from typing import Optional
from PIL import Image
import pytesseract
from smolagents import tool

@tool
def process_image_tool(image_path: str, action: str, crop_box: Optional[list[int]] = None) -> str:
    """Processes an image: performs OCR (reads text) or crops it.

    Args:
        image_path: The path to the image file (e.g., 'image.png').
        action: The action to perform: 'ocr' to read text or 'crop' to trim the image.
        crop_box: A list of 4 integers [left, upper, right, lower] for cropping. Required only if action='crop'.
    """
    if not os.path.exists(image_path):
        return f"Error: File {image_path} not found."
        
    try:
        img = Image.open(image_path)
        
        if action == "ocr":
            text = pytesseract.image_to_string(img)
            return text if text.strip() else "Text not found."
            
        elif action == "crop":
            if not crop_box or len(crop_box) != 4:
                return "Error: Provide 4 coordinates [left, upper, right, lower] for cropping."
            
            cropped_img = img.crop(tuple(crop_box))
            output_path = f"cropped_{os.path.basename(image_path)}"
            cropped_img.save(output_path)
            return f"Image successfully cropped and saved as '{output_path}'."
            
        return "Unknown action. Use 'ocr' or 'crop'."
    except Exception as e:
        return f"Error occurred while processing image: {str(e)}"