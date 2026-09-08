from smolagents import tool
import whisper
import os

# Завантажуємо легку модель (tiny/base/small/medium/large)
whisper_model = whisper.load_model("base")

@tool
def transcribe_audio_tool(audio_path: str) -> str:
    """Транскрибує аудіофайл (MP3, WAV, M4A) у текст за допомогою моделі Whisper.

    Args:
        audio_path: Шлях до аудіофайлу на диску.
    """
    if not os.path.exists(audio_path):
        return f"Помилка: Файл {audio_path} не знайдено."
        
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Помилка при транскрибації: {str(e)}"