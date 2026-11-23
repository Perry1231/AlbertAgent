from smolagents import tool

@tool
def transcribe_audio_tool(audio_path: str) -> str:
    """Transcribes an audio file to text.

    Args:
        audio_path: Path to the audio file.

    Returns:
        str: Transcribed text or error message.
    """
    import os

    if not os.path.exists(audio_path):
        return f"Error: Audio file '{audio_path}' not found."

    try:
        import whisper  # Імпорт ТІЛЬКИ у блоці try всередині функції!
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except ImportError:
        return "Error: whisper library is not installed in the environment."
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"