import os
import re
import base64
from openai import OpenAI
from smolagents import tool
from youtube_transcript_api import YouTubeTranscriptApi


def _get_groq_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe an attached audio file (.mp3, .wav, .m4a, .ogg, .flac) into text using Groq Whisper.

    Args:
        file_path: Local path to the audio file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    try:
        client = _get_groq_client()
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                response_format="text",
            )
        return str(transcript).strip()
    except Exception as e:
        return f"Audio transcription error: {e}"


@tool
def analyze_image(file_path: str, prompt: str) -> str:
    """Analyze an attached image (.png, .jpg, .jpeg, .webp) and answer a specific visual question about it.

    Args:
        file_path: Local path to the image file.
        prompt: Detailed question about what to inspect or read in the image.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    try:
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else (ext or "png")
        with open(file_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode("utf-8")

        client = _get_groq_client()
        resp = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=800,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Image analysis error: {e}"


@tool
def get_youtube_transcript(url: str) -> str:
    """Fetch the spoken transcript / subtitles of a YouTube video by its URL or video ID.

    Args:
        url: Full YouTube video URL (e.g. https://www.youtube.com/watch?v=...) or 11-character video ID.
    """
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", url)
    video_id = match.group(1) if match else url.strip()
    try:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        text = " ".join(snippet.text for snippet in fetched)
        return text[:6000] if text else "Transcript is empty."
    except Exception as e:
        return f"YouTube transcript error: {e}"