from smolagents import tool

@tool
def extract_video_frames_tool(video_path: str) -> str:
    """Extracts frame information from a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        str: Video details or error message.
    """
    import os

    if not os.path.exists(video_path):
        return f"Error: Video file '{video_path}' not found."

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return f"Video frame count: {frame_count}, FPS: {fps}"
    except Exception as e:
        return f"Error processing video: {str(e)}"