from smolagents import tool
import cv2
import os

@tool
def extract_video_frames_tool(video_path: str, fps_interval: int = 1) -> str:
    """Витягує кадри з відеофайлу (MP4, AVI, MOV) із заданим інтервалом у секундах.

    Args:
        video_path: Шлях до відеофайлу.
        fps_interval: Інтервал збереження кадрів у секундах (за замовчуванням кожну 1 секунду).
    """
    if not os.path.exists(video_path):
        return f"Помилка: Файл {video_path} не знайдено."
        
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        frame_count = 0
        saved_count = 0
        output_dir = "extracted_frames"
        os.makedirs(output_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Зберігаємо кадр кожні N секунд
            if frame_count % (fps * fps_interval) == 0:
                frame_filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                
            frame_count += 1

        cap.release()
        return f"Успішно збережено {saved_count} кадрів у папку '{output_dir}'."
    except Exception as e:
        return f"Помилка при витягуванні кадрів: {str(e)}"