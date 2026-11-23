import os
import tempfile
import gradio as gr
import requests
import pandas as pd
from custom_agent import run_agent

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    if profile:
        username = f"{profile.username}"
    else:
        return "Будь ласка, спочатку натисни 'Sign in with Hugging Face' зверху!", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
    except Exception as e:
        return f"Помилка отримання питань: {e}", None

    results_log = []
    answers_payload = []

    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question", "")
        file_name = item.get("file_name")

        # Якщо у задачі є прикріплений файл — завантажуємо його локально для агента
        if file_name:
            try:
                file_url = f"{api_url}/files/{task_id}"
                f_resp = requests.get(file_url, timeout=20)
                f_resp.raise_for_status()
                tmp_dir = tempfile.mkdtemp()
                local_path = os.path.join(tmp_dir, file_name)
                with open(local_path, "wb") as f:
                    f.write(f_resp.content)
                question_text = f"{question_text}\n\n[ATTACHED LOCAL FILE PATH: {local_path}]"
            except Exception as fe:
                print(f"File download warning for {task_id}: {fe}")

        try:
            submitted_answer = str(run_agent(question_text)).strip()
        except Exception as e:
            submitted_answer = f"AGENT ERROR: {e}"

        answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
        results_log.append({
            "Task ID": task_id,
            "Question": item.get("question", "")[:120] + "...",
            "Submitted Answer": submitted_answer,
        })

    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }

    try:
        resp = requests.post(submit_url, json=submission_data, timeout=60)
        resp.raise_for_status()
        result_data = resp.json()
        status_message = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return status_message, pd.DataFrame(results_log)
    except Exception as e:
        return f"Помилка відправки результатів: {e}", pd.DataFrame(results_log)


with gr.Blocks() as demo:
    gr.Markdown("# AlbertAgent — Final Evaluation Runner")
    gr.Markdown("1. Увійди через свій акаунт Hugging Face.\n2. Натисни кнопку нижче, щоб пройти 20 задач курсу та отримати сертифікат.")
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=6, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=False)