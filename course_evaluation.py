import json
import requests

from custom_agent import run_agent


API_URL = "https://agents-course-unit4-scoring.hf.space"


def get_questions():
    response = requests.get(
        f"{API_URL}/questions",
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def main():

    print("=" * 60)
    print("HUGGING FACE AGENTS COURSE EVALUATION")
    print("=" * 60)

    print()
    print("Loading course questions...")

    questions = get_questions()

    print(f"Questions loaded: {len(questions)}")

    print()
    print("=" * 60)
    print("FIRST QUESTION")
    print("=" * 60)

    print(json.dumps(
        questions[0],
        indent=2,
        ensure_ascii=False,
    ))

    task_id = questions[0]["task_id"]
    question = questions[0]["question"]

    print()
    print("=" * 60)
    print("RUNNING ALBERTAGENT")
    print("=" * 60)

    print()
    print("Question:")
    print(question)

    try:
        answer = run_agent(question)

    except Exception as e:
        answer = f"ERROR: {type(e).__name__}: {e}"

    print()
    print("=" * 60)
    print("AGENT ANSWER")
    print("=" * 60)

    print(answer)

    result = {
        "task_id": task_id,
        "submitted_answer": str(answer).strip(),
    }

    with open(
        "course_test_result.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 60)
    print("SAVED")
    print("=" * 60)

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()