import requests
from bs4 import BeautifulSoup
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """
    Відвідує вказаний URL та повертає текстовий вміст вебсторінки без HTML-тегів.
    Корисно для читання статей, документації або деталей новин за посиланням.

    Args:
        url: Повна адреса вебсторінки (наприклад, 'https://uk.wikipedia.org/wiki/Штучний_інтелект').
    """
    try:
        # Імітуємо звичайний браузер за допомогою User-Agent
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Очищаємо HTML від тегів
        soup = BeautifulSoup(response.text, "html.parser")

        # Видаляємо скрипти, стилі та навігаційні блоки
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Отримуємо чистий текст
        text = soup.get_text(separator="\n")

        # Видаляємо зайві порожні рядки
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        # Обмежуємо довжину тексту, щоб не перевищити ліміт токенів моделі
        max_length = 5000
        if len(clean_text) > max_length:
            return clean_text[:max_length] + f"\n\n...[Текст скорочено до {max_length} символів]"

        return clean_text if clean_text else "Сторінка порожня або не містить текстового вмісту."

    except requests.RequestException as e:
        return f"Помилка завантаження сторінки '{url}': {str(e)}"