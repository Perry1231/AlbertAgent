import requests
from bs4 import BeautifulSoup
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """
    Visits a specified URL and extracts its main textual content while removing HTML tags.
    Useful for reading full articles, blog posts, or online documentation found via search.

    Args:
        url: The full web page URL (e.g., 'https://en.wikipedia.org/wiki/Artificial_intelligence').
    """
    try:
        # User-Agent header to mimic a standard browser request
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/119.0.0.0 Safari/537.36"
            )
        }
        
        # Fetch the webpage with a 10-second timeout
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML markup using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Decompose non-content scripts, styles, and navigational elements
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()

        # Extract plain text content
        text = soup.get_text(separator="\n")

        # Clean up whitespace and empty lines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        # Truncate output to prevent exceeding context window limits
        max_length = 5000
        if len(clean_text) > max_length:
            return clean_text[:max_length] + f"\n\n...[Content truncated to {max_length} characters]"

        return clean_text if clean_text else "The webpage is empty or contains no readable text."

    except requests.RequestException as e:
        return f"Failed to fetch content from '{url}': {str(e)}"
    


@tool
def smart_web_scraper(url: str, prompt: str) -> str:
    """Витягує структуровані дані з веб-сторінки за допомогою природної мови та LLM.

    Args:
        url: Повне посилання на веб-сторінку (наприклад, 'https://example.com/products').
        prompt: Опис того, які саме дані потрібно знайти та витягнути з цієї сторінки.
    """
    try:
        scraper = SmartScraperGraph(
            prompt=prompt,
            source=url,
            config=graph_config
        )
        result = scraper.run()
        return str(result)
    except Exception as e:
        return f"Помилка при скрейпінгу: {str(e)}"

    