from smolagents import tool

@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage and returns its text content.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        str: Extracted text content from the webpage.
    """
    import requests
    from bs4 import BeautifulSoup

    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:4000]
    except Exception as e:
        return f"Error visiting webpage: {str(e)}"

@tool
def smart_web_scraper(url: str, prompt: str) -> str:
    """Scrapes structured information from a webpage using a prompt.

    Args:
        url: The URL to scrape.
        prompt: Description of the data to extract.

    Returns:
        str: Extracted content or fallback search result.
    """
    import requests
    from bs4 import BeautifulSoup

    # Якщо scrapegraphai не встановлено або не зконфігуровано — використовуємо надійний fallback
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text[:4000]
    except Exception as e:
        return f"Error scraping web page: {str(e)}"