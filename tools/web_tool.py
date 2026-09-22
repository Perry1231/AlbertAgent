from smolagents import tool

import requests
from bs4 import BeautifulSoup


# ==========================================
# WEB SEARCH
# ==========================================

@tool
def search(
    query: str,
    top_n: int = 5,
) -> str:
    """
    Searches the web and returns relevant search results.

    Args:
        query: Search query.
        top_n: Maximum number of search results.

    Returns:
        A text summary of search results.
    """

    try:

        from ddgs import DDGS

        top_n = max(1, min(top_n, 10))

        results = DDGS().text(
            query,
            max_results=top_n,
        )

        if not results:
            return "No search results found."

        output = []

        for i, result in enumerate(results, 1):

            title = result.get("title", "")
            url = result.get("href", "")
            body = result.get("body", "")

            output.append(
                f"{i}. {title}\n"
                f"URL: {url}\n"
                f"Snippet: {body}"
            )

        return "\n\n".join(output)

    except Exception as e:

        return (
            f"Search error: "
            f"{type(e).__name__}: {e}"
        )


# ==========================================
# VISIT WEBPAGE
# ==========================================

@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage and returns its text content.

    Args:
        url: URL of the webpage to visit.

    Returns:
        Extracted text from the webpage.
    """

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for element in soup(
            ["script", "style", "noscript"]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        if not text:
            return "No text content found."

        return text[:12000]

    except Exception as e:

        return (
            f"Webpage error: "
            f"{type(e).__name__}: {e}"
        )


# ==========================================
# SMART WEB SCRAPER
# ==========================================

@tool
def smart_web_scraper(
    url: str,
    prompt: str,
) -> str:
    """
    Scrapes structured information from a webpage.

    Args:
        url: URL to scrape.
        prompt: Description of the data to extract.

    Returns:
        Webpage text that can be analyzed according to the prompt.
    """

    try:

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for element in soup(
            ["script", "style", "noscript"]
        ):
            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        if not text:
            return "No content found."

        return (
            "SCRAPED WEBPAGE:\n\n"
            f"{text[:12000]}\n\n"
            "EXTRACTION REQUEST:\n"
            f"{prompt}"
        )

    except Exception as e:

        return (
            f"Scraper error: "
            f"{type(e).__name__}: {e}"
        )


# ==========================================
# FIND IN PAGE
# ==========================================

@tool
def find_in_page(
    pattern: str,
    page_content: str,
) -> str:
    """
    Finds occurrences of a pattern in webpage content.

    Args:
        pattern: Text to search for.
        page_content: Webpage text.

    Returns:
        Relevant lines containing the pattern.
    """

    try:

        if not page_content:
            return "Page content is empty."

        pattern_lower = pattern.lower()

        lines = page_content.splitlines()

        matches = []

        for index, line in enumerate(lines):

            if pattern_lower in line.lower():

                start = max(0, index - 2)
                end = min(
                    len(lines),
                    index + 3,
                )

                context = lines[start:end]

                matches.append(
                    "\n".join(context)
                )

        if not matches:

            return (
                f"No occurrences found for: "
                f"{pattern}"
            )

        return (
            f"Found {len(matches)} occurrence(s) "
            f"for '{pattern}':\n\n"
            + "\n\n---\n\n".join(matches[:10])
        )

    except Exception as e:

        return (
            f"Find error: "
            f"{type(e).__name__}: {e}"
        )