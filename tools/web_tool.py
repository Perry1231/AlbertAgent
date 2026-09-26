from smolagents import tool

import io
import requests
from bs4 import BeautifulSoup


MAX_WEB_TEXT = 1000
MAX_SEARCH_RESULTS = 5


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}


# ==========================================
# WEB SEARCH
# ==========================================

@tool
def search(query: str, top_n: int = 5) -> str:
    """
    Searches the web and returns useful search results.

    Args:
        query: The search query.
        top_n: Maximum number of search results to return.

    Returns:
        Search results with titles, URLs, and snippets.
    """

    try:
        from ddgs import DDGS

        top_n = max(1, min(top_n, MAX_SEARCH_RESULTS))

        results = DDGS().text(
            query,
            max_results=top_n,
        )

        if not results:
            return "No search results found."

        output = []

        for i, result in enumerate(results, 1):

            title = result.get("title") or ""
            url = result.get("href") or ""
            body = result.get("body") or ""

            if not title and not url and not body:
                continue

            output.append(
                f"{i}. {title}\n"
                f"URL: {url}\n"
                f"Snippet: {body}"
            )

        if not output:
            return "Search returned empty results."

        return "\n\n".join(output)

    except Exception as e:
        return f"Search error: {type(e).__name__}: {e}"


# ==========================================
# PDF EXTRACTION
# ==========================================

def extract_pdf_text(content: bytes) -> str:
    """
    Extract readable text from a PDF.
    """

    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))

        pages = []

        # Do not process enormous PDFs.
        for page in reader.pages[:10]:

            try:
                text = page.extract_text()

                if text:
                    pages.append(text)

            except Exception:
                continue

        if not pages:
            return "PDF was opened, but no readable text could be extracted."

        text = "\n\n".join(pages)

        return text[:MAX_WEB_TEXT]

    except ImportError:
        return (
            "PDF detected, but pypdf is not installed. "
            "Install it with: pip install pypdf"
        )

    except Exception as e:
        return (
            f"PDF extraction error: "
            f"{type(e).__name__}: {e}"
        )


# ==========================================
# WEBPAGE
# ==========================================

@tool
def visit_webpage(url: str) -> str:
    
    """
    Visits a webpage or PDF and returns readable text content.

    Args:
        url: URL to visit.

    Returns:
        Extracted webpage or PDF text.
    """

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15,
        )

        response.raise_for_status()

        content_type = (
            response.headers
            .get("Content-Type", "")
            .lower()
        )

        # ======================================
        # PDF
        # ======================================

        if (
            "application/pdf" in content_type
            or url.lower().split("?")[0].endswith(".pdf")
        ):

            text = extract_pdf_text(
                response.content
            )

            return (
                f"PDF DOCUMENT\n"
                f"URL: {url}\n\n"
                f"{text}"
            )

        # ======================================
        # HTML
        # ======================================

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

        return text[:MAX_WEB_TEXT]

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
    Scrapes a webpage or PDF and returns relevant readable content.

    Args:
        url: URL to scrape.
        prompt: Description of the data to extract.

    Returns:
        Extracted content.
    """

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15,
        )

        response.raise_for_status()

        content_type = (
            response.headers
            .get("Content-Type", "")
            .lower()
        )

        return limit_output(
            "SCRAPED WEBPAGE:\n\n"
            f"{text}\n\n"
            "EXTRACTION REQUEST:\n"
            f"{prompt}"
)
        # ======================================
        # PDF
        # ======================================

        if (
            "application/pdf" in content_type
            or url.lower().split("?")[0].endswith(".pdf")
        ):

            text = extract_pdf_text(
                response.content
            )

            return (
                "SCRAPED PDF:\n\n"
                f"{text}\n\n"
                "EXTRACTION REQUEST:\n"
                f"{prompt}"
            )

        # ======================================
        # HTML
        # ======================================

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
            f"{text[:MAX_WEB_TEXT]}\n\n"
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
        page_content: Webpage content.

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