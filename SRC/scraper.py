import requests
from bs4 import BeautifulSoup


def fetch_page(url):
    """
    Downloads a webpage and returns a BeautifulSoup object.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def inspect_page(url):
    """
    Inspect the structure of a webpage.
    """

    soup = fetch_page(url)

    print("=" * 60)
    print("PAGE TITLE")
    print("=" * 60)
    print(soup.title.text if soup.title else "No title")

    print()

    print("=" * 60)
    print("HEADINGS")
    print("=" * 60)

    for heading in soup.find_all(["h1", "h2", "h3"])[:10]:
        print(heading.get_text(strip=True))

    print()

    print("=" * 60)
    print("FIRST 20 LINKS")
    print("=" * 60)

    for link in soup.find_all("a", href=True)[:20]:
        print(link.get_text(strip=True), "->", link["href"])


    return soup