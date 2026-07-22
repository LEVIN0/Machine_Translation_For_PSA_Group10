import requests
from bs4 import BeautifulSoup


class PSAScraper:

    def __init__(self):

        self.headers = {
            "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

    def get_page(self, url):

        response = requests.get(
            url,
            headers=self.headers,
            timeout=30
        )

        response.raise_for_status()

        return BeautifulSoup(
            response.text,
            "html.parser"
        )

    def get_paragraphs(self, url):

        soup = self.get_page(url)

        paragraphs = []

        for p in soup.find_all("p"):

            text = p.get_text(" ", strip=True)

            if len(text) > 40:

                paragraphs.append(text)

        return paragraphs