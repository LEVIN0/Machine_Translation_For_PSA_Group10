import requests
from bs4 import BeautifulSoup


class PSACollector:

    def collect(self, url):

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=20
            )

            response.raise_for_status()

        except Exception as e:

            print(f"Failed: {url}")
            print(e)

            return []

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        paragraphs = []

        for p in soup.find_all("p"):

            text = p.get_text(" ", strip=True)

            if len(text.split()) >= 6:

                paragraphs.append(text)

        return paragraphs