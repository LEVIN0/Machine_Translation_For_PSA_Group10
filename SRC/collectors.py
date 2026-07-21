from SRC.scraper import fetch_page

def collect_redcross():
    """
    Collect announcement titles from Kenya Red Cross.
    """

    url = "https://www.redcross.or.ke"

    soup = fetch_page(url)

    articles = []

    headings = soup.find_all(["h2", "h3"])

    for heading in headings:

        title = heading.get_text(strip=True)

        if len(title) < 10:
            continue

        articles.append({
            "Title": title
        })

    return articles