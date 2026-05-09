import requests
from bs4 import BeautifulSoup
from typing import List, Dict


def search_web(query: str, limit: int = 5) -> List[Dict]:
    """Perform a lightweight web search using DuckDuckGo HTML and fetch top pages.

    Returns a list of dicts: {title, url, snippet, text}
    """
    results = []
    try:
        r = requests.get("https://html.duckduckgo.com/html/", params={"q": query}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("http") and len(links) < limit:
                links.append((a.get_text(strip=True) or href, href))

        for title, url in links[:limit]:
            snippet = ""
            text = ""
            try:
                r2 = requests.get(url, timeout=10)
                r2.raise_for_status()
                soup2 = BeautifulSoup(r2.text, "html.parser")
                p = soup2.find_all("p")
                if p:
                    snippet = " ".join([x.get_text(strip=True) for x in p[:2]])[:400]
                    text = "\n\n".join([x.get_text(strip=True) for x in p[:10]])
            except Exception:
                snippet = "(could not fetch content)"

            results.append({"title": title, "url": url, "snippet": snippet, "text": text})
    except Exception:
        pass

    return results
