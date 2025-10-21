import trafilatura, requests
from readability import Document
from bs4 import BeautifulSoup

def extract_text_from_url(url: str) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            txt = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
            if txt.strip():
                return txt
    except Exception:
        pass
    try:
        html = requests.get(url, timeout=10).text
        doc = Document(html)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        return soup.get_text(separator=" ").strip()
    except Exception:
        return ""
