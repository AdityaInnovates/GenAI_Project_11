import requests
from bs4 import BeautifulSoup

def extract_text(url: str) -> str:
    """
    Fetch and extract article text from a given URL.
    
    Args:
        url (str): The URL of the news article.
        
    Returns:
        str: The extracted text of the article.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text() for p in paragraphs)
    
    return text.strip()
