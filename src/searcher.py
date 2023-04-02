import requests
from googlesearch import search
from bs4 import BeautifulSoup


class Searcher:
    def __init__(self):
        pass

    def search(self, query):
        url = self.find_url(query)
        print(f"URL: {url}")
        page = requests.get(url)
        text = self.parse_page(page.text)
        return text

    def find_url(self, query):
        results = []
        for result in search(term=query, num_results=10, lang="en"):
            results.append(result)
        return results[0]

    def parse_page(self, html_text):
        soup = BeautifulSoup(html_text, "html.parser")
        p_tags = soup.find_all("p")
        text = ""
        for tag in p_tags:
            text += tag.get_text() + "\n"
        return text


if __name__ == "__main__":
    query = "the battle of the buldge"

    searcher = Searcher()

    results = searcher.get_text_from_query(query)

    print(results)
