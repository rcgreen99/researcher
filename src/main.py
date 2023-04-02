import sys
from src.searcher import Searcher
from src.summarizer import Summarizer

if __name__ == "__main__":
    query = sys.argv[1]

    searcher = Searcher()
    search_result = searcher.search(query)

    summarizer = Summarizer()
    summary = summarizer.summarize(search_result)

    print(f"Summary:\n{summary}")
