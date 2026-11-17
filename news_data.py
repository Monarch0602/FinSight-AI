
import yfinance as yf


def fetch_financial_news(symbol):

    try:
        news_items = yf.Ticker(symbol).news

        headlines = []
        seen = set()

        for item in news_items:

            content = item.get("content", {})

            title = (
                content.get("title")
                or item.get("title")
            )

            if title and title not in seen:
                headlines.append(title)
                seen.add(title)

        return headlines[:15]

    except Exception as error:
        print("News fetching error:", error)
        return []
