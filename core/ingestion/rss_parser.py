import feedparser
from datetime import datetime
from time import mktime
from typing import List
from core.domain.article import Article

class RSSParser:
    """Adaptador de Ingestão: Faz o fetch de feeds RSS e instancia Articles imutáveis."""
    def __init__(self, feeds: List[str]):
        self.feeds = feeds

    def fetch_all(self) -> List[Article]:
        articles = []
        for feed_url in self.feeds:
            parsed = feedparser.parse(feed_url)
            for entry in parsed.entries:
                dt = datetime.utcnow()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    dt = datetime.fromtimestamp(mktime(entry.published_parsed))
                
                content = ""
                if hasattr(entry, 'content'):
                    content = " ".join([c.value for c in entry.content])
                elif hasattr(entry, 'summary'):
                    content = entry.summary
                
                article = Article(
                    url=getattr(entry, 'link', ''),
                    title=getattr(entry, 'title', ''),
                    content=content,
                    published_at=dt,
                    source=feed_url
                )
                articles.append(article)
        return articles
