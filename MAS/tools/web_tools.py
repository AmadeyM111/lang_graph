# web-tools for Researcher

@tool 
async def web_search(query: str) -> List[str]:
    """Простейший web-поиск: возвращает список ссылок."""
    try:
        url = "https://duckduckgo.com/html/"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, data={"q": query})
            # Упрощенный парсинг ссылок
            hrefs = []
            for line in r.text.splitlines():
                if 'result__a' in line and 'href=' in line:
                    start = line.find('href="') + 6
                    end = line.find('"', start)
                    link = line[start:end]
                    if link.startswith("http"):
                        hrefs.append(link)
                if len(hrefs) >= 5:
                    break
        return hrefs or ["https://example.com"]
    except Exception as e:
        return [f"ERROR:{e}"]

@tool
async def fetch_url(url: str) -> str:
    """Скачивает HTML/текст по URL."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            return r.text[:50_000] # ограничение объема
    except Exception as e:
        return f"ERROR:{e}"