import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin


class ParserService:

    async def fetch(self, session, url):
        """Асинхронно загружает страницу."""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
        except aiohttp.ClientError as e:
            print(f"Ошибка подключения: {e} для {url}")
        return None

    async def validate_url(self, session, url):
        """Проверяет, что страница не имеет ошибки 404."""
        html = await self.fetch(session, url)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.find('title')
            if title and "404" in title.get_text():
                return None
            return html
        return None
    
    async def get_links_from_page(self, session, url, base_url):
        """Извлекает ссылки с одной страницы."""
        html = await self.validate_url(session, url)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            links = set()
            for a_tag in soup.find_all('a', href=True):
                absolute_url = urljoin(base_url, a_tag['href'])
                if absolute_url.startswith(base_url):
                    links.add(absolute_url)
            return links
        return set()

    async def parse_article_content(self, session, url):
        """Парсит данные из <article>."""
        html = await self.validate_url(session, url)
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            try:
                article = soup.find('article')
                if article:
                    title = article.find('h1').get_text(strip=True) if article.find('h1') else None
                    content = "\n".join(
                        p.get_text(strip=True) for p in article.find_all('p')
                    )
                    return {"title": title, "content": content, "url": url}
            except AttributeError:
                print(f"Ошибка парсинга {url}")
        return None
    
    async def crawl(self, base_url):
        """Собирает все ссылки с базового URL."""
        visited = set()
        to_visit = {base_url}
        all_links = set()

        async with aiohttp.ClientSession() as session:
            while to_visit:
                tasks = []
                for url in to_visit:
                    if url not in visited:
                        tasks.append(self.get_links_from_page(session, url, base_url))
                to_visit = set()
                if tasks:
                    results = await asyncio.gather(*tasks)
                    for links in results:
                        print(links)
                        to_visit.update(links - visited)
                        all_links.update(links)
                visited.update(to_visit)

        return all_links

    async def parse_all_links(self, links):
        """Парсит содержимое всех собранных ссылок."""
        async with aiohttp.ClientSession() as session:
            parse_tasks = [self.parse_article_content(session, url) for url in links]
            parsed_results = await asyncio.gather(*parse_tasks)
            return [result for result in parsed_results if result]
    
    async def fetch_and_parse_data(self, base_url):
        """Получает все статьи и парсит с них артикли"""
        links = await self.crawl(base_url)
        parsed_data = await self.parse_all_links(links)
        return parsed_data