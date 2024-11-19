import numpy as np
import faiss
from services.embedding_service import EmbeddingService
import asyncio


class KnowledgeService:
    def __init__(self, knowledge_base, embedding_service: EmbeddingService, dimension=1536):
        """
        Сервис для работы с базой знаний.
        :param knowledge_base: Список данных базы знаний с сайта (в формате [{'title': ..., 'content': ..., 'url': ...}]).
        :param embedding_service: Экземпляр EmbeddingService для работы с эмбеддингами.
        :param dimension: Размерность эмбеддингов.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.embedding_service = embedding_service
        self.knowledge_base = knowledge_base
        self.embeddings = None

    async def build_index(self):
        """
        Создание векторного индекса для базы знаний.
        """
        tasks = [
            (entry, self.embedding_service.get_embedding(entry['title']))
            for entry in self.knowledge_base if entry["title"]
        ]

        embeddings = await asyncio.gather(*[task[1] for task in tasks])

        for (entry, embedding) in zip(tasks, embeddings):
            if embedding is not None and len(embedding) == self.dimension:
                self.index.add(np.array([embedding]))
            else:
                print(f"Invalid embedding for entry: {entry[0]['title']}")

        self.embeddings = np.array([embedding for embedding in embeddings if embedding is not None])
        
    async def search(self, query, k=1, threshold=0.60):
        """
        Поиск в базе знаний на основе запроса.
        :param query: Запрос пользователя.
        :param k: Количество ближайших соседей для поиска.
        :param threshold: Пороговое расстояние для фильтрации.
        """
        query_embedding = await self.embedding_service.get_embedding(query)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for i in range(k):
            idx = indices[0][i]
            if distances[0][i] <= threshold and idx < len(self.knowledge_base):
                entry = self.knowledge_base[idx]
                results.append({
                    "title": entry['title'],
                    "content": entry['content'],
                    "url": entry['url'],
                    "distance": distances[0][i]
                })

        return results