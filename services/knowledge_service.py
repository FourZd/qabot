import numpy as np
import faiss
from services.embedding_service import EmbeddingService
import asyncio
from utils.logger import logger

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
        Асинхронное создание векторного индекса для базы знаний.
        """
        embeddings = []
        valid_entries = []

        async def process_entry(entry):
            """
            Асинхронная обработка одной записи.
            """
            if entry["title"]:
                embedding = await self.embedding_service.get_embedding(entry["title"])
                if embedding is not None and len(embedding) == self.dimension:
                    return embedding, entry  # Возвращаем эмбеддинг и запись
            return None, None

        # Асинхронно обрабатываем все записи в knowledge_base
        results = await asyncio.gather(*(process_entry(entry) for entry in self.knowledge_base))

        # Разбираем результаты
        for embedding, entry in results:
            if embedding is not None:
                embeddings.append(embedding)
                valid_entries.append(entry)

        if embeddings:
            # Добавляем эмбеддинги в FAISS индекс
            self.index.add(np.array(embeddings, dtype=np.float32))
            self.knowledge_base = valid_entries  # Обновляем базу знаний только валидными записями
        else:
            logger.info("No valid embeddings to add to the index.")

        self.embeddings = np.array(embeddings, dtype=np.float32)
        
    async def search(self, query, k=1, threshold=1.5):
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
            logger.info(distances[0][i])
            if distances[0][i] <= threshold and idx < len(self.knowledge_base):
                entry = self.knowledge_base[idx]
                results.append({
                    "title": entry['title'],
                    "content": entry['content'],
                    "url": entry['url'],
                    "distance": distances[0][i]
                })

        return results