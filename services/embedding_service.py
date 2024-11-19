from openai import AsyncOpenAI
import numpy as np
import os


class EmbeddingService:
    def __init__(self, openai_client: AsyncOpenAI):
        """
        Сервис для работы с эмбеддингами OpenAI.
        :param openai_client: Асинхронный клиент OpenAI.
        """
        self.openai_client = openai_client

    async def get_embedding(self, text, model="text-embedding-3-small"):
        """
        Получение эмбеддинга для текста.
        """
        response = await self.openai_client.embeddings.create(input=text, model=model)
        return response.data[0].embedding