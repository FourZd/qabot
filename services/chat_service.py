from openai import AsyncOpenAI


class ChatService:
    def __init__(self, openai_client: AsyncOpenAI):
        """
        Сервис для генерации ответов с помощью OpenAI Chat API.
        :param openai_client: Асинхронный клиент OpenAI.
        """
        self.openai_client = openai_client

    async def generate_answer(self, query, context=None):
        """
        Генерация ответа с использованием OpenAI Chat Completion API.
        """
        system_message = "You are an AI QA bot. Answer user queries using relevant context if available. If not, try to answer by yourself, but say that you don't have the answer in your knowledge base."
        messages = [{"role": "system", "content": system_message}]

        if context:
            messages.append({"role": "system", "content": f"Relevant context: {context}"})

        messages.append({"role": "user", "content": query})

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=150
        )
        return response.choices[0].message.content
