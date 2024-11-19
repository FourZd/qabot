import streamlit as st
from services.knowledge_service import KnowledgeService
from services.chat_service import ChatService
from services.embedding_service import EmbeddingService
from services.parser_service import ParserService
import asyncio
from openai import AsyncOpenAI
import os
from utils.logger import logger

openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

embedding_service = EmbeddingService(openai_client)
chat_service = ChatService(openai_client)
parser_service = ParserService()

BASE_URL = "https://www.promptingguide.ai/"


async def main():
    parsing_placeholder = st.empty()

    if "parsed_data" not in st.session_state:
        parsing_placeholder.write("Парсим данные с сайта...")
        with st.spinner("Идет сбор данных..."):
            parsed_data = await parser_service.fetch_and_parse_data(BASE_URL)
            st.session_state.parsed_data = parsed_data

        parsing_placeholder.empty()

    parsed_data = st.session_state.parsed_data

    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeService(
            knowledge_base=parsed_data, embedding_service=embedding_service
        )
        await st.session_state.kb.build_index()

    kb = st.session_state.kb

    st.title("AI QA Bot with Knowledge Base")
    st.markdown(
        "Задайте свой вопрос, и бот постарается ответить, используя знания из базы."
    )

    query = st.text_input("Введите ваш вопрос:")

    if query:
        searching_placeholder = st.empty()
        searching_placeholder.write("Ищем информацию в базе знаний...")
        with st.spinner("Обработка..."):
            results = await kb.search(query, k=1)
            logger.info(f"results found, {results}")
        searching_placeholder.empty()

        if results:

            article_title = results[0]["title"]
            article_context = results[0]["content"]
            article_url = results[0]["url"]

            st.markdown(f"### Найденный контекст: [{article_title}]({article_url})")
            st.write(article_context)
        else:
            article_context = None
            article_url = None
            st.warning("Контекст не найден в базе знаний.")

        st.write("Генерация ответа...")
        with st.spinner("Создаем ответ..."):
            answer = await chat_service.generate_answer(query, article_context)
        st.markdown("### Ответ:")
        st.write(answer)

        if article_url:
            st.markdown(f"**Источник:** [Открыть статью]({article_url})")


if __name__ == "__main__":
    asyncio.run(main())
