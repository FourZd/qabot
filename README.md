# AI QA Bot with Knowledge Base

## Описание

Это тестовый **AI-бот**, который помогает пользователю находить ответы на вопросы, используя заранее собранную базу знаний. База данных формируется из артиклей, автоматически извлечённых с указанного веб-сайта (например, *promptingguide.ai*). Бот отвечает на запросы, предоставляя релевантный контекст из базы знаний и генерируя ответ, даже если подобной информации нет у бота изначально.

---

## Ключевые функции

### 1. Сбор данных с сайта
- Использует асинхронный парсер для сбора статей, ссылок и их содержимого с указанного веб-сайта.
- Фильтрует недоступные или некорректные страницы (например, ошибки *404*).

### 2. Создание базы знаний
- Извлечённые данные структурируются в виде записей с заголовками, текстом и ссылками.
- Каждая запись преобразуется в векторное представление с помощью эмбеддингов OpenAI.

### 3. Поиск релевантного контекста
- Используется библиотека **FAISS** для быстрого поиска ближайших соседей в векторном пространстве.
- Находит записи из базы знаний, которые наиболее релевантны пользовательскому запросу.

### 4. Генерация ответов
- На основе контекста из базы знаний AI генерирует ответ, используя **OpenAI Chat API**.
- Если релевантный контекст не найден, бот честно сообщает об этом и старается дать ответ самостоятельно.

---

## Интерфейс пользователя

- Реализован с помощью **Streamlit**.
- Пользователь вводит свой вопрос в текстовое поле и получает структурированный ответ с указанием источника информации.

---

## Технологический стек

### Backend:
- Асинхронная обработка данных с помощью **asyncio** и **aiohttp**.
- Обработка естественного языка через **OpenAI API** (GPT и эмбеддинги).

### Интерфейс:
- **Streamlit** для быстрого создания UI.
- Простое, понятное взаимодействие с пользователем.

### Поиск:
- **FAISS** (Facebook AI Similarity Search) для работы с векторными эмбеддингами и поиска ближайших соседей.

### Парсинг данных:
- **BeautifulSoup** для извлечения содержимого HTML-страниц.
- Асинхронный краулер для обхода ссылок и извлечения данных.

---

## Основной сценарий работы

### 1. Инициализация:
- Бот загружает и индексирует базу знаний при первом запуске.
- Создаёт векторный индекс всех заголовков и текстов статей.

### 2. Поиск ответа:
- Пользователь вводит запрос.
- Бот ищет релевантные записи в базе знаний с помощью **FAISS**.
- Находит наиболее подходящую статью и предоставляет контекст.

### 3. Генерация ответа:
- Контекст из базы знаний передаётся в модель OpenAI для генерации ответа.
- Ответ показывается пользователю вместе с источником информации.

---

## Запуск

### Через Poetry:
```
poetry install poetry run streamlit run main.py
```

### Через Docker:
```
docker compose up --build
```

После этого сервис будет доступен по адресу: **http://0.0.0.0:8501/**
