import openai
import faiss
import numpy as np
import json
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

def get_embedding(text):
    print("Getting embedding of the text using text-embedding-3-small model")
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    print("Got the response of the embedding model! Returning it as an np array")
    return np.array(response.data[0].embedding)

dimension = 1536  # Dimension of text-embedding-3-small from openai docs
index = faiss.IndexFlatL2(dimension)

embeddings = []
for entry in knowledge_base:
    # В реальном проекте, было бы необходимо хранить эмбеддинги в постоянной бд, дабы не тратить токены зря
    # и избежать лишней нагрузки. Но тут тестовый проект, сами понимаете, время не бесконечное :) 
    embedding = get_embedding(entry['content'])
    print("Knowledge base embeddings are", embedding, "appending to emb list...")
    embeddings.append(embedding)
    index.add(np.array([embedding]))

knowledge_embeddings = np.array(embeddings)

def search_knowledge_base(query, k=1, threshold=0.7):
    print("Getting an embedding for the query")
    query_embedding = get_embedding(query)
    print("Query embedding", query_embedding)
    distances, indices = index.search(np.array([query_embedding]), k)
    print("Distances indices", distances, indices)
    results = []
    for i in range(k):
        if distances[0][i] >= threshold:
            print("Distances are above threshold, returning context from the base")
            entry = knowledge_base[indices[0][i]]
            results.append({
                "title": entry['title'],
                "content": entry['content'],
                "distance": distances[0][i]
            })
        else:
            print("Distances are under threshold, ignoring...", distances[0][i], threshold)
    return results

def generate_answer(query):
    results = search_knowledge_base(query, k=1)
    if results:
        print("Knowledge found in the knowledge base! Let's add it to the bot context")
        context = results[0]['content']
        print("Creating completion with context")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You're an QA bot. You will get user questions and should simply return the answer. Here some information that could be related to user question. If it's helpful, use it in the answer: {context}"},
                {"role": "user", "content": f"User answer is: {query}"}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
    else:
        print("Uh oh! Context not found. Hopely, chatgpt will answer it... Somehow.")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You're an QA bot. You will get user questions and should simply return the answer."},
                {"role": "user", "content": f"User answer is: {query}"}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content

query = "Chain-of-Thought Prompting"
answer = generate_answer(query)
print(answer)
