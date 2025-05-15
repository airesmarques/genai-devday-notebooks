# query_rag.py
# Query application for the RAG system

import os
import argparse
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import requests

# Name for vector search index
INDEX_NAME = "vector_index"

def vector_search(collection, user_query, k=5, num_candidates=150):
    print("Performing vector search...")
    embedding_model = SentenceTransformer("thenlper/gte-small")
    query_embedding = embedding_model.encode(user_query).tolist()
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "queryVector": query_embedding,
                "path": "embedding",
                "limit": k,
                "numCandidates": num_candidates,
            }
        },
        {"$project": {"_id": 0, "body": 1}},
    ]
    results = collection.aggregate(pipeline)
    docs = list(results)
    print(f"Retrieved {len(docs)} documents")
    return docs

def create_prompt(documents, user_query):
    context = "\n\n".join([doc.get("body", "") for doc in documents])
    prompt = (
        "Answer the question based only on the following context. "
        "If the context is empty, say I DON'T KNOW\n\n"
        f"Context:\n{context}\n\nQuestion: {user_query}"
    )
    return prompt

def generate_answer(collection, user_query, serverless_url):
    docs = vector_search(collection, user_query)
    prompt = create_prompt(docs, user_query)
    messages = [{"role": "user", "content": prompt}]
    print("Requesting answer from LLM...")
    response = requests.post(
        serverless_url,
        json={"task": "completion", "data": messages},
    )
    response.raise_for_status()
    data = response.json()
    return data.get("text", "")

def main():
    load_dotenv()
    mongodb_uri = os.getenv("MONGODB_URI")
    serverless_url = os.getenv("SERVERLESS_URL")
    if not mongodb_uri or not serverless_url:
        print("Error: MONGODB_URI and SERVERLESS_URL must be set in .env")
        exit(1)

    parser = argparse.ArgumentParser(description="RAG Query Application")
    parser.add_argument(
        "question",
        help="Question to ask the RAG application",
        nargs="?",  # Make argument optional
    )
    args = parser.parse_args()

    client = MongoClient(mongodb_uri, appname="rag-query")
    db = client["mongodb_genai_devday_rag"]
    collection = db["knowledge_base"]

    # Check if collection exists and has data
    count = collection.count_documents({})
    if count == 0:
        print("Error: Knowledge base is empty. Run prepare_rag.py first.")
        exit(1)

    # Check if index exists
    indexes = collection.list_search_indexes()
    if not any(index.get("name") == INDEX_NAME for index in indexes):
        print(f"Error: Vector index '{INDEX_NAME}' not found. Run prepare_rag.py first.")
        exit(1)

    # Interactive mode if no question is provided
    if not args.question:
        print("RAG Query Interactive Mode (type 'exit' to quit)")
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == "exit":
                break
            print("\nAnswer:")
            answer = generate_answer(collection, question, serverless_url)
            print(answer)
    else:
        # One-off query mode
        print("\nAnswer:")
        answer = generate_answer(collection, args.question, serverless_url)
        print(answer)

if __name__ == "__main__":
    main()