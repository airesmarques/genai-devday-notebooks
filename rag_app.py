# rag_app.py
# Command-line RAG application using MongoDB and Serverless LLM.

import os
import argparse
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import requests

# Name for vector search index
INDEX_NAME = "vector_index"


def prepare(collection):
    # Load dataset
    print("Loading dataset...")
    data = load_dataset("mongodb/mongodb-docs", split="train")
    docs = pd.DataFrame(data).to_dict("records")

    # Chunk documents
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        separators=["\n\n", "\n", " ", "", "#", "##", "###"],
        chunk_size=200,
        chunk_overlap=30,
    )
    split_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.get("body", ""))
        for chunk in chunks:
            temp = doc.copy()
            temp["body"] = chunk
            split_docs.append(temp)
    print(f"Generated {len(split_docs)} chunks")

    # Embed documents
    print("Embedding documents...")
    embedding_model = SentenceTransformer("thenlper/gte-small")
    def get_embedding(text):
        return embedding_model.encode(text).tolist()

    embedded_docs = []
    for doc in split_docs:
        doc["embedding"] = get_embedding(doc["body"])
        embedded_docs.append(doc)

    # Ingest into MongoDB
    print("Ingesting into MongoDB...")
    collection.delete_many({})
    collection.insert_many(embedded_docs)
    count = collection.count_documents({})
    print(f"Ingested {count} documents")

    # Create vector search index
    print("Creating vector search index...")
    model = {
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": len(embedded_docs[0]["embedding"]),
                    "similarity": "cosine",
                }
            ]
        },
    }
    try:
        collection.drop_search_index(name=INDEX_NAME)
    except Exception:
        pass
    collection.create_search_index(model=model)
    print("Index created")
    return INDEX_NAME


def vector_search(collection, user_query, index_name, k=5, num_candidates=150):
    print("Performing vector search...")
    embedding_model = SentenceTransformer("thenlper/gte-small")
    query_embedding = embedding_model.encode(user_query).tolist()
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "queryVector": query_embedding,
                "path": "embedding",
                "k": k,
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


def generate_answer(collection, index_name, user_query, serverless_url):
    docs = vector_search(collection, user_query, index_name)
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

    parser = argparse.ArgumentParser(description="RAG CLI application")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip data ingestion and index creation (assumes collection is already prepared)",
    )
    parser.add_argument(
        "question",
        help="Question to ask the RAG application",
    )
    args = parser.parse_args()

    client = MongoClient(mongodb_uri, appname="rag-cli")
    db = client["mongodb_genai_devday_rag"]
    collection = db["knowledge_base"]

    if args.skip_prepare:
        index_name = INDEX_NAME
    else:
        index_name = prepare(collection)

    print("\nAnswer:")
    answer = generate_answer(collection, index_name, args.question, serverless_url)
    print(answer)

if __name__ == "__main__":
    main()
