# prepare_rag.py
# Data preparation script for the RAG application

import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Name for vector search index
INDEX_NAME = "vector_index"

def prepare_data():
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
    
    return embedded_docs

def create_vector_index(collection, embedded_docs):
    # Create vector search index
    print("Creating vector search index...")
    
    # Check if Atlas search is available on this cluster
    try:
        # Get database command info to verify capabilities
        db_info = collection.database.command("buildInfo")
        print(f"Connected to MongoDB version: {db_info.get('version', 'unknown')}")
        
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
        
        # Try to drop existing index first
        try:
            collection.drop_search_index(name=INDEX_NAME)
            print(f"Dropped existing index '{INDEX_NAME}'")
        except Exception as e:
            if "not found" in str(e).lower():
                print(f"No existing index '{INDEX_NAME}' found")
            else:
                print(f"Warning when dropping index: {str(e)}")
        
        # Create the index
        collection.create_search_index(model=model)
        print(f"Vector search index '{INDEX_NAME}' created successfully")
        
    except Exception as e:
        print(f"Error creating vector search index: {str(e)}")
        print("Make sure your Atlas cluster supports Vector Search (M10 or higher tier)")
        exit(1)
        
    return INDEX_NAME

def main():
    load_dotenv()
    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        print("Error: MONGODB_URI must be set in .env")
        exit(1)

    # Connect to Atlas with appropriate timeout
    try:
        client = MongoClient(mongodb_uri, 
                            appname="rag-data-preparation",
                            serverSelectionTimeoutMS=5000)
        # Verify connection is working
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas")
    except Exception as e:
        print(f"Error connecting to MongoDB Atlas: {str(e)}")
        print("Please check your connection string and network connectivity")
        exit(1)
        
    db = client["mongodb_genai_devday_rag"]
    collection = db["knowledge_base"]
    
    # Check if collection already has data
    count = collection.count_documents({})
    if count > 0:
        prompt = input(f"Collection already contains {count} documents. Replace? (y/n): ")
        if prompt.lower() != 'y':
            print("Preparation cancelled.")
            return
    
    # Process data
    embedded_docs = prepare_data()
    
    # Ingest into MongoDB
    print("Ingesting into MongoDB...")
    collection.delete_many({})
    collection.insert_many(embedded_docs)
    count = collection.count_documents({})
    print(f"Ingested {count} documents")
    
    # Create index
    create_vector_index(collection, embedded_docs)
    print("Preparation complete! The data is ready for querying.")

if __name__ == "__main__":
    main()