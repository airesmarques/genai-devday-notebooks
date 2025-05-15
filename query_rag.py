# query_rag.py
# Query application for the RAG system

import os
import argparse
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
from datetime import datetime

# Name for vector search index
INDEX_NAME = "vector_index"

# Initialize the re-ranking model after your embedding model
rerank_model = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")

def vector_search(collection, user_query, k=10, num_candidates=150):
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
        {"$project": {"_id": 0, "body": 1, "source": 1, "title": 1}},
    ]
    results = collection.aggregate(pipeline)
    docs = list(results)
    print(f"Retrieved {len(docs)} documents")
    return docs

def load_instructions():
    """Load custom instructions from file."""
    try:
        with open("chatbot_instructions.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("Warning: Instructions file not found. Using default instructions.")
        return (
            "You are a helpful customer service chatbot named Assistente PagaLava. "
            "Only answer questions related to laundry services."
        )

def create_prompt(documents, user_query):
    """Create a prompt with instructions and context."""
    instructions = load_instructions()
    
    # Extract document bodies for re-ranking
    doc_bodies = [doc.get("body", "") for doc in documents]
    
    # Re-rank documents if we have more than 1
    if len(doc_bodies) > 1:
        print("Re-ranking documents...")
        reranked_results = rerank_model.rank(
            user_query, 
            doc_bodies,
            return_documents=True, 
            top_k=min(5, len(doc_bodies))
        )
        context = "\n\n".join([d.get("text", "") for d in reranked_results])
    else:
        # No need to re-rank if only 1 or 0 documents
        context = "\n\n".join(doc_bodies)
    
    prompt = (
        f"{instructions}\n\n"
        f"Answer the question based only on the following context.\n"
        f"If the context doesn't contain relevant information, respond in Portuguese: "
        f"\"Não tenho essa informação específica, mas posso ajudar com outras perguntas sobre os serviços da PagaLava.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {user_query}"
    )
    return prompt

def is_on_topic(query):
    """Basic check to determine if query is on-topic for a laundromat chatbot."""
    laundry_keywords = [
        # English keywords
        "laundry", "wash", "dry", "clean", "detergent", "stain", "fold", 
        "machine", "dryer", "service", "hours", "price", "cost",
        "location", "pagalava", "payment", "clothes", "fabric", "lavanderia",
        
        # Portuguese keywords from FAQ
        "lavagem", "secagem", "demora", "tempo", "detergente", "amaciador", 
        "roupa", "máquina", "erro", "porta", "abertos", "horas", "espuma", 
        "ecológicos", "pagamento", "cartão", "engomadoria", "recolha", 
        "entrega", "ciclos", "vincos", "esqueci", "câmaras", "linha vermelha",
        "tira-nódoas", "higienizante", "loja", "rua", "brasil", "vasconcelos"
    ]
    
    query_lower = query.lower()
    
    # Simple keyword matching - can be enhanced with more sophisticated methods
    for keyword in laundry_keywords:
        if keyword in query_lower:
            return True
    
    # If no keywords match, we'll still let it through to the RAG system
    # which will use the instruction file to guide its response
    return True

# Add these functions for message history

def store_chat_message(collection, session_id, role, content):
    """Store a chat message in MongoDB."""
    message = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    }
    collection.insert_one(message)
    print(f"Stored {role} message for session {session_id}")

def retrieve_session_history(collection, session_id):
    """Retrieve chat history for a session."""
    cursor = collection.find({"session_id": session_id}).sort("timestamp", 1)
    if cursor:
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in cursor]
        print(f"Retrieved {len(messages)} previous messages for session {session_id}")
        return messages
    return []

# Now modify the generate_answer function to use memory
def generate_answer(db, collection, user_query, serverless_url, session_id="default"):
    """Generate answer with conversation memory."""
    # Make sure we have a history collection
    history_collection = db["chat_history"]
    
    # Create index on session_id if it doesn't exist
    if "session_id" not in history_collection.index_information():
        history_collection.create_index("session_id")
    
    # Get chat history
    messages = []
    
    # First, check if this is on-topic
    if not is_on_topic(user_query):
        print("Off-topic query detected, but will let instructions guide the model")
    
    # Get relevant documents
    docs = vector_search(collection, user_query)
    
    # Create context prompt
    context_prompt = create_prompt(docs, user_query)
    
    # Instead of system message, use user message for the context/instructions
    # This is compatible with more LLM APIs
    user_context_message = {"role": "user", "content": context_prompt}
    messages.append(user_context_message)
    
    # Add a mock assistant response to maintain the conversation flow
    messages.append({"role": "assistant", "content": "I'll help you with that question."})
    
    # Add message history from previous interactions
    message_history = retrieve_session_history(history_collection, session_id)
    messages.extend(message_history)
    
    # Add current user question
    user_message = {"role": "user", "content": user_query}
    messages.append(user_message)
    
    # Get response from LLM
    print("Requesting answer from LLM...")
    response = requests.post(
        serverless_url,
        json={"task": "completion", "data": messages},
    )
    response.raise_for_status()
    data = response.json()
    answer = data.get("text", "")
    
    # Store the conversation
    store_chat_message(history_collection, session_id, "user", user_query)
    store_chat_message(history_collection, session_id, "assistant", answer)
    
    return answer

def main():
    load_dotenv()
    mongodb_uri = os.getenv("MONGODB_URI")
    serverless_url = os.getenv("SERVERLESS_URL")
    if not mongodb_uri or not serverless_url:
        print("Error: MONGODB_URI and SERVERLESS_URL must be set in .env")
        exit(1)

    parser = argparse.ArgumentParser(description="PagaLava Chatbot")
    parser.add_argument(
        "question",
        help="Question to ask the chatbot",
        nargs="?",  # Make argument optional
    )
    parser.add_argument(
        "--session", "-s",
        help="Session ID for conversation memory (default: default)",
        default="default"
    )
    args = parser.parse_args()

    client = MongoClient(mongodb_uri, appname="pagalava-chatbot")
    db = client["mongodb_genai_devday_rag"]
    collection = db["knowledge_base"]

    # Check if collection exists and has data
    count = collection.count_documents({})
    if count == 0:
        print("Error: Knowledge base is empty. Run prepare_rag_pagalava.py first.")
        exit(1)

    # Check if index exists
    indexes = collection.list_search_indexes()
    if not any(index.get("name") == INDEX_NAME for index in indexes):
        print(f"Error: Vector index '{INDEX_NAME}' not found. Run prepare_rag_pagalava.py first.")
        exit(1)

    # Interactive mode if no question is provided
    if not args.question:
        print(f"Assistente PagaLava - Sessão: {args.session}")
        print("Digite 'sair' para encerrar o chat.")
        while True:
            question = input("\nComo posso ajudar com suas necessidades de lavanderia? ")
            if question.lower() in ["sair", "exit"]:
                break
            print("\nAssistente PagaLava:")
            answer = generate_answer(db, collection, question, serverless_url, args.session)
            print(answer)
    else:
        # One-off query mode
        print("\nAssistente PagaLava:")
        answer = generate_answer(db, collection, args.question, serverless_url, args.session)
        print(answer)

if __name__ == "__main__":
    main()