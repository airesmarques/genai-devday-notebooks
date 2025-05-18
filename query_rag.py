# query_rag.py
# Query application for the RAG system

import os
import argparse
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
from datetime import datetime
import openai
import re
from visense_api import ViSenseAPI
import uuid

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
    """Check if the query is on-topic for the laundromat domain."""
    laundry_keywords = [
        "lavanderia", "lavandaria", "máquina", "maquina", "lavar", "secar", "secadora", 
        "lavadora", "detergente", "pagamento", "cartão", "cartao", "cesto", "moedas", 
        "roupa", "roupas", "porta", "machine", "valor", "preço", "preco", "tempo", "pagalava"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in laundry_keywords)

def is_technical_issue(query):
    """Check if query is about a technical issue that may require intervention."""
    issue_keywords = [
        "não funciona", "nao funciona", "erro", "error", "problema", "avariada", 
        "não liga", "nao liga", "bloqueada", "travada", "stuck", "porta não abre",
        "porta nao abre", "não abre", "nao abre", "a02", "a03", "código", "codigo",
        "não começa", "nao comeca", "parou", "não aceita", "nao aceita"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in issue_keywords)

def has_required_info(query, session_memory=None):
    """Check if we have necessary information to take action."""
    # Extract information from query
    laundry_id, machine_number = extract_machine_info(query)
    
    # If session memory exists, use it to supplement missing info
    if not laundry_id and session_memory and session_memory.get("laundry_id"):
        laundry_id = session_memory.get("laundry_id")
    
    if not machine_number and session_memory and session_memory.get("machine_number"):
        machine_number = session_memory.get("machine_number")
    
    # Return a tuple of (has_all_info, laundry_id, machine_number)
    return (bool(laundry_id and machine_number), laundry_id, machine_number)

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

# Add this function to recognize machine references
def extract_machine_info(query):
    """Extract laundry ID and machine number from user query."""
    # Default to L1 if no specific laundry mentioned
    laundry_id = "L1"
    
    # Check if L2/Avenida is explicitly mentioned
    if any(location in query.lower() for location in ["l2", "avenida", "antónio vasconcelos", "antonio vasconcelos"]):
        laundry_id = "L2"
    elif any(location in query.lower() for location in ["l1", "brasil", "rua do brasil"]):
        laundry_id = "L1"
        
    # Try to extract machine number - look for digits after words like "machine", "máquina", etc.
    machine_match = re.search(r'(?:máquina|maquina|machine|número|numero|number)\s*(?:de\s*(?:lavar|secar)\s*)?(?:número|numero|number)?\s*(\d+)', 
                           query.lower())
    
    machine_number = None
    if machine_match:
        machine_number = machine_match.group(1)
    
    return laundry_id, machine_number

# Add a function to detect door issues
def is_door_issue(query):
    """Check if query is about door not opening."""
    door_keywords = [
        "porta não abre", "porta nao abre", "não consigo abrir", "nao consigo abrir",
        "porta presa", "porta bloqueada", "porta fechada", "door", "porta", 
        "não abre", "nao abre", "stuck", "bloqueada", "travada"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in door_keywords)

def should_reboot(history, user_query):
    """
    Determine if the user has previously reported a door issue and now confirms reboot.
    Returns (should_reboot: bool, laundry_id, machine_number)
    """
    confirm_keywords = ["sim", "yes", "confirmo", "reiniciar", "reboot", "reset", "ok"]
    # Check if user_query is a confirmation
    if any(word in user_query.lower() for word in confirm_keywords):
        # Look for the most recent door issue in the history
        for msg in reversed(history):
            if msg["role"] == "user" and is_door_issue(msg["content"]):
                laundry_id, machine_number = extract_machine_info(msg["content"])
                if machine_number:
                    return True, laundry_id, machine_number
    return False, None, None

def store_session_memory(collection, session_id, laundry_id=None, machine_number=None, error_code=None):
    """Store or update session memory with important info."""
    update = {
        "session_id": session_id,
        "laundry_id": laundry_id,
        "machine_number": machine_number,
        "error_code": error_code,
        "updated_at": datetime.now()
    }
    # Only update fields that are not None
    update = {k: v for k, v in update.items() if v is not None or k in ["session_id", "updated_at"]}
    collection.update_one(
        {"session_id": session_id},
        {"$set": update},
        upsert=True
    )

def retrieve_session_memory(collection, session_id):
    """Retrieve session memory for a session."""
    doc = collection.find_one({"session_id": session_id})
    if doc:
        return {
            "laundry_id": doc.get("laundry_id"),
            "machine_number": doc.get("machine_number"),
            "error_code": doc.get("error_code")
        }
    return {}

# Modify generate_answer to handle machine actions
def generate_answer(db, collection, user_query, openai_api_key, session_id="default", model="gpt-4o"):
    """Generate answer with conversation memory and API actions."""
    history_collection = db["chat_history"]
    session_mem_collection = db["session_memory"]

    if "session_id" not in history_collection.index_information():
        history_collection.create_index("session_id")
    if "session_id" not in session_mem_collection.index_information():
        session_mem_collection.create_index("session_id")

    message_history = retrieve_session_history(history_collection, session_id)
    session_memory = retrieve_session_memory(session_mem_collection, session_id)

    # Check if this is a door issue
    if is_door_issue(user_query):
        laundry_id, machine_number = extract_machine_info(user_query)
        store_session_memory(session_mem_collection, session_id, laundry_id=laundry_id, machine_number=machine_number)
        if machine_number:
            # Ask for confirmation to reboot
            answer = (
                f"Detectei que a porta da máquina {machine_number} ({laundry_id}) está com problema. "
                "Deseja que eu tente reiniciar a máquina remotamente? Responda 'sim' para confirmar."
            )
            store_chat_message(history_collection, session_id, "user", user_query)
            store_chat_message(history_collection, session_id, "assistant", answer)
            return answer

    # Check if user is confirming reboot after a door issue
    reboot, laundry_id, machine_number = should_reboot(message_history, user_query)
    if reboot:
        # Use session memory if missing info
        if not laundry_id or not machine_number:
            mem = retrieve_session_memory(session_mem_collection, session_id)
            laundry_id = laundry_id or mem.get("laundry_id")
            machine_number = machine_number or mem.get("machine_number")
        try:
            api = ViSenseAPI()
            result = api.reboot_machine(laundry_id, machine_number)
            if result.get("status") == "success":
                action_response = (
                    f"✅ Máquina {machine_number} da lavandaria {laundry_id} reiniciada com sucesso! "
                    "Por favor, tente abrir a porta novamente. Caso não consiga, aguarde 1 minuto e tente novamente."
                )
            else:
                action_response = (
                    f"❌ Não foi possível reiniciar a máquina {machine_number}. "
                    f"Erro: {result.get('message', 'desconhecido')}. Por favor, tente seguir as instruções manuais ou contacte-nos."
                )
        except Exception as e:
            action_response = (
                f"❌ Ocorreu um erro ao tentar reiniciar a máquina: {str(e)}. "
                "Por favor, tente seguir as instruções manuais ou contacte-nos."
            )
        store_chat_message(history_collection, session_id, "user", user_query)
        store_chat_message(history_collection, session_id, "assistant", action_response)
        return action_response

    # ...existing RAG flow...
    docs = vector_search(collection, user_query)
    context_prompt = create_prompt(docs, user_query)
    messages = [{"role": "system", "content": context_prompt}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": user_query})

    client = openai.OpenAI(api_key=openai_api_key)
    print("Requesting answer from GPT-4...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    answer = response.choices[0].message.content

    store_chat_message(history_collection, session_id, "user", user_query)
    store_chat_message(history_collection, session_id, "assistant", answer)
    return answer

def main():
    load_dotenv()
    mongodb_uri = os.getenv("MONGODB_URI")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not mongodb_uri or not openai_api_key:
        print("Error: MONGODB_URI and OPENAI_API_KEY must be set in .env")
        exit(1)

    parser = argparse.ArgumentParser(description="PagaLava Chatbot")
    parser.add_argument(
        "question",
        help="Question to ask the chatbot",
        nargs="?",  # Make argument optional
    )
    parser.add_argument(
        "--session", "-s",
        help="Session ID for conversation memory (default: generate new)",
        default=None
    )
    parser.add_argument(
        "--model", "-m",
        help="OpenAI model to use (default: gpt-4o)",
        default="gpt-4o"
    )
    args = parser.parse_args()

    # Generate a unique session ID if not provided
    if args.session is None:
        session_id = str(uuid.uuid4())
        print(f"New session created with ID: {session_id}")
    else:
        session_id = args.session

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
        print(f"Assistente PagaLava - Sessão: {session_id} (usando {args.model})")
        print("Digite 'sair' para encerrar o chat.")
        while True:
            question = input("\nComo posso ajudar com suas necessidades de lavanderia? ")
            if question.lower() in ["sair", "exit"]:
                break
            print("\nAssistente PagaLava:")
            answer = generate_answer(db, collection, question, openai_api_key, session_id, args.model)
            print(answer)
    else:
        # One-off query mode
        print("\nAssistente PagaLava:")
        answer = generate_answer(db, collection, args.question, openai_api_key, session_id, args.model)
        print(answer)

if __name__ == "__main__":
    main()