# app/embedder.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)

# Step 1: Chunking
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Step 2: Embedding and saving
def embed_and_store(chunks, index_path="data/faiss_index", metadata=None):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    if metadata:
        metadatas = [metadata for _ in chunks]
        new_db = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    else:
        new_db = FAISS.from_texts(chunks, embeddings)

    if os.path.exists(index_path + "/index.faiss"):
        try:
            existing_db = FAISS.load_local(index_path, embeddings)
            existing_db.merge_from(new_db)
            db = existing_db
            logging.info("🧩 Merged with existing FAISS index.")
        except Exception as e:
            logging.error(f"❌ Failed to load existing index: {e}")
            db = new_db
    else:
        db = new_db

    db.save_local(index_path)
    logging.info(f"✅ FAISS index saved to: {index_path}")

# Combined processing function
def process_text_to_faiss(text, index_path="data/faiss_index", source_name=None):
    logging.info("🔹 Chunking text...")
    chunks = chunk_text(text)
    logging.info(f"🔹 {len(chunks)} chunks created.")

    metadata = {"source": source_name} if source_name else None
    logging.info("🔹 Embedding and saving to FAISS...")
    embed_and_store(chunks, index_path=index_path, metadata=metadata)

def query_faiss_index(query, index_path="data/faiss_index", k=5):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    try:
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load FAISS index at {index_path}: {e}")

    results = db.similarity_search_with_score(query, k=k)
    return results

