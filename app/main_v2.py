import streamlit as st
import os
import uuid
import pandas as pd
import docx
import openpyxl
import logging
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import re
import uuid

# --- Load Environment ---
load_dotenv()
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Configure ---
openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)
logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def get_embedding(text):
    return openai_embeddings.embed_query(text)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_excel(file):
    wb = openpyxl.load_workbook(file, read_only=True)
    sheet = wb.active
    data = list(sheet.values)
    return "\n".join(["\t".join([str(cell) for cell in row]) for row in data])

def extract_text(file, filename):
    if filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif filename.endswith(".csv"):
        df = pd.read_csv(file)
        return df.to_string(index=False)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file)
    elif filename.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif filename.endswith(".xlsx"):
        return extract_text_from_excel(file)
    else:
        return ""

def sanitize_key(s):
    return re.sub(r'[^A-Za-z0-9_\-=]', '_', s)

def upload_chunks_to_azure(chunks, source_name):
    documents = []
    safe_uuid = uuid.uuid4().hex  # generate a unique identifier per upload
    safe_source_name = sanitize_key(source_name)

    for i, chunk in enumerate(chunks):
        embedding_vector = get_embedding(chunk)
        doc_id = f"{safe_source_name}_{i}_{safe_uuid}"
        doc = {
            "id": doc_id,
            "content": chunk,
            "embedding": embedding_vector,
            "source_filename": source_name
        }
        documents.append(doc)

    try:
        results = search_client.upload_documents(documents=documents)
        succeeded = sum(1 for r in results if r.succeeded)
        st.success(f"✅ Uploaded {succeeded}/{len(documents)} chunks to Azure Cognitive Search.")
    except Exception as e:
        st.error(f"❌ Error uploading documents: {e}")

def search_azure_vector_index(query, top_k=5):
    try:
        query_embedding = get_embedding(query)
        results = search_client.search(
            search_text="*",  # wildcard for vector-only search
            vector=query_embedding,
            top=top_k,
            vector_fields="embedding"
        )
        return [(doc["content"], doc.get("@search.score")) for doc in results]
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# --- Streamlit UI ---

st.title("📂 Upload Any Document for Azure Vector Search")

uploaded_files = st.file_uploader(
    "Upload .pdf, .docx, .csv, .xlsx, or .txt files",
    type=["pdf", "docx", "csv", "xlsx", "txt"],
    accept_multiple_files=True
)

for file in uploaded_files:
    filename = file.name
    st.markdown(f"### 📄 Processing: {filename}")
    try:
        extracted_text = extract_text(file, filename)
        if not extracted_text.strip():
            st.warning(f"⚠️ No text could be extracted from `{filename}`.")
            continue

        st.text_area("Extracted Text", extracted_text[:3000], height=200)
        if st.button(f"🔍 Upload `{filename}` to Azure Search", key=filename):
            chunks = chunk_text(extracted_text)
            st.write(f"Split into {len(chunks)} chunks.")
            upload_chunks_to_azure(chunks, filename)
    except Exception as e:
        st.error(f"❌ Failed to process `{filename}`: {e}")

query = st.text_input("🔎 Ask a question about your documents:")

if query:
    results = search_azure_vector_index(query)
    if results:
        st.subheader("Search Results")
        for i, (content, score) in enumerate(results, start=1):
            st.markdown(f"**Result {i} (Score: {score:.4f})**")
            st.write(content)
    else:
        st.info("No matching content found.")
