# test_embedder.py

from app.embedder import process_text_to_faiss

sample_text = """Your clean text goes here. It can be extracted from PDF, DOCX, or CSV, already handled in Streamlit."""
process_text_to_faiss(sample_text)
